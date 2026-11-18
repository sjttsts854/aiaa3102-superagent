import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from superagent.generated_code import CodePolicyError, GeneratedModel, run_generated_candidate, validate_source
from superagent.runner import snapshot_source


SOURCE = '''import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SquaredFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.column_stack([values, values ** 2])

def build_model(X_train, y_train, seed):
    assert list(X_train.columns) == ['column_0', 'column_1']
    return make_pipeline(SquaredFeatures(), StandardScaler(), SVC(kernel='rbf', probability=True, random_state=seed))
'''


@pytest.fixture
def task(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(90, 2))
    frame = pd.DataFrame(values, columns=['private_name', 'other_name'])
    frame['label'] = (np.sum(values ** 2, axis=1) > 1.3).astype(int)
    frame.iloc[:70].to_csv(tmp_path/'train.csv', index=False)
    frame.iloc[70:].to_csv(tmp_path/'valid.csv', index=False)
    run = tmp_path/'captured'
    run.mkdir()
    snapshot_source(run)
    path = run/'task_snapshot.json'
    path.write_text(json.dumps({'data_type':'tabular', 'train_path':str(tmp_path/'train.csv'),
                               'validation_path':str(tmp_path/'valid.csv'), 'target_column':'label'}))
    return path, frame.iloc[70:].drop(columns='label')


def candidate(source=SOURCE):
    return {'algorithm':'generated_python', 'name':'custom_svc_features', 'source':source}


def test_new_algorithm_custom_feature_serialization_and_fresh_process(task, tmp_path):
    path, X = task
    folder = tmp_path/'trial'
    result = run_generated_candidate(path, candidate(), folder, 35)
    assert result['status'] == 'success', result
    assert result['score'] > .7
    assert result['validation_labels_passed_to_generated_code'] is False
    assert result['runtime_is_os_security_sandbox'] is False
    payload = joblib.load(folder/'worker/input.joblib')
    assert set(payload) == {'mode','source','seed','X_train','y_train','prediction_features'}
    assert set(payload['prediction_features']) == {'train','validation'}
    model = joblib.load(folder/'pipeline.joblib')
    assert isinstance(model, GeneratedModel)
    assert model.predict(X).shape == (20,)
    probability = model.predict_proba(X)
    assert probability.shape == (20,2) and np.allclose(probability.sum(axis=1),1)
    x_path = tmp_path/'fresh_input.joblib'
    joblib.dump(X, x_path)
    script = 'import joblib,sys; m=joblib.load(sys.argv[1]); x=joblib.load(sys.argv[2]); print(m.predict(x).tolist())'
    out = subprocess.run([sys.executable,'-c',script,str(folder/'pipeline.joblib'),str(x_path)],capture_output=True,text=True,check=True)
    assert json.loads(out.stdout.strip()) == model.predict(X).tolist()
    with pytest.raises(ValueError, match='schema'):
        model.predict(X.rename(columns={'private_name':'wrong'}))


@pytest.mark.parametrize('source,kind', [
    ('def build_model(X_train, y_train, seed)\n    return None','SyntaxError'),
    ('def build_model(X_train, y_train, seed):\n    return 1 / 0','ZeroDivisionError'),
    ('import os\ndef build_model(X_train, y_train, seed):\n    return None','CodePolicyError'),
])
def test_failures_return_repair_diagnostics(task,tmp_path,source,kind):
    result=run_generated_candidate(task[0],candidate(source),tmp_path/'failed',30)
    assert result['status']=='failed' and result['error_type']==kind, result
    assert 0 < len(result['error_feedback']) <=800
    assert str(tmp_path) not in result['error_feedback']


@pytest.mark.parametrize('body', [
    "open('secret.txt').read()", "__import__('os').environ", "np.load('secret.npy')",
    "pd.read_csv('secret.csv')", "getattr(np, 'load')('secret.npy')", "X_train.__class__",
])
def test_direct_access_rejected(body):
    with pytest.raises(CodePolicyError):
        validate_source('import numpy as np\nimport pandas as pd\ndef build_model(X_train,y_train,seed):\n    '+body+'\n    return None')


def test_indirect_scientific_file_read_blocked(task,tmp_path):
    secret=tmp_path/'secret.txt'
    secret.write_text('never expose this marker')
    source = "from sklearn.feature_extraction.text import CountVectorizer\ndef build_model(X_train,y_train,seed):\n    CountVectorizer(input='filename').fit(["+repr(str(secret))+"])\n    return None"
    result=run_generated_candidate(task[0],candidate(source),tmp_path/'file_attempt',30)
    assert result['status']=='failed' and result['error_type']=='CodePolicyError', result
    assert 'never expose' not in result['error_feedback']


def test_timeout_stops_generated_loop(task,tmp_path):
    source='def build_model(X_train,y_train,seed):\n    while True:\n        pass'
    result=run_generated_candidate(task[0],candidate(source),tmp_path/'timeout',3)
    assert result['error_type']=='TimeoutError', result


def test_invalid_predictions_rejected_in_parent(task,tmp_path):
    source='''import numpy as np
class BadModel:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        return np.zeros((len(X),2))
def build_model(X_train,y_train,seed):
    return BadModel()
'''
    result=run_generated_candidate(task[0],candidate(source),tmp_path/'bad_prediction',30)
    assert result['status']=='failed' and 'one-dimensional' in result['error_feedback'],result


def test_no_probability_classifier_remains_usable(task,tmp_path):
    source=SOURCE.replace("SVC(kernel='rbf', probability=True, random_state=seed)","SVC(kernel='rbf', random_state=seed)")
    result=run_generated_candidate(task[0],candidate(source),tmp_path/'no_probability',30)
    assert result['status']=='success',result
    model=joblib.load(tmp_path/'no_probability/pipeline.joblib')
    assert not hasattr(model,'predict_proba')
    assert model.predict(task[1]).shape==(20,)


def test_environment_secrets_not_inherited_and_state_guard(task,tmp_path,monkeypatch):
    import superagent.generated_code as runtime
    monkeypatch.setenv('AIGC_API_KEY','synthetic-private-key-for-test')
    original=runtime.subprocess.run
    captures=[]
    def capture(*args,**kwargs):
        captures.append(kwargs['env'])
        return original(*args,**kwargs)
    monkeypatch.setattr(runtime.subprocess,'run',capture)
    result=run_generated_candidate(task[0],candidate(),tmp_path/'isolated_env',30)
    assert result['status']=='success',result
    assert all('AIGC_API_KEY' not in environment and 'synthetic-private-key-for-test' not in environment.values() for environment in captures)
    model=joblib.load(tmp_path/'isolated_env/pipeline.joblib')
    assert model.python_version==tuple(sys.version_info[:3])
    model.state += b'changed'
    with pytest.raises(ValueError,match='state fingerprint'):
        model.predict(task[1])


def test_source_limit_uses_utf8_bytes():
    with pytest.raises(CodePolicyError,match='UTF-8'):
        validate_source('def build_model(X_train,y_train,seed):\n    pass\n#'+'字'*4000)


def test_network_import_not_permitted():
    with pytest.raises(CodePolicyError,match='Import'):
        validate_source('import socket\ndef build_model(X_train,y_train,seed):\n    return socket.socket()')


@pytest.mark.parametrize('hook', [
    'class Model:\n    __reduce__ = lambda self: None',
    "Model = type('Model', (), {'__setstate__': lambda self, state: None})",
])
def test_serialization_hook_assignments_not_permitted(hook):
    with pytest.raises(CodePolicyError, match='hooks'):
        validate_source(hook+'\ndef build_model(X_train,y_train,seed):\n    return Model()')


def test_custom_numpy_reduction_features_survive_reload(task, tmp_path):
    source = SOURCE.replace(
        'return np.column_stack([values, values ** 2])',
        'return np.column_stack([values, values ** 2, values.sum(axis=1), values.mean(axis=1), values.std(axis=1)])',
    )
    folder = tmp_path/'numpy_reductions'
    result = run_generated_candidate(task[0], candidate(source), folder, 30)
    assert result['status'] == 'success', result
    model = joblib.load(folder/'pipeline.joblib')
    assert model.predict(task[1]).shape == (20,)
    probability = model.predict_proba(task[1])
    assert probability.shape == (20, 2)
    assert np.allclose(probability.sum(axis=1), 1)


@pytest.mark.parametrize('statement', [
    'import numpy._core._methods',
    'from numpy._core import _methods',
])
def test_direct_numpy_private_imports_still_denied(statement):
    with pytest.raises(CodePolicyError, match='Import'):
        validate_source(statement+'\ndef build_model(X_train,y_train,seed):\n    return None')
