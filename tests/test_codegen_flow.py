"""Independent synthetic integration proof: generate, fail, repair, select, export.

Only the HTTP boundary is faked in the full-flow test. Training, scoring,
serialization, refitting and inference use the real runner and worker processes.
"""
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest

from superagent.generated_code import GeneratedModel
from superagent.runner import export_predictions, run
from superagent.usage import LLMConfig


REPAIRED_SOURCE = '''import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class InteractionFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.column_stack([values, values[:, 0] * values[:, 1]])

def build_model(X_train, y_train, seed):
    assert list(X_train.columns) == ['column_0', 'column_1']
    return make_pipeline(InteractionFeatures(), StandardScaler(), SVC(kernel='linear', C=10, probability=True, random_state=seed))
'''
BROKEN_SOURCE = REPAIRED_SOURCE.replace('make_pipeline(InteractionFeatures()', 'make_pipeline(MissingInteractionFeatures()')


@pytest.fixture
def synthetic_task(tmp_path):
    rng = np.random.default_rng(43)

    def make_frame(repeats):
        signs = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), (repeats, 1))
        values = signs * rng.uniform(.8, 1.2, size=signs.shape)
        frame = pd.DataFrame(values, columns=['private_feature_a', 'private_feature_b'])
        frame['label'] = (values[:, 0] * values[:, 1] > 0).astype(int)
        return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)

    train, validation, prediction = make_frame(24), make_frame(8), make_frame(3)
    train.to_csv(tmp_path / 'train.csv', index=False)
    validation.to_csv(tmp_path / 'validation.csv', index=False)
    prediction.drop(columns='label').to_csv(tmp_path / 'prediction.csv', index=False)
    path = tmp_path / 'task.json'
    path.write_text(json.dumps({
        'name': 'private_synthetic_task_name', 'data_type': 'tabular', 'target_column': 'label',
        'train_path': str(tmp_path / 'train.csv'), 'validation_path': str(tmp_path / 'validation.csv'),
        'predict_path': str(tmp_path / 'prediction.csv'), 'max_experiments': 4, 'seed': 42,
    }), encoding='utf-8')
    return path, prediction


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    config = LLMConfig(
        base_url='https://provider.example/v1', model='synthetic-code-model',
        api_key_env='CODEGEN_FLOW_FAKE_KEY', model_family='deepseek',
        model_release_date='2026-01-01', pricing_as_of='2026-10-21',
        input_rmb_per_million=2., output_rmb_per_million=8.,
        price_source='https://provider.example/prices', model_source='https://provider.example/model',
        verified=True,
    )
    monkeypatch.setenv(config.api_key_env, 'fake-key-for-local-test-only')
    path = tmp_path / 'fake_config.json'
    path.write_text(json.dumps(asdict(config)), encoding='utf-8')
    return path


def _plan(source, name):
    return {'action': 'experiment', 'candidate': {
        'algorithm': 'generated_python', 'name': name, 'source': source,
    }, 'reason': 'Add an interaction feature to represent the measured nonlinear structure.'}


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_real_runner_repairs_generated_code_selects_refits_and_reloads(synthetic_task, fake_config, tmp_path, monkeypatch):
    task_path, prediction = synthetic_task
    contexts = []
    responses = iter([
        _plan(BROKEN_SOURCE, 'interaction_svc_first_attempt'),
        _plan(REPAIRED_SOURCE, 'interaction_svc_repaired'),
        {'action': 'stop', 'reason': 'The repaired candidate has perfect held-out accuracy; stop.'},
    ])

    def fake_post(config, payload, api_key):
        assert api_key == 'fake-key-for-local-test-only'
        contexts.append(json.loads(payload['messages'][1]['content']))
        return httpx.Response(200, request=httpx.Request('POST', config.base_url), json={
            'model': config.model, 'usage': {'prompt_tokens': 100, 'completion_tokens': 50},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(next(responses))}}],
        })

    monkeypatch.setattr(importlib.import_module('superagent.planner'), 'post_completion', fake_post)
    output = tmp_path / 'run'
    summary = run(task_path, output, 'llm-code', fake_config, experiment_timeout=40, refit_timeout=50)

    assert summary['status'] == 'completed', summary
    assert summary['run_mode'] == 'llm-code'
    assert summary['llm_requests'] == summary['valid_llm_decisions'] == 3
    assert summary['generated_code_experiments'] == 2
    assert summary['successful_generated_code_experiments'] == 1
    assert summary['selected_generated_code'] is True
    assert summary['experiments_attempted'] == 3 and summary['experiments_completed'] == 2
    assert summary['invalid_plan_count'] == 0  # Runtime bugs are trials, not malformed planner responses.
    assert summary['planner_stop_reason'] == 'planner_requested_stop'
    assert summary['selected_trial'] == 3 and summary['best_validation_score'] == 1.0
    assert summary['refit_success'] is True and summary['official_test_evaluated'] is False
    assert summary['usage']['accounted_cost_rmb'] == pytest.approx(.0018)
    assert summary['accounting_complete'] is True

    experiments = _jsonl(output / 'experiments.jsonl')
    assert [trial['status'] for trial in experiments] == ['success', 'failed', 'success']
    assert experiments[0]['candidate']['algorithm'] == 'logistic_regression'
    assert experiments[0]['score'] < experiments[2]['score']
    assert experiments[1]['error_type'] == 'NameError'
    assert 'MissingInteractionFeatures' in experiments[1]['error_feedback']
    assert experiments[2]['validation_labels_passed_to_generated_code'] is False
    for trial, source in [(2, BROKEN_SOURCE), (3, REPAIRED_SOURCE)]:
        assert (output / 'experiments' / f'trial_{trial:03d}' / 'candidate.py').read_text(encoding='utf-8') == source

    assert contexts[0]['latest_generated_candidate'] is None
    assert len(contexts[0]['validation_history']) == 1
    failed = contexts[1]['latest_generated_candidate']
    assert failed['source'] == BROKEN_SOURCE and failed['status'] == 'failed'
    assert 'MissingInteractionFeatures' in failed['error_feedback']
    assert failed['generated_line'] == experiments[1]['generated_line'] == 16
    assert contexts[1]['validation_history'][-1]['error_type'] == 'NameError'
    assert contexts[1]['validation_history'][-1]['generated_line'] == 16
    assert contexts[2]['latest_generated_candidate']['source'] == REPAIRED_SOURCE
    assert contexts[2]['latest_generated_candidate']['status'] == 'success'
    assert [column['name'] for column in contexts[0]['training_profile']['feature_schema']] == ['column_0', 'column_1']
    prompt_text = json.dumps(contexts)
    assert 'private_feature_a' not in prompt_text and 'private_synthetic_task_name' not in prompt_text
    assert str(tmp_path) not in prompt_text

    fit_payload = joblib.load(output / 'experiments/trial_003/worker/input.joblib')
    assert set(fit_payload) == {'mode', 'source', 'seed', 'X_train', 'y_train', 'prediction_features'}
    assert set(fit_payload['prediction_features']) == {'train', 'validation'}
    assert len(fit_payload['y_train']) == 96
    assert list(fit_payload['X_train']) == ['column_0', 'column_1']
    refit_payload = joblib.load(output / 'refit/worker/input.joblib')
    # A separately provided validation file stays held out even during refit.
    assert len(refit_payload['y_train']) == 96 and refit_payload['prediction_features'] == {}
    pd.testing.assert_series_equal(refit_payload['y_train'], fit_payload['y_train'])

    bundle = joblib.load(output / 'model.joblib')
    assert isinstance(bundle['pipeline'], GeneratedModel)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    from superagent import __version__
    from superagent.utils import file_hash
    assert bundle['project_version'] == manifest['project_version'] == summary['project_version'] == __version__
    assert bundle['run_id'] == manifest['run_id'] == summary['run_id']
    assert summary['llm_config_sha256'] == manifest['llm_config_sha256'] == file_hash(output / 'llm_config.snapshot.json')
    assert summary['model_sha256'] == file_hash(output / 'model.joblib')
    assert all(row['project_version'] == __version__ for row in _jsonl(output / 'llm_calls.jsonl'))
    reloaded = export_predictions(bundle, tmp_path / 'prediction.csv', tmp_path / 'reloaded.csv')
    original = pd.read_csv(output / 'predictions.csv')
    pd.testing.assert_frame_equal(original, pd.read_csv(reloaded['predictions']))
    np.testing.assert_array_equal(original['label'], prediction['label'])
    probabilities = pd.read_csv(reloaded['probabilities'])
    assert probabilities.shape == (12, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert len([row for row in _jsonl(output / 'llm_calls.jsonl') if row['event'] == 'request_completed']) == 3
    events = [row['event'] for row in _jsonl(output / 'events.jsonl')]
    assert events.index('model_selected') < events.index('prediction_started')
    assert (output / 'source_snapshot/superagent/coding_planner.py').is_file()
    assert (output / 'source_snapshot/superagent/generated_code.py').is_file()


@pytest.mark.parametrize('command', ['run', 'campaign'])
def test_cli_forwards_codegen_mode(command, monkeypatch, capsys):
    received = []

    def fake_entry(*args, **kwargs):
        received.append((args, kwargs))
        return {'status': 'completed'}

    module = importlib.import_module('superagent.runner' if command == 'run' else 'superagent.campaign')
    monkeypatch.setattr(module, command, fake_entry)
    config_flag = '--llm-config' if command == 'run' else '--llm-configs'
    monkeypatch.setattr(sys, 'argv', ['superagent', command, '--task', 'synthetic-task.json',
                                    '--output', 'synthetic-output', '--planner', 'llm-code',
                                    config_flag, 'fake-config.json'])
    assert importlib.import_module('superagent.__main__').main() == 0
    args, _ = received[0]
    assert args[2 if command == 'run' else -1] == 'llm-code'
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


def test_campaign_forwards_codegen_mode_and_shared_budget(synthetic_task, fake_config, tmp_path, monkeypatch):
    task_path, _ = synthetic_task
    task = json.loads(task_path.read_text(encoding='utf-8'))
    task.pop('predict_path')
    task_path.write_text(json.dumps(task), encoding='utf-8')
    calls = []

    def fake_run(task, output, planner_mode, config, experiment_timeout, refit_timeout, budget_limit):
        calls.append((planner_mode, budget_limit))
        return {
            'status': 'completed', 'run_mode': planner_mode, 'llm_requests': 2, 'valid_llm_decisions': 2,
            'generated_code_experiments': 1, 'successful_generated_code_experiments': 1,
            'selected_generated_code': True, 'best_validation_score': .9,
            'selected_candidate': _plan(REPAIRED_SOURCE, 'interaction_svc')['candidate'],
            'model': str(output / 'model.joblib'), 'source_sha256': 'same-source',
            'usage': {'accounted_cost_rmb': .1, 'accounting_complete': True, 'blocked': False},
        }

    module = importlib.import_module('superagent.campaign')
    monkeypatch.setattr(module, 'run', fake_run)
    result = module.campaign(task_path, [fake_config, fake_config], tmp_path / 'campaign',
                             budget_rmb=.5, planner_mode='llm-code')
    assert [mode for mode, _ in calls] == ['llm-code', 'llm-code']
    assert [budget for _, budget in calls] == pytest.approx([.5, .4])
    assert result['status'] == 'completed' and result['planner_mode'] == 'llm-code'
    assert result['selection']['candidate']['algorithm'] == 'generated_python'
    assert result['cumulative_cost_rmb'] == pytest.approx(.2)
    assert result['test_evaluated'] is False
