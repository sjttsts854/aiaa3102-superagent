import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from superagent.data import TaskSpec
from superagent.models import build_pipeline, candidate_catalog
from superagent.pair_features import PairedTfidfTransformer


def pair_data():
    return pd.DataFrame({
        "left": ["bright good", "dark bad", "nice joy", "sad pain", "warm happy", "cold awful"],
        "right": ["dark bad", "bright good", "sad pain", "nice joy", "cold awful", "warm happy"],
    })


def pair_task(tmp_path):
    return TaskSpec(data_type="paired_text", train_path=tmp_path / "train.csv",
                    target_column="label", text_columns=["left", "right"])


def test_shared_vocabulary_train_only_and_exact_swap():
    train = pd.DataFrame({"a": ["leftonly common", "common"], "b": ["rightonly", None]})
    features = PairedTfidfTransformer(columns=["a", "b"]).fit(train)
    assert set(features.vectorizer_.vocabulary_) == {"leftonly", "common", "rightonly"}
    vocabulary = features.vectorizer_.vocabulary_.copy()
    idf = features.vectorizer_.idf_.copy()
    valid = pd.DataFrame({"a": ["validationonly leftonly"], "b": ["rightonly"]})
    transformed = features.transform(valid)
    swapped = valid.rename(columns={"a": "b", "b": "a"})
    np.testing.assert_allclose(transformed.toarray(), -features.transform(swapped).toarray())
    assert features.vectorizer_.vocabulary_ == vocabulary
    np.testing.assert_array_equal(features.vectorizer_.idf_, idf)
    assert "validationonly" not in vocabulary
    assert clone(features).columns == ["a", "b"]


@pytest.mark.parametrize("representation,analyzer,ngram", [
    ("difference", "word", [1, 1]), ("difference", "char", [2, 3]),
    ("slots", "word", [1, 2]),
])
def test_pair_pipeline_serialization_and_order(tmp_path, representation, analyzer, ngram):
    task = pair_task(tmp_path)
    X = pair_data()
    pipeline = build_pipeline(task, {"algorithm": "paired_text_logistic", "params": {
        "representation": representation, "analyzer": analyzer, "ngram_range": ngram,
        "C": 10, "max_features": 1000, "min_df": 1,
    }}, X).fit(X, [1, 0, 1, 0, 1, 0])
    expected = pipeline.predict_proba(X)
    np.testing.assert_array_equal(pipeline.predict(X), [1, 0, 1, 0, 1, 0])
    # DataFrame storage order must not silently reverse the configured pair order.
    np.testing.assert_allclose(expected, pipeline.predict_proba(X[["right", "left"]]))
    swapped = X.rename(columns={"left": "right", "right": "left"})
    if representation == "difference":
        np.testing.assert_allclose(expected[:, 1], 1 - pipeline.predict_proba(swapped)[:, 1], atol=1e-12)
        assert pipeline.named_steps["classifier"].fit_intercept is False
    model_path = tmp_path / "pair.joblib"
    joblib.dump(pipeline, model_path)
    np.testing.assert_array_equal(expected, joblib.load(model_path).predict_proba(X))


@pytest.mark.parametrize("params", [
    {"representation": "concatenated_text"}, {"analyzer": "bad"},
    {"ngram_range": [0, 1]}, {"max_features": 99}, {"min_df": 0},
    {"C": True}, {"unknown": 1},
])
def test_pair_invalid_candidate_rejected(tmp_path, params):
    with pytest.raises(ValueError):
        build_pipeline(pair_task(tmp_path), {"algorithm": "paired_text_logistic", "params": params}, pair_data())


def test_pair_missing_column_and_unsupported_algorithm(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        PairedTfidfTransformer(columns=["left", "right"]).fit(pd.DataFrame({"left": ["text"]}))
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        build_pipeline(pair_task(tmp_path), {"algorithm": "text_naive_bayes"}, pair_data())
    assert "paired_text_logistic" in candidate_catalog("paired_text")
