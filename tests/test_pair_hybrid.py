import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from superagent.pair_hybrid import PairedHybridTransformer


def training_pairs():
    return pd.DataFrame({
        "a": ["Bright warm DAY!", "cold bleak weather", "Happy news 123!", "sad report", "warm cheerful news", None],
        "b": ["cold bleak weather", "Bright warm DAY!", "sad report", "Happy news 123!", None, "warm cheerful news"],
    })


@pytest.mark.parametrize("style", [True, False])
def test_exact_pair_swap_normalized_blocks_and_identical_pair_zero(style):
    features = PairedHybridTransformer(["a", "b"], min_df=1, style_features=style).fit(training_pairs())
    unseen = pd.DataFrame({"a": ["NEW news 999?", None, "same exact text"],
                           "b": ["cold report!", "warm weather", "same exact text"]})
    matrix = features.transform(unseen)
    swapped = features.transform(unseen.rename(columns={"a": "b", "b": "a"}))
    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_array_equal(matrix.toarray(), -swapped.toarray())
    np.testing.assert_array_equal(matrix[-1].toarray(), np.zeros((1, matrix.shape[1])))
    word_width = len(features.word_.get_feature_names_out())
    char_width = len(features.char_.get_feature_names_out())
    for block in (matrix[:2, :word_width], matrix[:2, word_width:word_width + char_width]):
        np.testing.assert_allclose(np.asarray(block.multiply(block).sum(axis=1)).ravel(), [1., 1.])
    assert len(features.get_feature_names_out()) == matrix.shape[1]
    assert matrix.shape[1] == word_width + char_width + (16 if style else 0)
    np.testing.assert_array_equal(matrix.toarray(), features.transform(unseen[["b", "a"]]).toarray())


def test_vocabulary_idf_and_style_scale_remain_training_only():
    features = PairedHybridTransformer(["a", "b"], min_df=1).fit(training_pairs())
    vocabularies = [transformer.vectorizer_.vocabulary_.copy() for transformer in (features.word_, features.char_)]
    idfs = [transformer.vectorizer_.idf_.copy() for transformer in (features.word_, features.char_)]
    scales = features.style_scaler_.scale_.copy()
    means = features.style_scaler_.mean_.copy()
    samples_seen = features.style_scaler_.n_samples_seen_
    assert "bright" in vocabularies[0]
    assert "bleak" in vocabularies[0]
    features.transform(pd.DataFrame({"a": ["VALIDATIONONLY☃" * 10000], "b": [None]}))
    assert "validationonly" not in features.word_.vectorizer_.vocabulary_
    for index, transformer in enumerate((features.word_, features.char_)):
        assert vocabularies[index] == transformer.vectorizer_.vocabulary_
        np.testing.assert_array_equal(idfs[index], transformer.vectorizer_.idf_)
    np.testing.assert_array_equal(scales, features.style_scaler_.scale_)
    np.testing.assert_array_equal(means, features.style_scaler_.mean_)
    assert features.style_scaler_.n_samples_seen_ == samples_seen
    assert features.style_scaler_.with_mean is False


def test_serialization_clone_and_probability_complement(tmp_path):
    X = training_pairs()
    transformer = PairedHybridTransformer(["a", "b"], min_df=1)
    assert clone(transformer).get_params() == transformer.get_params()
    pipeline = Pipeline([("hybrid", transformer), ("classifier", LogisticRegression(C=10, fit_intercept=False))])
    pipeline.fit(X, [1, 0, 1, 0, 1, 0])
    predicted = pipeline.predict_proba(X)
    np.testing.assert_array_equal(pipeline.predict(X), [1, 0, 1, 0, 1, 0])
    swapped = X.rename(columns={"a": "b", "b": "a"})
    np.testing.assert_allclose(predicted[:, 1], 1 - pipeline.predict_proba(swapped)[:, 1], atol=1e-12)
    file = tmp_path / "hybrid.joblib"
    joblib.dump(pipeline, file)
    np.testing.assert_array_equal(predicted, joblib.load(file).predict_proba(X))


def test_missing_fields_and_unfitted_fail_clearly():
    transformer = PairedHybridTransformer(["a", "b"], min_df=1)
    with pytest.raises(NotFittedError):
        transformer.transform(training_pairs())
    with pytest.raises(ValueError, match="missing"):
        transformer.fit(pd.DataFrame({"a": ["words here"]}))


@pytest.mark.parametrize("params", [
    {"style_features": "yes"}, {"word_max_features": True}, {"char_max_features": 0},
    {"min_df": 1.5}, {"char_ngram_range": (0, 3)}, {"char_ngram_range": (3, 2)},
])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        PairedHybridTransformer(["a", "b"], **params).fit(training_pairs())
