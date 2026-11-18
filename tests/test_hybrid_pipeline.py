import joblib
import numpy as np
import pandas as pd
import pytest

from superagent.data import TaskSpec
from superagent.models import build_pipeline


@pytest.mark.parametrize("style", [True, False])
def test_catalog_hybrid_roundtrip_and_order(tmp_path, style):
    task = TaskSpec("paired_text", tmp_path / "train.csv", "label", text_columns=["a", "b"])
    X = pd.DataFrame({"a": ["amazing happy world!", "boring ordinary event", "happy good event", "bad sad world"],
                      "b": ["boring ordinary event", "amazing happy world!", "bad sad world", "happy good event"]})
    y = [1, 0, 1, 0]
    candidate = {"algorithm": "paired_text_hybrid", "params": {"min_df": 1, "style_features": style}}
    pipeline = build_pipeline(task, candidate, X).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    swapped = X.rename(columns={"a": "b", "b": "a"})
    np.testing.assert_allclose(loaded.predict_proba(X)[:, 1], 1 - loaded.predict_proba(swapped)[:, 1])
    np.testing.assert_array_equal(loaded.predict(X), y)
    with pytest.raises(ValueError, match="boolean"):
        build_pipeline(task, {"algorithm": "paired_text_hybrid", "params": {"style_features": "false"}}, X)
