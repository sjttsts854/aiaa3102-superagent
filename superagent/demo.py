"""Small synthetic development fixtures; never course evaluation results."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .utils import write_json


def make_demo(directory: Path) -> list[Path]:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    expected = [directory / f"{kind}_{suffix}" for kind in ("tabular", "text") for suffix in ("train.csv", "predict.csv", "task.json")]
    if any(path.exists() for path in expected):
        raise ValueError("Demo files already exist; choose a new directory")
    rng = np.random.default_rng(42)
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, n_redundant=1, class_sep=1.2, random_state=42)
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    frame.insert(0, "sample_id", [f"sample_{i:04d}" for i in range(len(frame))])
    frame["category"] = rng.choice(["north", "south", "east"], len(frame))
    frame.loc[rng.choice(320, 20, replace=False), "feature_0"] = np.nan
    frame["label"] = y
    frame.iloc[:320].to_csv(directory / "tabular_train.csv", index=False)
    frame.iloc[320:].drop(columns="label").to_csv(directory / "tabular_predict.csv", index=False)
    topics = {"sport": ["team", "match", "goal", "stadium", "coach"], "science": ["research", "laboratory", "atom", "experiment", "theory"], "food": ["recipe", "kitchen", "flavor", "meal", "chef"]}
    rows = []
    for i in range(240):
        label = list(topics)[i % 3]
        words = rng.choice(topics[label], 4, replace=True).tolist() + rng.choice(["today", "new", "report", "interesting"], 3).tolist()
        rng.shuffle(words)
        rows.append({"sample_id": f"text_{i:04d}", "text": " ".join(words), "label": label})
    text = pd.DataFrame(rows)
    text.iloc[:180].to_csv(directory / "text_train.csv", index=False)
    text.iloc[180:].drop(columns="label").to_csv(directory / "text_predict.csv", index=False)
    tasks = []
    for kind in ("tabular", "text"):
        task = {"name": f"synthetic_{kind}_development_demo", "data_type": kind, "train_path": f"{kind}_train.csv", "predict_path": f"{kind}_predict.csv", "target_column": "label", "id_column": "sample_id", "metric": "f1_macro", "seed": 42, "validation_fraction": 0.2, "max_experiments": 4}
        if kind == "text":
            task["text_column"] = "text"
        path = directory / f"{kind}_task.json"
        write_json(path, task)
        tasks.append(path)
    return tasks
