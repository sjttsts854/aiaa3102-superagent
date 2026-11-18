"""Independent synthetic fixtures for code-generation integration, not benchmarks."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import make_moons

from .utils import write_json


def make_code_demo(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Code demo directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2718)
    tasks = []
    for kind in ("tabular", "text", "image"):
        folder = directory / kind
        folder.mkdir()
        n_train, n_valid, n_predict = 180, 60, 15
        n = n_train + n_valid + n_predict
        if kind == "tabular":
            X, y = make_moons(n_samples=n, noise=0.17, random_state=2718)
            frame = pd.DataFrame(X, columns=["coordinate_a", "coordinate_b"])
            frame["label"] = y
        elif kind == "text":
            # Same vocabulary across classes; relative marker position matters.
            rows = []
            for i in range(n):
                label = i % 3
                words = rng.choice(["new", "daily", "small", "report", "item", "note"], 18).tolist()
                first, second = sorted(rng.choice(18, 2, replace=False).tolist())
                words[first], words[second] = (("alpha", "beta"), ("beta", "alpha"), ("alpha", "alpha"))[label]
                rows.append({"text": " ".join(words), "label": label})
            frame = pd.DataFrame(rows)
        else:
            assets = folder / "images"
            assets.mkdir()
            yy, xx = np.indices((28, 28))
            rows = []
            for i in range(n):
                label = i % 3
                axis = (xx, yy, xx + yy)[label]
                phase = rng.integers(0, 8)
                pattern = ((axis + phase) % 8 < 4).astype(float)
                pixels = np.repeat((60 + 130 * pattern)[..., None], 3, axis=2)
                pixels += rng.normal(0, 18, pixels.shape)
                name = f"images/sample_{i:04d}.png"
                Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(folder / name)
                rows.append({"image": name, "label": label})
            frame = pd.DataFrame(rows)
        frame.iloc[:n_train].to_csv(folder / "train.csv", index=False)
        frame.iloc[n_train:n_train + n_valid].to_csv(folder / "validation.csv", index=False)
        frame.iloc[n_train + n_valid:].drop(columns="label").to_csv(folder / "predict.csv", index=False)
        task = {"name": f"synthetic_codegen_{kind}", "data_type": kind,
                "train_path": "train.csv", "validation_path": "validation.csv",
                "target_column": "label",
                "metric": "accuracy", "seed": 2718, "max_experiments": 4}
        if kind == "text":
            task["text_column"] = "text"
        if kind == "image":
            task.update(image_column="image", image_size=[28, 28])
        path = folder / "task.json"
        write_json(path, task)
        tasks.append(path)
    write_json(directory / "purpose.json", {"purpose": "Independent synthetic code-generation integration checks",
               "official_course_data_used": False, "scores_are_course_results": False,
               "seed": 2718, "tasks": [str(path.relative_to(directory)) for path in tasks]})
    return tasks
