"""Small deterministic image baseline; no pretrained weights or downloads."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ImageGridFeatures(TransformerMixin, BaseEstimator):
    """Average RGB grid cells plus channel moments from in-memory image arrays.

    This deliberately modest baseline gives image tasks a runnable first result.
    The coding planner can implement other features and estimators independently.
    """

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    @staticmethod
    def _images(X) -> list[np.ndarray]:
        frame = pd.DataFrame(X)
        if frame.shape[1] != 1 or len(frame) == 0:
            raise ValueError("Image features need one nonempty column of RGB arrays")
        images = frame.iloc[:, 0].tolist()
        if any(not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
               or image.dtype != np.uint8 for image in images):
            raise ValueError("Every image must be a uint8 RGB array")
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("Image arrays must have the same shape")
        return images

    def fit(self, X, y=None):
        images = self._images(X)
        if type(self.grid_size) is not int or not 2 <= self.grid_size <= 16:
            raise ValueError("grid_size must be an integer between 2 and 16")
        if self.grid_size > min(images[0].shape[:2]):
            raise ValueError("grid_size cannot exceed the input image dimensions")
        self.image_shape_ = images[0].shape
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self, "image_shape_")
        images = self._images(X)
        if images[0].shape != self.image_shape_:
            raise ValueError("Image dimensions differ from the fitted model")
        features = []
        for raw in images:
            image = raw.astype(np.float64) / 255.0
            cells = [cell.mean(axis=(0, 1)) for row in np.array_split(image, self.grid_size, axis=0)
                     for cell in np.array_split(row, self.grid_size, axis=1)]
            features.append(np.concatenate([np.concatenate(cells), image.mean(axis=(0, 1)), image.std(axis=(0, 1))]))
        return np.asarray(features)
