"""Small, portable regression model used by the gold-price application."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RatioLinearRegressor:
    """Least-squares regressor for predicting the next-day price ratio."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RatioLinearRegressor":
        self.feature_names_in_ = list(features.columns)
        x_values = features.to_numpy(dtype=float)
        design_matrix = np.column_stack([np.ones(len(x_values)), x_values])
        parameters, *_ = np.linalg.lstsq(design_matrix, target.to_numpy(dtype=float), rcond=None)
        self.intercept_ = float(parameters[0])
        self.coef_ = parameters[1:]
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        missing_features = set(self.feature_names_in_) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing model features: {', '.join(sorted(missing_features))}")
        x_values = features[self.feature_names_in_].to_numpy(dtype=float)
        return self.intercept_ + x_values @ self.coef_
