from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing_extensions import Self

TreeModel = TypeVar("TreeModel", lgb.Booster, xgb.Booster, cb.CatBoost)


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, TreeModel]


class BaseModel(ABC):
    def __init__(self: Self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def save_model(self, save_dir: Path) -> None:
        joblib.dump(self.result.models, save_dir / "models.pkl")

    def fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> TreeModel:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def run_cv_training(self: Self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Run cross-validation training and store out-of-fold predictions and models."""

        def predict_with_model(model, X_valid):
            """Helper function to handle different model types during prediction."""
            if isinstance(model, lgb.Booster):
                return model.predict(X_valid)

            elif isinstance(model, xgb.Booster):
                return model.predict(xgb.DMatrix(X_valid))

            return model.predict(X_valid)

        # Initialize variables
        oof_preds = np.zeros(len(X))
        models = {}
        kfold = KFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        # Cross-validation loop
        for fold, (train_idx, valid_idx) in enumerate(
            tqdm(kfold.split(X), total=self.cfg.data.n_splits, desc="cv", leave=False), 1
        ):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Train model and store predictions
            model = self.fit(X_train, y_train, X_valid, y_valid)
            oof_preds[valid_idx] = predict_with_model(model, X_valid)
            models[f"fold_{fold}"] = model

        # Clean up and save results
        del X_train, X_valid, y_train, y_valid
        gc.collect()

        self.result = ModelResult(oof_preds=oof_preds, models=models)
        return self
