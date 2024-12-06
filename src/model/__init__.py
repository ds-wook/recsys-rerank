from typing import TypeVar

from omegaconf import DictConfig

from .base import *
from .boosting import *

BulidModel = TypeVar("BulidModel", bound=BaseModel)


def bulid_model(cfg: DictConfig) -> BulidModel:
    model_type = {"lightgbm": LightGBMTrainer(cfg), "xgboost": XGBoostTrainer(cfg), "catboost": CatBoostTrainer(cfg)}

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError
