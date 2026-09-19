"""Gradient-boosting wrappers behind one interface, plus rank blending."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import config

Kind = Literal["lgb", "cat", "xgb"]

LGB_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
}

CAT_PARAMS: dict[str, object] = {
    "loss_function": "Logloss",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "verbose": 0,
    "allow_writing_files": False,
    "thread_count": 8,
}

XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "nthread": 8,
    "verbosity": 0,
}


@dataclass(frozen=True)
class ModelSpec:
    kind: Kind
    n_rounds: int = 800
    params: dict[str, object] = field(default_factory=dict)

    def resolved(self, seed: int) -> dict[str, object]:
        base = {"lgb": LGB_PARAMS, "cat": CAT_PARAMS, "xgb": XGB_PARAMS}[self.kind]
        merged = {**base, **self.params}
        seed_key = {"lgb": "seed", "cat": "random_seed", "xgb": "seed"}[self.kind]
        merged[seed_key] = seed
        return merged


def fit_predict(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int = config.SEED,
) -> np.ndarray:
    params = spec.resolved(seed)
    if spec.kind == "lgb":
        import lightgbm as lgb

        booster = lgb.train(
            params, lgb.Dataset(x_train, y_train), num_boost_round=spec.n_rounds
        )
        return booster.predict(x_valid)
    if spec.kind == "cat":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(iterations=spec.n_rounds, **params)
        model.fit(x_train, y_train)
        return model.predict_proba(x_valid)[:, 1]
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    booster = xgb.train(params, dtrain, num_boost_round=spec.n_rounds)
    return booster.predict(xgb.DMatrix(x_valid))


def feature_gain(
    spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, seed: int = config.SEED
) -> pd.Series:
    import lightgbm as lgb

    if spec.kind != "lgb":
        raise ValueError("gain importance is only read from the LightGBM model")
    booster = lgb.train(
        spec.resolved(seed), lgb.Dataset(x_train, y_train), num_boost_round=spec.n_rounds
    )
    return pd.Series(
        booster.feature_importance("gain"), index=x_train.columns
    ).sort_values(ascending=False)


def rank_average(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Blend in rank space: the metric is ordinal, so ranks compose safely."""
    weights = weights or [1.0] * len(predictions)
    total = float(sum(weights))
    stacked = np.zeros(len(predictions[0]), dtype=float)
    for pred, weight in zip(predictions, weights, strict=True):
        ranks = pd.Series(pred).rank(pct=True).to_numpy()
        stacked += weight * ranks
    return stacked / total
