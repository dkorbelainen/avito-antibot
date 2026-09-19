"""Validation harness. All experiment decisions are made through `run_experiment`."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .model import ModelSpec, fit_predict


def _load_official_metric():
    """Score with the organisers' own file so local numbers match the platform."""
    spec = importlib.util.spec_from_file_location("official_metric", config.ROOT / "metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRIC = _load_official_metric()
precision_at_recall = METRIC.precision_at_recall
recall_at_fpr = METRIC.recall_at_fpr


def score(y_true: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "p_at_r70": float(precision_at_recall(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, pred)),
        "recall_at_fpr1": float(recall_at_fpr(y_true, pred, 0.01)),
    }


def oof_predictions(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_folds: int = config.N_FOLDS,
) -> np.ndarray:
    oof = np.zeros(len(x), dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x, y):
        oof[valid_idx] = fit_predict(
            spec, x.iloc[train_idx], y.iloc[train_idx], x.iloc[valid_idx], seed=seed
        )
    return oof


def repeated_cv(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
    n_seeds: int = config.N_SEEDS,
    n_folds: int = config.N_FOLDS,
) -> dict[str, object]:
    """Primary scheme: repeated stratified CV, reported as mean +/- std over seeds."""
    per_seed: list[dict[str, float]] = []
    oof_sum = np.zeros(len(x), dtype=float)
    for offset in range(n_seeds):
        oof = oof_predictions(spec, x, y, seed=config.SEED + offset, n_folds=n_folds)
        per_seed.append(score(y, oof))
        oof_sum += pd.Series(oof).rank(pct=True).to_numpy()
    frame = pd.DataFrame(per_seed)
    report: dict[str, object] = {f"{k}_mean": float(frame[k].mean()) for k in frame}
    report |= {f"{k}_std": float(frame[k].std()) for k in frame}
    report |= {f"blend_{k}": v for k, v in score(y, oof_sum / n_seeds).items()}
    return report | {"oof": oof_sum / n_seeds}


def forward_chain(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
    day_index: pd.Series,
    first_valid_day: int = 7,
) -> dict[str, float]:
    """Secondary scheme: train on every earlier day, validate on day k."""
    days = sorted(day_index.unique())
    rows: list[dict[str, float]] = []
    for day in days[first_valid_day:]:
        train_mask = day_index < day
        valid_mask = day_index == day
        pred = fit_predict(spec, x[train_mask], y[train_mask], x[valid_mask])
        rows.append(score(y[valid_mask], pred))
    frame = pd.DataFrame(rows)
    return {f"fc_{k}_mean": float(frame[k].mean()) for k in frame} | {
        f"fc_{k}_std": float(frame[k].std()) for k in frame
    }


def log_result(name: str, report: dict[str, object], extra: dict[str, object] | None = None) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        **{k: v for k, v in report.items() if k != "oof"},
        **(extra or {}),
    }
    with config.RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_report(name: str, report: dict[str, object], chain: dict[str, float] | None = None) -> str:
    head = (
        f"{name:<28} "
        f"P@R70 {report['p_at_r70_mean']:.4f}±{report['p_at_r70_std']:.4f}  "
        f"ROC {report['roc_auc_mean']:.4f}  "
        f"PR {report['pr_auc_mean']:.4f}  "
        f"R@FPR1 {report['recall_at_fpr1_mean']:.4f}"
    )
    if chain is None:
        return head
    return head + f"  | fc P@R70 {chain['fc_p_at_r70_mean']:.4f}  fc ROC {chain['fc_roc_auc_mean']:.4f}"
