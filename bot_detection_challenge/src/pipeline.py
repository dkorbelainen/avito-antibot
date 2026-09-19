"""Dataset assembly: raw files in, model-ready matrices out."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, data, features


@dataclass(frozen=True)
class Dataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    day_index: pd.Series
    train_ids: pd.Index
    test_ids: pd.Index


def _feature_code_hash() -> str:
    """Cache key tied to the feature code, so edits can never serve a stale matrix."""
    source = (config.ROOT / "src" / "features.py").read_bytes()
    return hashlib.sha256(source).hexdigest()[:12]


def build_dataset(cache: bool = True) -> Dataset:
    cache_path = config.CACHE_DIR / f"features_{_feature_code_hash()}.parquet"
    train, test = data.load_splits()
    meta = pd.concat([train.drop(columns=["target"]), test], ignore_index=True)

    if cache and cache_path.exists():
        matrix = pd.read_parquet(cache_path)
    else:
        events = data.clip_to_window(data.load_events(), meta)
        data.assert_no_future_leak(events, meta)
        matrix = features.build_features(
            events, meta, fit_ids=pd.Index(train["cookie_id"])
        )
        if cache:
            config.CACHE_DIR.mkdir(exist_ok=True)
            matrix.to_parquet(cache_path)

    matrix = matrix.astype("float64").replace([np.inf, -np.inf], np.nan)
    # Some scoped blocks reproduce a global one exactly (the cursor exists only on
    # web, so webp_* equals ptr_*). Duplicates only dilute column sampling.
    matrix = matrix.loc[:, ~matrix.T.duplicated().to_numpy()]
    train_ids = pd.Index(train["cookie_id"])
    test_ids = pd.Index(test["cookie_id"])
    day_index = pd.Series(
        (train["window_start_ts"] - train["window_start_ts"].min()).dt.days.to_numpy(),
        index=range(len(train)),
    )
    return Dataset(
        x_train=matrix.loc[train_ids].reset_index(drop=True),
        y_train=train["target"].reset_index(drop=True),
        x_test=matrix.loc[test_ids].reset_index(drop=True),
        day_index=day_index,
        train_ids=train_ids,
        test_ids=test_ids,
    )


def write_submission(test_ids: pd.Index, scores: np.ndarray, path=None) -> pd.DataFrame:
    """Scores are min-max rescaled to [0, 1]; the metric only reads their order."""
    path = path or config.SUBMISSION_PATH
    lo, hi = float(np.min(scores)), float(np.max(scores))
    normalised = (scores - lo) / (hi - lo) if hi > lo else np.full(len(scores), 0.5)
    submission = pd.DataFrame({"cookie_id": test_ids, "score": normalised})
    if submission["cookie_id"].duplicated().any():
        raise AssertionError("duplicate cookie_id in submission")
    submission.to_csv(path, index=False)
    return submission
