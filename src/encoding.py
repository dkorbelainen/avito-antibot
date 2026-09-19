"""Fold-safe target encoding of the raw categorical keys.

Three fields are high-cardinality enough that the feature blocks can only summarise
them: the exact User-Agent string, the dominant item category and the dominant item
location. The encoding is smoothed toward the global positive rate and is refitted
inside every fold, so a cookie never contributes to the statistic it is scored by.

The User-Agent is the one that earns its place: all 148 strings in test also occur in
train, and a string's positive rate correlates 0.59 between the two halves of the
training week, so the statistic transfers rather than memorises.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

KEY_COLUMNS = ("key_ua", "key_category", "key_location")

# Prior weight in the smoothing. A key needs roughly this many cookies before its own
# rate outweighs the global one.
SMOOTHING = 20.0


def build_keys(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One dominant value per cookie for each encoded field."""
    grouped = events.groupby("cookie_id", observed=True)

    def dominant(column: str) -> pd.Series:
        return grouped[column].agg(
            lambda values: values.value_counts().idxmax() if values.notna().any() else "?"
        )

    keys = pd.DataFrame(
        {
            "key_ua": dominant("user_agent"),
            "key_category": dominant("item_category"),
            "key_location": dominant("item_location"),
        }
    )
    return keys.reindex(meta["cookie_id"]).fillna("?")


def fit(keys: pd.DataFrame, y: pd.Series, smoothing: float = SMOOTHING) -> dict[str, pd.Series]:
    """Smoothed positive rate per key value, learned on the fitting rows only."""
    prior = float(y.mean())
    maps: dict[str, pd.Series] = {}
    for column in KEY_COLUMNS:
        stats = y.groupby(keys[column].to_numpy()).agg(["sum", "count"])
        maps[column] = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    maps["__prior__"] = pd.Series([prior], index=["prior"])
    return maps


def transform(keys: pd.DataFrame, maps: dict[str, pd.Series]) -> pd.DataFrame:
    """Encoded columns for any set of cookies; unseen values fall back to the prior."""
    prior = float(maps["__prior__"].iloc[0])
    return pd.DataFrame(
        {
            f"te_{column.removeprefix('key_')}": keys[column]
            .map(maps[column])
            .fillna(prior)
            .to_numpy(dtype="float64")
            for column in KEY_COLUMNS
        },
        index=keys.index,
    )


def augment(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    keys_train: pd.DataFrame,
    others: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Fit on the training rows, append the encoded columns to every frame given."""
    maps = fit(keys_train, y_train)
    fitted = pd.concat(
        [x_train.reset_index(drop=True), transform(keys_train, maps).reset_index(drop=True)],
        axis=1,
    )
    applied = [
        pd.concat(
            [frame.reset_index(drop=True), transform(frame_keys, maps).reset_index(drop=True)],
            axis=1,
        )
        for frame, frame_keys in others
    ]
    return fitted, applied


def keys_cache_path() -> Path:
    return config.CACHE_DIR / "keys.parquet"
