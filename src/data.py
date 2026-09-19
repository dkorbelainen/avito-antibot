"""Loading and cleaning. The window filter lives here and nowhere else."""

from __future__ import annotations

import pandas as pd

from . import config

_META_DATES = ["cookie_created_at", "window_start_ts", "window_end_ts"]

# The raw feed spells the same platform in several cases, and "iphone" is "ios".
_PLATFORM_ALIASES = {"iphone": "ios"}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config.TRAIN_PATH, parse_dates=_META_DATES)
    test = pd.read_csv(config.TEST_PATH, parse_dates=_META_DATES)
    return train, test


def load_events() -> pd.DataFrame:
    events = pd.read_csv(config.EVENTS_PATH, parse_dates=["event_ts"])
    events["platform"] = (
        events["platform"].str.lower().replace(_PLATFORM_ALIASES).astype("category")
    )
    events["event_name"] = events["event_name"].astype("category")
    return events


def clip_to_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only events observable at the end of each cookie's window.

    The single choke point guarding against future leakage: 12.4% of the raw rows
    are timestamped after `window_end_ts` and must never reach a feature.
    """
    bounds = meta[["cookie_id", "window_start_ts", "window_end_ts"]]
    merged = events.merge(bounds, on="cookie_id", how="inner")
    in_window = merged["event_ts"].between(
        merged["window_start_ts"], merged["window_end_ts"]
    )
    kept = merged.loc[in_window].drop(columns=["window_start_ts", "window_end_ts"])
    return kept.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(
        drop=True
    )


def assert_no_future_leak(events: pd.DataFrame, meta: pd.DataFrame) -> None:
    bounds = meta.set_index("cookie_id")
    ends = events["cookie_id"].map(bounds["window_end_ts"])
    starts = events["cookie_id"].map(bounds["window_start_ts"])
    if (events["event_ts"] > ends).any() or (events["event_ts"] < starts).any():
        raise AssertionError("events outside the observation window reached features")
