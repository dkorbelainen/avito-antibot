"""Time-series view of a cookie's event stream.

Everything here treats the events as an irregular point process rather than as a bag
of actions. Three families, each answering a different question about the stream:

* **binned activity** — the window is cut into fixed bins and the resulting count
  series is described (peak load, dispersion, occupancy, run structure). A collector
  that fires a batch of requests leaves a tall narrow bin; a person leaves a flat one.
* **compression** — how well the symbol stream gzips, a cheap proxy for algorithmic
  repetitiveness that needs no vocabulary and no fitting.
* **irregular-sampling statistics** — circular concentration of the time of day and
  the shape of the gap distribution on a log scale.

These matter most where the cursor is absent: on mobile the binned peaks are the
single strongest signal available (`b5m_max` alone separates at ROC-AUC 0.72 inside
the mobile subgroup, against 0.36 P@R70 for the full model there).
"""

from __future__ import annotations

import gzip
import math

import numpy as np
import pandas as pd

BIN_WIDTHS_S = {60: "b1m", 300: "b5m", 1800: "b30m"}

# Permutation entropy order: 3 needs only 4 events, which the median mobile cookie has.
_PERM_ORDER = 3


def _compression_ratio(values: pd.Series) -> float:
    """Compressed size over raw size of the symbol stream, 1.0 meaning incompressible."""
    raw = "".join(values).encode()
    if not raw:
        return np.nan
    return len(gzip.compress(raw, 6)) / len(raw)


def _symbol_stream(events: pd.DataFrame, column: str, width: int) -> pd.Series:
    """One short token per event, concatenated per cookie."""
    tokens = events[column].astype(str).str.slice(0, width).fillna("?")
    return tokens.groupby(events["cookie_id"], observed=True).apply("".join)


def _run_lengths(flags: np.ndarray) -> tuple[int, int]:
    """Longest run of True and of False in a boolean occupancy vector."""
    if flags.size == 0:
        return 0, 0
    edges = np.flatnonzero(np.diff(flags.astype(np.int8)))
    bounds = np.concatenate(([-1], edges, [flags.size - 1]))
    lengths = np.diff(bounds)
    values = flags[bounds[1:]]
    on = lengths[values].max(initial=0)
    off = lengths[~values].max(initial=0)
    return int(on), int(off)


def _binned_block(events: pd.DataFrame) -> pd.DataFrame:
    """Count series over fixed bins, described at three time scales."""
    seconds = events["event_ts"].astype("int64") // 10**9
    offset = seconds - seconds.groupby(events["cookie_id"], observed=True).transform("min")
    index = pd.Index(events["cookie_id"].drop_duplicates().sort_values(), name="cookie_id")
    out = pd.DataFrame(index=index)

    for width, tag in BIN_WIDTHS_S.items():
        bins = (offset // width).astype("int64")
        counts = (
            pd.DataFrame({"cookie_id": events["cookie_id"].to_numpy(), "bin": bins.to_numpy()})
            .groupby(["cookie_id", "bin"], observed=True)
            .size()
        )
        per_cookie = counts.groupby(level=0)
        span_bins = (offset.groupby(events["cookie_id"], observed=True).max() // width) + 1

        out[f"{tag}_max"] = per_cookie.max()
        out[f"{tag}_mean"] = per_cookie.mean()
        out[f"{tag}_fano"] = per_cookie.var() / per_cookie.mean()
        out[f"{tag}_occupancy"] = per_cookie.size() / span_bins
        shares = counts / counts.groupby(level=0).transform("sum")
        out[f"{tag}_entropy"] = -(shares * np.log(shares)).groupby(level=0).sum()
        # Peak share says whether one bin carries the whole window.
        out[f"{tag}_peak_share"] = per_cookie.max() / per_cookie.sum()

        runs = counts.groupby(level=0).apply(
            lambda s: _run_lengths(
                np.isin(np.arange(s.index.get_level_values(1).max() + 1), s.index.get_level_values(1))
            )
        )
        out[f"{tag}_run_on"] = runs.map(lambda pair: pair[0])
        out[f"{tag}_run_off"] = runs.map(lambda pair: pair[1])
    return out


def _compression_block(events: pd.DataFrame) -> pd.DataFrame:
    """How repetitive the action / catalogue / pacing streams are."""
    index = pd.Index(events["cookie_id"].drop_duplicates().sort_values(), name="cookie_id")
    out = pd.DataFrame(index=index)

    streams = {
        "cmp_event": _symbol_stream(events, "event_name", 3),
        "cmp_category": _symbol_stream(events, "item_category", 3),
        "cmp_location": _symbol_stream(events, "item_location", 3),
    }
    for name, stream in streams.items():
        out[name] = stream.map(_compression_ratio).reindex(index)

    # Action plus its gap bucket: catches "same action, same pacing" loops that neither
    # stream shows on its own.
    seconds = events["event_ts"].astype("int64") // 10**9
    gaps = seconds.groupby(events["cookie_id"], observed=True).diff()
    bucket = pd.cut(gaps, [-1, 1, 5, 30, 300, np.inf], labels=list("abcde")).astype(str)
    paced = events["event_name"].astype(str).str.slice(0, 2) + bucket
    out["cmp_paced"] = paced.groupby(events["cookie_id"], observed=True).apply("".join).map(
        _compression_ratio
    )
    return out


def _permutation_entropy(values: np.ndarray, order: int = _PERM_ORDER) -> float:
    """Bandt-Pompe entropy of the ordinal patterns in a series (normalised to [0, 1])."""
    if values.size < order + 1:
        return np.nan
    windows = np.lib.stride_tricks.sliding_window_view(values, order)
    patterns = np.argsort(windows, axis=1, kind="stable")
    codes = (patterns * (order ** np.arange(order))).sum(axis=1)
    _, counts = np.unique(codes, return_counts=True)
    shares = counts / counts.sum()
    return float(-(shares * np.log(shares)).sum() / np.log(math.factorial(order)))


def _irregular_block(events: pd.DataFrame) -> pd.DataFrame:
    """Circular time-of-day concentration and log-scale gap shape."""
    index = pd.Index(events["cookie_id"].drop_duplicates().sort_values(), name="cookie_id")
    out = pd.DataFrame(index=index)
    seconds = events["event_ts"].astype("int64") // 10**9

    angle = (seconds % 86400) / 86400 * 2 * np.pi
    circular = pd.DataFrame(
        {"cookie_id": events["cookie_id"].to_numpy(), "cos": np.cos(angle), "sin": np.sin(angle)}
    ).groupby("cookie_id", observed=True).mean()
    resultant = np.hypot(circular["cos"], circular["sin"])
    size = events.groupby("cookie_id", observed=True).size()
    out["tod_resultant"] = resultant
    # Rayleigh statistic: concentration weighted by how much evidence there is for it.
    out["tod_rayleigh"] = size * resultant**2

    gaps = pd.DataFrame(
        {"cookie_id": events["cookie_id"].to_numpy(), "dt": seconds.groupby(events["cookie_id"], observed=True).diff().to_numpy()}
    ).dropna()
    grouped = gaps.groupby("cookie_id")["dt"]
    out["gap_log_std"] = grouped.apply(lambda s: float(np.log1p(s).std()))
    out["gap_log_iqr"] = grouped.apply(
        lambda s: float(np.log1p(s).quantile(0.75) - np.log1p(s).quantile(0.25))
    )
    out["gap_near_mode"] = grouped.apply(
        lambda s: float(((s - s.mode().iloc[0]).abs() <= 1).mean())
    )
    out["gap_perm_entropy"] = grouped.apply(lambda s: _permutation_entropy(s.to_numpy()))
    return out


def build_block(events: pd.DataFrame) -> pd.DataFrame:
    """All time-series features, one row per cookie present in `events`."""
    return pd.concat(
        [_binned_block(events), _compression_block(events), _irregular_block(events)],
        axis=1,
    )
