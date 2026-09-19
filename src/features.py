"""Per-cookie feature construction.

Every block takes the already window-clipped event frame and returns a frame
indexed by `cookie_id`. `build_features` is the only public entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, timeseries

_EPS = 1e-9

# Single letters keep the n-gram vocabulary small and the documents short.
_EVENT_CODE = {
    "search_results_view": "S",
    "item_view": "I",
    "photo_swipe": "P",
    "seller_page_view": "L",
    "contact_phone_show": "H",
    "contact_chat_open": "C",
    "contact_message_sent": "M",
    "favorite_add": "F",
    "login": "G",
    "captcha_shown": "K",
}


def _entropy(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """Shannon entropy of `value_col` within each cookie, in nats."""
    counts = frame.groupby(["cookie_id", value_col], observed=True).size()
    shares = counts / counts.groupby(level=0).transform("sum")
    return (-shares * np.log(shares)).groupby(level=0).sum()


def _share_table(frame: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Per-cookie counts of each value of `column`, one column per value."""
    table = (
        frame.groupby(["cookie_id", column], observed=True)
        .size()
        .unstack(fill_value=0)
        .astype("float64")
    )
    table.columns = [f"{prefix}{c}" for c in table.columns]
    return table


def _grouped_corr(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Pearson correlation of two columns inside each cookie, vectorised."""
    valid = frame[["cookie_id", x, y]].dropna()
    g = valid.groupby("cookie_id")
    n = g.size()
    sx, sy = g[x].sum(), g[y].sum()
    sxx = g[x].apply(lambda s: float(np.dot(s, s)))
    syy = g[y].apply(lambda s: float(np.dot(s, s)))
    sxy = (valid[x] * valid[y]).groupby(valid["cookie_id"]).sum()
    cov = sxy - sx * sy / n
    den = np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))
    return (cov / den.replace(0, np.nan)).where(n > 2)


def _block_volume_and_mix(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby("cookie_id")
    out = pd.DataFrame(index=g.size().index)
    out["n_events"] = g.size()
    out["n_events_log"] = np.log1p(out["n_events"])

    counts = _share_table(events, "event_name", "cnt_")
    counts = counts.reindex(
        columns=[f"cnt_{n}" for n in config.EVENT_NAMES], fill_value=0.0
    )
    rates = counts.div(out["n_events"], axis=0)
    rates.columns = [c.replace("cnt_", "rate_") for c in counts.columns]

    out = out.join(counts).join(rates)
    out["rate_engagement"] = rates[
        [f"rate_{n}" for n in config.ENGAGEMENT_EVENTS]
    ].sum(axis=1)
    out["rate_contact"] = rates[[f"rate_{n}" for n in config.CONTACT_EVENTS]].sum(axis=1)
    out["ratio_search_item"] = counts["cnt_search_results_view"] / (
        counts["cnt_item_view"] + 1.0
    )
    out["ratio_photo_item"] = counts["cnt_photo_swipe"] / (counts["cnt_item_view"] + 1.0)
    out["ratio_contact_seller"] = out["rate_contact"] / (
        rates["rate_seller_page_view"] + 0.01
    )

    hours = events.assign(hour=events["event_ts"].dt.hour)
    out["n_active_hours"] = hours.groupby("cookie_id")["hour"].nunique()
    out["events_per_active_hour"] = out["n_events"] / out["n_active_hours"]
    out["hour_entropy"] = _entropy(hours, "hour")
    night = hours["hour"].between(1, 5)
    out["night_rate"] = night.groupby(hours["cookie_id"]).mean()
    return out


def _block_transitions(events: pd.DataFrame, top_bigrams: list[str]) -> pd.DataFrame:
    seq = events[["cookie_id", "event_name"]].copy()
    seq["prev"] = seq.groupby("cookie_id", observed=True)["event_name"].shift()
    pairs = seq.dropna(subset=["prev"]).copy()
    pairs["bigram"] = (
        pairs["prev"].astype(str) + ">" + pairs["event_name"].astype(str)
    )

    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    n_pairs = pairs.groupby("cookie_id").size()
    # A count is zero when the cookie has no qualifying pair, not unknown.
    out["n_bigrams_uniq"] = pairs.groupby("cookie_id")["bigram"].nunique().reindex(out.index).fillna(0)
    out["bigram_entropy"] = _entropy(pairs, "bigram")
    out["self_loop_rate"] = (
        (pairs["prev"].astype(str) == pairs["event_name"].astype(str))
        .groupby(pairs["cookie_id"])
        .mean()
    )

    counts = _share_table(pairs, "bigram", "bg_")
    counts = counts.reindex(columns=[f"bg_{b}" for b in top_bigrams], fill_value=0.0)
    out = out.join(counts.div(n_pairs, axis=0))
    return out


def _block_timing(events: pd.DataFrame) -> pd.DataFrame:
    frame = events[["cookie_id", "event_ts"]].copy()
    frame["dt"] = frame.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    gaps = frame.dropna(subset=["dt"])
    g = gaps.groupby("cookie_id")

    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    quantiles = g["dt"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    quantiles.columns = [f"dt_q{int(q * 100)}" for q in quantiles.columns]
    out = out.join(quantiles)
    out["dt_min"] = g["dt"].min()
    out["dt_max"] = g["dt"].max()
    out["dt_mean"] = g["dt"].mean()
    out["dt_std"] = g["dt"].std()
    out["dt_iqr"] = out["dt_q75"] - out["dt_q25"]
    out["dt_cv"] = out["dt_std"] / (out["dt_mean"] + 1.0)
    out["dt_mad"] = (
        (gaps["dt"] - gaps["cookie_id"].map(out["dt_q50"])).abs().groupby(gaps["cookie_id"]).median()
    )
    # Regular pacing is the clearest machine tell, so measure dispersion several ways.
    out["dt_burstiness"] = (out["dt_std"] - out["dt_mean"]) / (
        out["dt_std"] + out["dt_mean"] + 1.0
    )
    out["dt_fano"] = out["dt_std"] ** 2 / (out["dt_mean"] + 1.0)
    out["dt_rel_iqr"] = out["dt_iqr"] / (out["dt_q50"] + 1.0)

    for threshold in config.FAST_GAP_THRESHOLDS_S:
        out[f"dt_lt{threshold}"] = (gaps["dt"] < threshold).groupby(
            gaps["cookie_id"]
        ).mean()

    bucketed = gaps.assign(bucket=np.floor(np.log1p(gaps["dt"])).astype(int))
    out["dt_log_entropy"] = _entropy(bucketed, "bucket")
    out["dt_mode_share"] = (
        bucketed.groupby(["cookie_id", "bucket"]).size().groupby(level=0).max()
        / bucketed.groupby("cookie_id").size()
    )

    lagged = gaps.assign(dt_prev=gaps.groupby("cookie_id")["dt"].shift())
    out["dt_autocorr1"] = _grouped_corr(lagged, "dt", "dt_prev")

    span = events.groupby("cookie_id")["event_ts"].agg(["min", "max"])
    out["span_h"] = (span["max"] - span["min"]).dt.total_seconds() / 3600.0
    out["events_per_h"] = events.groupby("cookie_id").size() / out["span_h"].clip(lower=1 / 60)
    return out


def _block_sessions(events: pd.DataFrame) -> pd.DataFrame:
    frame = events[["cookie_id", "event_ts"]].copy()
    dt = frame.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    frame["new_session"] = dt.isna() | (dt > config.SESSION_GAP_S)
    frame["session_id"] = frame.groupby("cookie_id")["new_session"].cumsum()

    per_session = frame.groupby(["cookie_id", "session_id"])["event_ts"].agg(
        ["size", "min", "max"]
    )
    per_session["duration_s"] = (
        per_session["max"] - per_session["min"]
    ).dt.total_seconds()
    g = per_session.groupby(level=0)

    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    out["n_sessions"] = g.size()
    out["sess_events_mean"] = g["size"].mean()
    out["sess_events_max"] = g["size"].max()
    out["sess_events_std"] = g["size"].std()
    out["sess_dur_mean"] = g["duration_s"].mean()
    out["sess_dur_max"] = g["duration_s"].max()
    out["sess_rate_mean"] = (
        per_session["size"] / (per_session["duration_s"] + 1.0)
    ).groupby(level=0).mean()
    out["sess_longest_share"] = g["size"].max() / g["size"].sum()
    out["sess_gap_mean"] = (
        per_session["min"].groupby(level=0).diff().dt.total_seconds().groupby(level=0).mean()
    )
    return out


def _block_content(events: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    g = events.groupby("cookie_id")
    for column in ["item_category", "item_location", "item_id", "search_query"]:
        present = events.dropna(subset=[column])
        out[f"{column}_nuniq"] = g[column].nunique()
        out[f"{column}_entropy"] = _entropy(present, column)
        filled = g[column].count()
        out[f"{column}_uniq_ratio"] = out[f"{column}_nuniq"] / (filled + 1.0)
        out[f"{column}_per_event"] = out[f"{column}_nuniq"] / g.size()

    seller = events.dropna(subset=["seller_type"])
    out["seller_pro_rate"] = (seller["seller_type"] == "pro").groupby(
        seller["cookie_id"]
    ).mean()
    out["seller_known_rate"] = g["seller_type"].count() / g.size()

    items = events.dropna(subset=["item_id"])
    out["item_repeat_rate"] = 1.0 - out["item_id_nuniq"] / (
        items.groupby("cookie_id").size() + 1.0
    )
    # Scrapers sweep breadth; people re-open the same few listings.
    out["items_per_category"] = out["item_id_nuniq"] / (out["item_category_nuniq"] + 1.0)
    out["locations_per_category"] = out["item_location_nuniq"] / (
        out["item_category_nuniq"] + 1.0
    )
    return out


def _block_pagination(events: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    search = events.dropna(subset=["search_page"]).copy()
    if search.empty:
        return out
    g = search.groupby("cookie_id")
    out["page_mean"] = g["search_page"].mean()
    out["page_max"] = g["search_page"].max()
    out["page_std"] = g["search_page"].std()
    out["page_first_share"] = (search["search_page"] == 1).groupby(
        search["cookie_id"]
    ).mean()
    out["page_deep_share"] = (search["search_page"] > config.DEEP_PAGE).groupby(
        search["cookie_id"]
    ).mean()

    search["step"] = g["search_page"].diff()
    ascending = search["step"] == 1
    out["page_step_up_share"] = ascending.groupby(search["cookie_id"]).mean()
    out["page_step_up_count"] = ascending.groupby(search["cookie_id"]).sum().reindex(out.index).fillna(0)
    # Longest strictly consecutive page run = length of the deepest sweep.
    run_id = (~ascending).groupby(search["cookie_id"]).cumsum()
    runs = ascending.groupby([search["cookie_id"], run_id]).sum()
    out["page_run_max"] = runs.groupby(level=0).max()

    out["queries_nuniq"] = g["search_query"].nunique()
    out["pages_per_query"] = g.size() / (out["queries_nuniq"] + 1.0)
    out["query_repeat_rate"] = 1.0 - out["queries_nuniq"] / g.size()
    out["query_len_mean"] = g["search_query"].apply(lambda s: s.str.len().mean())
    return out


def _block_pointer(events: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    out["pointer_coverage"] = events.groupby("cookie_id")["pointer_x"].count() / (
        events.groupby("cookie_id").size()
    )
    pointer = events.dropna(subset=["pointer_x", "pointer_y"]).copy()
    if pointer.empty:
        return out
    g = pointer.groupby("cookie_id")
    out["ptr_x_std"] = g["pointer_x"].std()
    out["ptr_y_std"] = g["pointer_y"].std()
    out["ptr_x_mean"] = g["pointer_x"].mean()
    out["ptr_y_mean"] = g["pointer_y"].mean()
    out["ptr_bbox"] = (g["pointer_x"].max() - g["pointer_x"].min()) * (
        g["pointer_y"].max() - g["pointer_y"].min()
    )
    out["ptr_uniq_share"] = (
        pointer.groupby(["cookie_id", "pointer_x", "pointer_y"]).size().groupby(level=0).size()
        / g.size()
    )

    pointer["dx"] = g["pointer_x"].diff()
    pointer["dy"] = g["pointer_y"].diff()
    pointer["step"] = np.hypot(pointer["dx"], pointer["dy"])
    pointer["dt"] = g["event_ts"].diff().dt.total_seconds()
    pointer["speed"] = pointer["step"] / (pointer["dt"] + 1.0)
    steps = pointer.dropna(subset=["step"]).groupby("cookie_id")
    out["ptr_step_med"] = steps["step"].median()
    out["ptr_step_std"] = steps["step"].std()
    out["ptr_speed_med"] = steps["speed"].median()
    out["ptr_speed_max"] = steps["speed"].max()
    out["ptr_step_cv"] = out["ptr_step_std"] / (steps["step"].mean() + 1.0)
    return out


def _block_client(events: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    g = events.groupby("cookie_id")
    n = g.size()

    platform = _share_table(events, "platform", "plat_")
    platform = platform.reindex(
        columns=[f"plat_{p}" for p in config.PLATFORMS], fill_value=0.0
    )
    out = out.join(platform.div(n, axis=0))
    out["n_platforms"] = g["platform"].nunique()
    out["platform_entropy"] = _entropy(events, "platform")
    out["n_user_agents"] = g["user_agent"].nunique()

    ua = events["user_agent"]
    out["ua_headless_rate"] = ua.str.contains("Headless").groupby(
        events["cookie_id"]
    ).mean()
    out["ua_mobile_rate"] = ua.str.contains(
        "Android|iPhone|Mobile", case=False, regex=True
    ).groupby(events["cookie_id"]).mean()
    # The family list has to name the scripted clients explicitly. Lumping curl,
    # Scrapy and python-requests into "other" put them in the same bucket as the
    # native Avito app, which is the largest non-browser client and mostly human.
    family = ua.str.extract(
        r"^(Avito|curl|Scrapy|node-fetch|python-requests|python-urllib3|Go-http-client)/"
    )[0]
    family = family.fillna(
        ua.str.extract(r"(YaBrowser|HeadlessChrome|Firefox|Chrome|Safari)")[0]
    ).fillna("other")
    family_counts = _share_table(events.assign(ua_family=family), "ua_family", "uafam_")
    out = out.join(family_counts.div(n, axis=0))
    out["ua_script_rate"] = (
        family.isin(config.SCRIPT_CLIENTS).groupby(events["cookie_id"]).mean()
    )

    # Version and device numbers: a collector pinned to one build looks different from
    # a population that drifts across releases.
    out["ua_app_version"] = (
        ua.str.extract(r"^Avito/(\d+)")[0].astype(float).groupby(events["cookie_id"]).max()
    )
    os_version = ua.str.extract(r"Android (\d+)")[0].fillna(
        ua.str.extract(r"iOS (\d+)")[0]
    )
    out["ua_os_version"] = os_version.astype(float).groupby(events["cookie_id"]).max()
    out["ua_browser_version"] = (
        ua.str.extract(r"(?:Chrome|Firefox|Safari)/(\d+)")[0]
        .astype(float)
        .groupby(events["cookie_id"])
        .max()
    )
    device = ua.str.extract(r"(?:Android \d+; |iPhone; iOS [\d.]+; )([^)]+)")[0]
    device_freq = device.value_counts(normalize=True)
    out["ua_device_freq"] = device.map(device_freq).groupby(events["cookie_id"]).mean()

    # A rare UA string is itself suspicious, independent of what it claims to be.
    ua_freq = ua.value_counts()
    out["ua_freq_min"] = ua.map(ua_freq).groupby(events["cookie_id"]).min()
    out["ua_freq_mean"] = ua.map(ua_freq).groupby(events["cookie_id"]).mean()
    return out


def _block_cookie_meta(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=meta["cookie_id"])
    created = meta["cookie_created_at"].to_numpy()
    start = meta["window_start_ts"].to_numpy()
    age_days = (start - created) / np.timedelta64(1, "D")
    out["cookie_age_days"] = age_days
    out["cookie_age_log"] = np.log1p(np.clip(age_days, 0, None))
    out["cookie_created_hour"] = meta["cookie_created_at"].dt.hour.to_numpy()
    out["cookie_created_dow"] = meta["cookie_created_at"].dt.dayofweek.to_numpy()
    out["window_dow"] = meta["window_start_ts"].dt.dayofweek.to_numpy()
    # No absolute day index: test windows lie entirely after the training range, so a
    # tree split on it cannot transfer.
    return out


def _block_window_coverage(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    bounds = meta.set_index("cookie_id")
    span = events.groupby("cookie_id")["event_ts"].agg(["min", "max"])
    out = pd.DataFrame(index=span.index)
    out["first_event_offset_h"] = (
        span["min"] - bounds.loc[span.index, "window_start_ts"]
    ).dt.total_seconds() / 3600.0
    out["last_event_offset_h"] = (
        bounds.loc[span.index, "window_end_ts"] - span["max"]
    ).dt.total_seconds() / 3600.0
    out["window_coverage"] = (
        (span["max"] - span["min"]).dt.total_seconds() / 3600.0
    ) / 24.0
    return out


def _block_id_structure(events: pd.DataFrame) -> pd.DataFrame:
    """Numeric shape of the visited item id set: sweeps look different from browsing."""
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    items = events.dropna(subset=["item_id"]).copy()
    if items.empty:
        return out
    g = items.groupby("cookie_id")
    out["idst_std"] = g["item_id"].std()
    out["idst_range"] = g["item_id"].max() - g["item_id"].min()
    out["idst_iqr"] = g["item_id"].quantile(0.75) - g["item_id"].quantile(0.25)
    out["idst_median"] = g["item_id"].median()
    items["step"] = g["item_id"].diff()
    steps = items.dropna(subset=["step"])
    out["idst_up_share"] = (steps["step"] > 0).groupby(steps["cookie_id"]).mean()
    out["idst_step_abs_med"] = steps["step"].abs().groupby(steps["cookie_id"]).median()
    out["idst_prefix_nuniq"] = (items["item_id"] // 1000).groupby(items["cookie_id"]).nunique()
    out["idst_range_per_item"] = out["idst_range"] / (g["item_id"].nunique() + 1.0)
    return out


def _block_dt_granularity(events: pd.DataFrame) -> pd.DataFrame:
    """How many distinct inter-event gaps a cookie produces, and how round they are."""
    frame = events[["cookie_id", "event_ts"]].copy()
    frame["dt"] = frame.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    gaps = frame.dropna(subset=["dt"])
    g = gaps.groupby("cookie_id")
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    out["dtg_nuniq"] = g["dt"].nunique()
    out["dtg_uniq_frac"] = out["dtg_nuniq"] / g.size()
    for step in (5, 10, 60):
        out[f"dtg_round{step}"] = (gaps["dt"] % step == 0).groupby(gaps["cookie_id"]).mean()
    out["dtg_mode_count"] = (
        gaps.groupby(["cookie_id", "dt"]).size().groupby(level=0).max()
    )
    return out


def _block_catalog_shares(
    events: pd.DataFrame, fit_ids: pd.Index | None = None
) -> pd.DataFrame:
    """Per-cookie share of each location / category / query in a fixed vocabulary.

    The vocabulary is read off the fitting cookies only; a scored cookie contributes
    nothing to the choice of columns it is later described by.
    """
    parts: list[pd.DataFrame] = []
    index = events["cookie_id"].drop_duplicates().sort_values()
    specs = [("item_location", "shloc_", 40), ("item_category", "shcat_", 20), ("search_query", "shq_", 40)]
    for column, prefix, top_k in specs:
        present = events.dropna(subset=[column])
        fit_rows = present if fit_ids is None else present[present["cookie_id"].isin(fit_ids)]
        keep = fit_rows[column].value_counts().head(top_k).index
        subset = present[present[column].isin(keep)]
        table = _share_table(subset, column, prefix)
        totals = present.groupby("cookie_id").size()
        parts.append(table.div(totals, axis=0).reindex(index))
    return pd.concat(parts, axis=1)


def _block_popularity(
    events: pd.DataFrame, fit_ids: pd.Index | None = None
) -> pd.DataFrame:
    """How crowded are the listings this cookie opens?

    A listing's audience is counted over the fitting cookies only, and the cookie's own
    contribution is removed, so the number describes everyone else's interest in the
    same listing. Collectors concentrate on listings the crowd also opens; a person
    keeps running into listings nobody else in the window touched.
    """
    index = events["cookie_id"].drop_duplicates().sort_values()
    items = events.dropna(subset=["item_id"])
    if items.empty:
        return pd.DataFrame(index=index)
    fit_rows = items if fit_ids is None else items[items["cookie_id"].isin(fit_ids)]
    views = fit_rows.groupby("item_id").size()
    audience = fit_rows.groupby("item_id")["cookie_id"].nunique()

    pairs = items.groupby(["cookie_id", "item_id"]).size().rename("own").reset_index()
    in_fit = pairs["cookie_id"].isin(fit_rows["cookie_id"].unique()).to_numpy()
    pairs["views_other"] = pairs["item_id"].map(views) - np.where(in_fit, pairs["own"], 0)
    pairs["aud_other"] = pairs["item_id"].map(audience) - in_fit
    # An item absent from the reference is unmeasured, not unpopular.
    known = pairs.dropna(subset=["aud_other"])
    g = known.groupby("cookie_id")

    out = pd.DataFrame(index=index)
    out["pop_aud_mean"] = np.log1p(g["aud_other"].mean())
    out["pop_aud_med"] = np.log1p(g["aud_other"].median())
    out["pop_aud_max"] = np.log1p(g["aud_other"].max())
    out["pop_aud_std"] = np.log1p(g["aud_other"].std())
    out["pop_views_mean"] = np.log1p(g["views_other"].mean())
    out["pop_views_std"] = np.log1p(g["views_other"].std())
    out["pop_solo_share"] = known.assign(solo=known["aud_other"] <= 0).groupby("cookie_id")["solo"].mean()
    return out


def _block_crowd(
    events: pd.DataFrame, fit_ids: pd.Index | None = None
) -> pd.DataFrame:
    """The cookie against the crowd on the same kind of page.

    Absolute dwell and absolute page depth mix the cookie's pace with what it happened
    to look at: a listing is read for longer than a result page, and some queries are
    normally paged deeper than others. Subtracting the population median for the same
    event type, and for the same query, leaves the part that is the cookie's own.
    """
    index = events["cookie_id"].drop_duplicates().sort_values()
    out = pd.DataFrame(index=index)

    frame = events[["cookie_id", "event_name", "event_ts", "search_page", "search_query"]].copy()
    dwell = (
        frame.groupby("cookie_id")["event_ts"].shift(-1) - frame["event_ts"]
    ).dt.total_seconds()
    frame["log_dwell"] = np.log1p(dwell.clip(lower=0))
    timed = frame.dropna(subset=["log_dwell"])
    fit_rows = timed if fit_ids is None else timed[timed["cookie_id"].isin(fit_ids)]
    expected = fit_rows.groupby("event_name", observed=True)["log_dwell"].median()
    delta = timed["log_dwell"] - timed["event_name"].map(expected).astype("float64")
    grouped = delta.groupby(timed["cookie_id"])
    out["crowd_dwell_mean"] = grouped.mean()
    out["crowd_dwell_std"] = grouped.std()
    out["crowd_dwell_med"] = grouped.median()

    paged = frame.dropna(subset=["search_page", "search_query"])
    if not paged.empty:
        fit_paged = paged if fit_ids is None else paged[paged["cookie_id"].isin(fit_ids)]
        depth = fit_paged.groupby("search_query")["search_page"].median()
        gap = paged["search_page"] - paged["search_query"].map(depth).astype("float64")
        grouped = gap.groupby(paged["cookie_id"])
        out["crowd_page_mean"] = grouped.mean()
        out["crowd_page_max"] = grouped.max()
    return out


def _block_conditional_timing(events: pd.DataFrame) -> pd.DataFrame:
    """Dwell time after each event type. People linger on a listing; scrapers do not."""
    frame = events[["cookie_id", "event_name", "event_ts"]].copy()
    frame["dwell"] = (
        frame.groupby("cookie_id")["event_ts"].shift(-1) - frame["event_ts"]
    ).dt.total_seconds()
    dwell = frame.dropna(subset=["dwell"])
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    for name in ("item_view", "search_results_view", "photo_swipe", "seller_page_view"):
        subset = dwell[dwell["event_name"] == name]
        if subset.empty:
            continue
        g = subset.groupby("cookie_id")["dwell"]
        out[f"dwell_{name}_med"] = g.median()
        out[f"dwell_{name}_mean"] = g.mean()
        out[f"dwell_{name}_std"] = g.std()
    out["dwell_item_vs_search"] = out.get("dwell_item_view_med", np.nan) / (
        out.get("dwell_search_results_view_med", np.nan) + 1.0
    )
    return out


def _block_navigation(events: pd.DataFrame) -> pd.DataFrame:
    """How often the cookie switches context between consecutive events."""
    out = pd.DataFrame(index=events["cookie_id"].drop_duplicates().sort_values())
    for column, prefix in (
        ("item_category", "nav_cat"),
        ("item_location", "nav_loc"),
        ("search_query", "nav_query"),
    ):
        present = events.dropna(subset=[column])
        if present.empty:
            continue
        previous = present.groupby("cookie_id")[column].shift()
        changed = (present[column] != previous).where(previous.notna())
        valid = changed.dropna()
        ids = present.loc[valid.index, "cookie_id"]
        out[f"{prefix}_switch_rate"] = valid.groupby(ids).mean()
        # Mean run length = how long a cookie stays inside one context before moving on.
        run_id = valid.astype(bool).groupby(ids).cumsum()
        runs = valid.groupby([ids, run_id]).size()
        out[f"{prefix}_run_mean"] = runs.groupby(level=0).mean()
        out[f"{prefix}_run_max"] = runs.groupby(level=0).max()

    # Funnel: how many listings each search turns into, and how many turn into contact.
    counts = events.groupby(["cookie_id", "event_name"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(config.EVENT_NAMES), fill_value=0)
    out["funnel_items_per_search"] = counts["item_view"] / (counts["search_results_view"] + 1.0)
    out["funnel_contact_per_item"] = (
        counts[list(config.CONTACT_EVENTS)].sum(axis=1) / (counts["item_view"] + 1.0)
    )
    out["funnel_seller_per_item"] = counts["seller_page_view"] / (counts["item_view"] + 1.0)
    out["funnel_swipe_per_item"] = counts["photo_swipe"] / (counts["item_view"] + 1.0)

    # Consecutive photo swipes: a human flips through a gallery, a scraper does not.
    swipes = events["event_name"].astype(str).eq("photo_swipe")
    breaks = (~swipes).groupby(events["cookie_id"]).cumsum()
    runs = swipes.groupby([events["cookie_id"], breaks]).sum()
    out["swipe_run_max"] = runs.groupby(level=0).max()

    sellers = events.dropna(subset=["item_id"])
    out["item_revisit_max"] = sellers.groupby(["cookie_id", "item_id"]).size().groupby(level=0).max()
    return out


def _scoped_block(
    events: pd.DataFrame,
    platforms: tuple[str, ...],
    builders: list,
    prefix: str,
) -> pd.DataFrame:
    """Recompute a set of blocks on one platform family only.

    Platform families differ in which fields they populate, so a cookie's timing and
    navigation on mobile are a different measurement from the same statistics on web.
    """
    index = events["cookie_id"].drop_duplicates().sort_values()
    subset = events[events["platform"].astype(str).isin(platforms)]
    if subset.empty:
        return pd.DataFrame(index=index)
    frame = pd.concat([build(subset) for build in builders], axis=1)
    frame.columns = [f"{prefix}{c}" for c in frame.columns]
    return frame.reindex(index)


def _block_pointer_web(events: pd.DataFrame) -> pd.DataFrame:
    """Pointer statistics restricted to desktop/web, where the field is populated."""
    return _scoped_block(events, ("web", "desktop"), [_block_pointer], "webp_")


def _block_pointer_deep(events: pd.DataFrame) -> pd.DataFrame:
    """Trajectory geometry of the cursor — the single strongest family in ablation."""
    index = events["cookie_id"].drop_duplicates().sort_values()
    pointer = events.dropna(subset=["pointer_x", "pointer_y"]).copy()
    out = pd.DataFrame(index=index)
    if pointer.empty:
        return out
    g = pointer.groupby("cookie_id")

    pointer["dx"] = g["pointer_x"].diff()
    pointer["dy"] = g["pointer_y"].diff()
    pointer["step"] = np.hypot(pointer["dx"], pointer["dy"])
    moves = pointer.dropna(subset=["step"])
    gm = moves.groupby("cookie_id")

    steps = gm["step"]
    quantiles = steps.quantile([0.1, 0.5, 0.9]).unstack()
    quantiles.columns = [f"ptrd_step_q{int(q * 100)}" for q in quantiles.columns]
    out = out.join(quantiles)
    bucketed = moves.assign(bucket=np.floor(np.log1p(moves["step"])).astype(int))
    out["ptrd_step_entropy"] = _entropy(bucketed, "bucket")
    out["ptrd_zero_step_share"] = (moves["step"] == 0).groupby(moves["cookie_id"]).mean()

    # Straight-line travel is a machine tell: net displacement over path length.
    net = np.hypot(
        g["pointer_x"].last() - g["pointer_x"].first(),
        g["pointer_y"].last() - g["pointer_y"].first(),
    )
    out["ptrd_straightness"] = net / (steps.sum() + 1.0)
    out["ptrd_path_len"] = steps.sum()

    angle = np.arctan2(moves["dy"], moves["dx"])
    turn = angle.groupby(moves["cookie_id"]).diff().abs()
    turn = np.minimum(turn, 2 * np.pi - turn).dropna()
    turn_ids = moves.loc[turn.index, "cookie_id"]
    out["ptrd_turn_med"] = turn.groupby(turn_ids).median()
    out["ptrd_turn_std"] = turn.groupby(turn_ids).std()
    out["ptrd_reversal_share"] = (turn > np.pi / 2).groupby(turn_ids).mean()

    # Synthetic coordinates often snap to a grid or pile up on the screen edges.
    for axis in ("pointer_x", "pointer_y"):
        short = axis.split("_")[1]
        out[f"ptrd_{short}_mod10"] = (pointer[axis] % 10 == 0).groupby(
            pointer["cookie_id"]
        ).mean()
    limits = {"pointer_x": 1920.0, "pointer_y": 1080.0}
    edge = (
        (pointer["pointer_x"] <= 1)
        | (pointer["pointer_y"] <= 1)
        | (pointer["pointer_x"] >= limits["pointer_x"] - 1)
        | (pointer["pointer_y"] >= limits["pointer_y"] - 1)
    )
    out["ptrd_edge_share"] = edge.groupby(pointer["cookie_id"]).mean()

    centre = np.hypot(
        pointer["pointer_x"] - limits["pointer_x"] / 2,
        pointer["pointer_y"] - limits["pointer_y"] / 2,
    )
    out["ptrd_centre_dist_med"] = centre.groupby(pointer["cookie_id"]).median()
    out["ptrd_centre_dist_std"] = centre.groupby(pointer["cookie_id"]).std()

    quadrant = (pointer["pointer_x"] > limits["pointer_x"] / 2).astype(int) * 2 + (
        pointer["pointer_y"] > limits["pointer_y"] / 2
    ).astype(int)
    quad_frame = pointer.assign(quadrant=quadrant)
    out["ptrd_quadrants"] = quad_frame.groupby("cookie_id")["quadrant"].nunique()
    out["ptrd_quadrant_entropy"] = _entropy(quad_frame, "quadrant")

    out["ptrd_xy_corr"] = _grouped_corr(pointer, "pointer_x", "pointer_y")
    out["ptrd_bbox_fill"] = (
        (g["pointer_x"].max() - g["pointer_x"].min())
        * (g["pointer_y"].max() - g["pointer_y"].min())
    ) / (limits["pointer_x"] * limits["pointer_y"])
    out["ptrd_repeat_share"] = 1.0 - (
        pointer.groupby(["cookie_id", "pointer_x", "pointer_y"]).size().groupby(level=0).size()
        / g.size()
    )

    # Which event types carry a cursor at all, and how fast it moves between them.
    pointer["dt"] = g["event_ts"].diff().dt.total_seconds()
    speed = (pointer["step"] / (pointer["dt"] + 1.0)).dropna()
    speed_ids = pointer.loc[speed.index, "cookie_id"]
    out["ptrd_speed_q10"] = speed.groupby(speed_ids).quantile(0.1)
    out["ptrd_speed_q90"] = speed.groupby(speed_ids).quantile(0.9)
    out["ptrd_speed_std"] = speed.groupby(speed_ids).std()
    for name in ("item_view", "search_results_view", "photo_swipe"):
        mask = events["event_name"].astype(str) == name
        subset = events[mask]
        if subset.empty:
            continue
        out[f"ptrd_cover_{name}"] = (
            subset.groupby("cookie_id")["pointer_x"].count() / subset.groupby("cookie_id").size()
        )
    return out


# Features worth judging relative to peers rather than on an absolute scale.
_RELATIVE_BASE = (
    "n_events",
    "dt_q50",
    "dt_iqr",
    "dtg_uniq_frac",
    "sess_events_mean",
    "item_location_nuniq",
    "item_category_nuniq",
    "item_id_nuniq",
    "page_mean",
    "page_max",
    "rate_item_view",
    "rate_engagement",
    "funnel_items_per_search",
    "webp_pointer_coverage",
    "nav_loc_switch_rate",
    "dwell_item_view_med",
    "ptrd_straightness",
    "events_per_h",
    "idst_range",
    "ua_freq_mean",
)


def _block_relative(
    matrix: pd.DataFrame, fit_ids: pd.Index | None = None
) -> pd.DataFrame:
    """Percentile of each core feature among same-platform peers.

    Absolute thresholds mean different things on mobile and on web, so a rank inside
    the platform peer group is the comparison a human analyst would actually make.
    The reference distribution comes from the fitting cookies alone and every cookie is
    mapped through it, so a scored cookie's percentile does not depend on which other
    cookies happen to be scored beside it.

    The same trick keyed on the window day was tested and rejected: it lifted random
    CV but cost 0.03 P@R70 on the forward-chaining check, because a within-day rank
    encodes that day's population rather than the cookie.
    """
    platform_columns = [f"plat_{p}" for p in config.PLATFORMS]
    dominant = matrix[platform_columns].idxmax(axis=1)
    fit_mask = (
        np.ones(len(matrix), bool) if fit_ids is None else matrix.index.isin(fit_ids)
    )
    out = pd.DataFrame(index=matrix.index)
    for column in (c for c in _RELATIVE_BASE if c in matrix.columns):
        values = matrix[column].to_numpy(dtype="float64")
        ranks = np.full(len(matrix), np.nan)
        for platform in dominant.unique():
            group = (dominant == platform).to_numpy()
            reference = np.sort(values[group & fit_mask])
            reference = reference[~np.isnan(reference)]
            if reference.size == 0:
                continue
            ranks[group] = np.searchsorted(reference, values[group], side="right") / reference.size
        ranks[np.isnan(values)] = np.nan
        out[f"rel_plat_{column}"] = ranks
    out["dominant_platform"] = pd.Categorical(dominant).codes
    return out


def _event_documents(events: pd.DataFrame) -> pd.DataFrame:
    """Two token streams per cookie: pure event order, and order plus pacing."""
    frame = events[["cookie_id", "event_name", "event_ts"]].copy()
    frame["code"] = frame["event_name"].map(_EVENT_CODE).astype(str)
    dt = frame.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    bucket = pd.cut(
        dt, [-1, 3, 10, 30, 120, np.inf], labels=["0", "1", "2", "3", "4"]
    ).astype(str)
    frame["paced"] = frame["code"] + bucket.fillna("s")
    grouped = frame.groupby("cookie_id")
    return pd.DataFrame(
        {
            "doc_order": grouped["code"].apply(" ".join),
            "doc_paced": grouped["paced"].apply(" ".join),
        }
    )


def _block_sequence_svd(
    events: pd.DataFrame, fit_ids: pd.Index | None
) -> pd.DataFrame:
    docs = _event_documents(events)
    fit_mask = docs.index.isin(fit_ids) if fit_ids is not None else np.ones(len(docs), bool)
    parts: list[pd.DataFrame] = []
    specs = [("doc_order", (1, 3), "seqo"), ("doc_paced", (1, 2), "seqp")]
    for column, ngrams, prefix in specs:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            ngram_range=ngrams,
            min_df=20,
            max_features=400,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(docs.loc[fit_mask, column])
        svd = TruncatedSVD(
            n_components=config.SVD_COMPONENTS // 2, random_state=config.SEED
        )
        svd.fit(matrix)
        transformed = svd.transform(vectorizer.transform(docs[column]))
        parts.append(
            pd.DataFrame(
                transformed,
                index=docs.index,
                columns=[f"{prefix}_{i}" for i in range(transformed.shape[1])],
            )
        )
    return pd.concat(parts, axis=1)


def top_bigrams(
    events: pd.DataFrame, fit_ids: pd.Index | None = None, k: int = config.TOP_BIGRAMS
) -> list[str]:
    """Vocabulary counted on the fitting cookies only, never on the scored ones."""
    if fit_ids is not None:
        events = events[events["cookie_id"].isin(fit_ids)]
    seq = events[["cookie_id", "event_name"]].copy()
    seq["prev"] = seq.groupby("cookie_id", observed=True)["event_name"].shift()
    pairs = seq.dropna(subset=["prev"])
    bigrams = pairs["prev"].astype(str) + ">" + pairs["event_name"].astype(str)
    return bigrams.value_counts().head(k).index.tolist()


def build_features(
    events: pd.DataFrame,
    meta: pd.DataFrame,
    fit_ids: pd.Index | None = None,
    bigrams: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble every block for the cookies present in `meta`.

    `fit_ids` restricts every fitted part — TF-IDF/SVD, the bigram and catalogue
    vocabularies, the peer-rank reference distributions — to those cookies, so nothing
    a scored cookie contains can influence how it is described.
    """
    bigrams = bigrams if bigrams is not None else top_bigrams(events, fit_ids)
    blocks = [
        _block_volume_and_mix(events),
        _block_transitions(events, bigrams),
        _block_timing(events),
        _block_sessions(events),
        _block_content(events),
        _block_pagination(events),
        _block_pointer(events),
        _block_client(events),
        _block_id_structure(events),
        _block_dt_granularity(events),
        _block_catalog_shares(events, fit_ids),
        _block_popularity(events, fit_ids),
        _block_crowd(events, fit_ids),
        _block_conditional_timing(events),
        _block_navigation(events),
        _block_pointer_web(events),
        _block_pointer_deep(events),
        # No mobile copy of the time-series block: a mobile-dominant cookie has
        # essentially no non-mobile events, so the copy repeats the global column and
        # only dilutes column sampling (PR-AUC 0.7959 with it, 0.7993 without).
        _scoped_block(events, ("android", "ios"), [_block_timing, _block_navigation], "mob_"),
        _scoped_block(events, ("web", "desktop"), [_block_timing], "web_"),
        timeseries.build_block(events),
        _block_window_coverage(events, meta),
        _block_sequence_svd(events, fit_ids),
    ]
    features = pd.concat(blocks, axis=1)
    features = features.join(_block_cookie_meta(meta), how="right")
    features = features.loc[meta["cookie_id"]]
    return pd.concat([features, _block_relative(features, fit_ids)], axis=1)
