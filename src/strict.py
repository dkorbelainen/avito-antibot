"""The strictest protocol: rebuild the features for every temporal split.

`validate.forward_chain` slices a matrix that was built once over the whole training
set, so a validation cookie is still described by vocabularies, peer-rank references and
listing audiences that were counted with later days present. For most blocks that is
harmless. For any statistic computed over the population it is not: F38 shows a block
that gains 0.11 P@R70 under the ordinary protocol and 0.028 under this one.

Here the matrix is rebuilt from the earlier period alone, so nothing the scored period
contains can reach a feature. This is the situation the hidden test is in.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, data, features, validate
from .model import ModelSpec, estimate_rounds, fit_predict, rank_average

def _matrix(
    events: pd.DataFrame, meta: pd.DataFrame, fit_ids: pd.Index, pool: pd.Series
) -> pd.DataFrame:
    matrix = features.build_features(events, meta, fit_ids=fit_ids, pool=pool)
    matrix = matrix.astype("float64").replace([np.inf, -np.inf], np.nan)
    return matrix.loc[:, ~matrix.T.duplicated().to_numpy()]


def split_report(
    cut_day: int, seeds: int = 3, drop: str = "", pooled: bool = True
) -> dict[str, float]:
    """Train on the days before `cut_day`, score the days from it on."""
    train, test = data.load_splits()
    meta = pd.concat([train.drop(columns=["target"]), test], ignore_index=True)
    events = data.clip_to_window(data.load_events(), meta)

    day = (train["window_start_ts"] - train["window_start_ts"].min()).dt.days
    fit_ids = pd.Index(train.loc[day < cut_day, "cookie_id"])
    score_ids = pd.Index(train.loc[day >= cut_day, "cookie_id"])

    meta_day = (meta["window_start_ts"] - train["window_start_ts"].min()).dt.days
    period = (meta_day // config.POOL_SPAN_DAYS).astype(str) if pooled else pd.Series("all", index=meta.index)
    pool = pd.Series(period.to_numpy(), index=pd.Index(meta["cookie_id"], name="cookie_id"))

    matrix = _matrix(events, meta, fit_ids, pool)
    if drop:
        matrix = matrix.drop(columns=matrix.filter(regex=drop).columns)

    y = train.set_index("cookie_id")["target"]
    x_fit, y_fit = matrix.loc[fit_ids], y.loc[fit_ids]
    x_val, y_val = matrix.loc[score_ids], y.loc[score_ids]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(x_fit, y_fit))
    spec = ModelSpec("lgb", estimate_rounds("lgb", x_fit, y_fit, folds[:3]))
    pred = rank_average(
        [fit_predict(spec, x_fit, y_fit, x_val, seed=config.SEED + i) for i in range(seeds)]
    )
    return validate.score(y_val, pred) | {"n_features": float(matrix.shape[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="strict")
    parser.add_argument("--cut", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--drop", default="", help="regex of columns to remove")
    parser.add_argument("--no-pool", action="store_true", help="one pool for every cookie")
    args = parser.parse_args()

    report = split_report(args.cut, args.seeds, args.drop, pooled=not args.no_pool)
    print(
        f"{args.name:<28} P@R70 {report['p_at_r70']:.4f}  ROC {report['roc_auc']:.4f}  "
        f"PR {report['pr_auc']:.4f}  R@FPR1 {report['recall_at_fpr1']:.4f}  "
        f"n {int(report['n_features'])}"
    )
    validate.log_result(
        f"strict_{args.name}",
        {f"{k}_mean": v for k, v in report.items()},
        {"cut": args.cut, "seeds": args.seeds, "drop": args.drop, "pooled": not args.no_pool},
    )


if __name__ == "__main__":
    main()
