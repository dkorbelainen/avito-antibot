"""CLI runner: one named experiment per invocation, appended to results.jsonl."""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import ModelSpec
from .pipeline import build_dataset


def estimate_rounds(
    x: pd.DataFrame, y: pd.Series, params: dict[str, object], max_rounds: int = 3000
) -> int:
    """Best iteration averaged over folds, used as the fixed round count later."""
    best: list[int] = []
    splitter = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    for train_idx, valid_idx in splitter.split(x, y):
        booster = lgb.train(
            {**params, "seed": config.SEED, "metric": "average_precision"},
            lgb.Dataset(x.iloc[train_idx], y.iloc[train_idx]),
            num_boost_round=max_rounds,
            valid_sets=[lgb.Dataset(x.iloc[valid_idx], y.iloc[valid_idx])],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        best.append(booster.best_iteration)
    return int(np.median(best))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", default="lgb", choices=["lgb", "cat", "xgb"])
    parser.add_argument("--rounds", type=int, default=0, help="0 = estimate by early stopping")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--features", default="", help="regex of feature names to keep")
    args = parser.parse_args()

    dataset = build_dataset(cache=not args.no_cache)
    x, y = dataset.x_train, dataset.y_train
    if args.features:
        x = x.filter(regex=args.features)

    rounds = args.rounds or estimate_rounds(x, y, ModelSpec(args.kind).resolved(config.SEED))
    spec = ModelSpec(kind=args.kind, n_rounds=rounds)

    report = validate.repeated_cv(spec, x, y, n_seeds=args.seeds)
    chain = None if args.no_chain else validate.forward_chain(spec, x, y, dataset.day_index)
    print(validate.format_report(args.name, report, chain))
    validate.log_result(
        args.name,
        report,
        {"kind": args.kind, "rounds": rounds, "n_features": x.shape[1], "seeds": args.seeds}
        | (chain or {}),
    )


if __name__ == "__main__":
    main()
