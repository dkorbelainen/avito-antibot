"""CLI runner: one named experiment per invocation, appended to results.jsonl."""

from __future__ import annotations

import argparse

from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import ModelSpec, estimate_rounds
from .pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", default="lgb", choices=["lgb", "cat", "xgb"])
    parser.add_argument("--rounds", type=int, default=0, help="0 = estimate by early stopping")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--block-chain", action="store_true", help="one 7/7 day split instead of per-day")
    parser.add_argument("--features", default="", help="regex of feature names to keep")
    parser.add_argument("--select-k", type=int, default=0, help="keep top-k by in-fold gain")
    args = parser.parse_args()

    dataset = build_dataset(cache=not args.no_cache)
    x, y = dataset.x_train, dataset.y_train
    if args.features:
        x = x.filter(regex=args.features)

    folds = list(
        StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y)
    )
    rounds = args.rounds or estimate_rounds(args.kind, x, y, folds[:3])
    spec = ModelSpec(kind=args.kind, n_rounds=rounds)

    report = validate.repeated_cv(spec, x, y, n_seeds=args.seeds, select_k=args.select_k)
    chain = (
        None
        if args.no_chain
        else validate.forward_chain(spec, x, y, dataset.day_index, block=args.block_chain)
    )
    print(validate.format_report(args.name, report, chain))
    validate.log_result(
        args.name,
        report,
        {
            "kind": args.kind,
            "rounds": rounds,
            "n_features": x.shape[1],
            "select_k": args.select_k,
            "seeds": args.seeds,
        }
        | (chain or {}),
    )


if __name__ == "__main__":
    main()
