"""Final fit and submission writing.

One family, LightGBM, chosen by measurement rather than by default: every subset of the
three gradient boosting families was scored out-of-fold on identical folds and the
weight search gave CatBoost and XGBoost zero weight (see `src/ensemble.py`). What does
pay is bagging over seeds — the same model refitted under several seeds and averaged in
rank space, which is worth about 0.006 PR-AUC over a single fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import Kind, ModelSpec, estimate_rounds, fit_predict, rank_average
from .pipeline import build_dataset, write_submission

# More seeds than the validation uses: the shipped scores are produced once, so the
# variance reduction is free, and the ranking is what the metric reads.
BAG_SEEDS = 10


def load_spec(kind: Kind, x: pd.DataFrame, y: pd.Series, folds: list) -> ModelSpec:
    """Tuned parameters when a search has been run for this family, defaults otherwise."""
    path = config.ROOT / f"params_{kind}.json"
    if path.exists():
        best = json.loads(path.read_text())
        return ModelSpec(kind=kind, n_rounds=int(best["rounds"]), params=best["params"])
    return ModelSpec(kind=kind, n_rounds=estimate_rounds(kind, x, y, folds[:3]))


def bagged_predictions(
    spec: ModelSpec, x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, bag: int
) -> np.ndarray:
    """Refit on the whole training set under `bag` seeds, averaged in rank space."""
    return rank_average(
        [fit_predict(spec, x, y, x_test, seed=config.SEED + offset) for offset in range(bag)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="lgb", choices=["lgb", "cat", "xgb"])
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument("--bag", type=int, default=BAG_SEEDS)
    parser.add_argument("--name", default="final")
    args = parser.parse_args()

    dataset = build_dataset()
    x, y, x_test = dataset.x_train, dataset.y_train, dataset.x_test
    folds = list(
        StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y)
    )
    spec = load_spec(args.kind, x, y, folds)

    report = validate.repeated_cv(spec, x, y, n_seeds=args.seeds)
    chain = validate.forward_chain(spec, x, y, dataset.day_index)
    print(validate.format_report(f"{args.kind} (rounds={spec.n_rounds})", report, chain))
    validate.log_result(
        args.name,
        report,
        {"kind": args.kind, "rounds": spec.n_rounds, "n_features": x.shape[1], "bag": args.bag}
        | chain,
    )

    scores = bagged_predictions(spec, x, y, x_test, args.bag)
    submission = write_submission(dataset.test_ids, scores)
    digest = hashlib.md5(config.SUBMISSION_PATH.read_bytes()).hexdigest()
    print(f"submission rows {len(submission)} md5 {digest}")
    np.save(config.CACHE_DIR / f"oof_{args.name}.npy", np.asarray(report["oof"]))


if __name__ == "__main__":
    main()
