"""Final fit and submission writing.

OOF predictions decide the blend weights; the shipped scores come from models
refitted on the whole training set and bagged over seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import Kind, ModelSpec, estimate_rounds, fit_predict, rank_average
from .pipeline import build_dataset, write_submission


def load_spec(kind: Kind, x: pd.DataFrame, y: pd.Series, folds: list) -> ModelSpec:
    path = config.ROOT / f"params_{kind}.json"
    if path.exists():
        tuned = json.loads(path.read_text())
        return ModelSpec(kind=kind, n_rounds=int(tuned["rounds"]), params=tuned["params"])
    return ModelSpec(kind=kind, n_rounds=estimate_rounds(kind, x, y, folds))


def search_weights(oof: dict[Kind, np.ndarray], y: pd.Series) -> dict[Kind, float]:
    """Coarse simplex search on PR-AUC — the stable half of the metric pair."""
    kinds = list(oof)
    grid = [w for w in itertools.product(range(5), repeat=len(kinds)) if sum(w) > 0]
    best_weights, best_score = None, -1.0
    for weights in grid:
        blended = rank_average([oof[k] for k in kinds], list(map(float, weights)))
        value = validate.score(y, blended)["pr_auc"]
        if value > best_score:
            best_weights, best_score = weights, value
    return {k: float(w) for k, w in zip(kinds, best_weights, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", default="lgb,cat,xgb")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument("--name", default="final")
    args = parser.parse_args()

    kinds: list[Kind] = [k.strip() for k in args.kinds.split(",")]
    dataset = build_dataset()
    x, y, x_test = dataset.x_train, dataset.y_train, dataset.x_test
    folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y))

    specs = {kind: load_spec(kind, x, y, folds) for kind in kinds}
    oof: dict[Kind, np.ndarray] = {}
    for kind, spec in specs.items():
        report = validate.repeated_cv(spec, x, y, n_seeds=args.seeds)
        oof[kind] = report["oof"]
        print(validate.format_report(kind, report))
        validate.log_result(f"{args.name}_{kind}", report, {"kind": kind, "rounds": spec.n_rounds})

    weights = search_weights(oof, y) if len(kinds) > 1 else {kinds[0]: 1.0}
    blended_oof = rank_average([oof[k] for k in kinds], [weights[k] for k in kinds])
    blend_report = validate.score(y, blended_oof)
    print(f"blend weights {weights} -> " + " ".join(f"{k} {v:.4f}" for k, v in blend_report.items()))
    validate.log_result(f"{args.name}_blend", {f"{k}_mean": v for k, v in blend_report.items()} | {f"{k}_std": 0.0 for k in blend_report}, {"weights": weights})

    test_predictions: list[np.ndarray] = []
    for kind, spec in specs.items():
        bagged = [
            fit_predict(spec, x, y, x_test, seed=config.SEED + offset)
            for offset in range(args.seeds)
        ]
        test_predictions.append(rank_average(bagged))
    scores = rank_average(test_predictions, [weights[k] for k in kinds])

    submission = write_submission(dataset.test_ids, scores)
    digest = hashlib.md5(config.SUBMISSION_PATH.read_bytes()).hexdigest()
    print(f"submission rows {len(submission)} md5 {digest}")
    np.save(config.CACHE_DIR / f"oof_{args.name}.npy", blended_oof)


if __name__ == "__main__":
    main()
