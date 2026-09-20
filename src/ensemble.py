"""Does the blend earn its cost? Every subset of the three model families, measured.

The out-of-fold predictions are computed once per family and cached, so the subset
comparison itself is free and every combination is scored on exactly the same folds and
the same seeds. Weights come from a coarse grid on PR-AUC, which is the stable half of
the metric pair.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import Kind, ModelSpec, estimate_rounds, rank_average
from .pipeline import build_dataset, feature_code_hash

KINDS: tuple[Kind, ...] = ("lgb", "cat", "xgb")


def spec_hash(spec: ModelSpec) -> str:
    """Cache key for the fitted model itself, so a parameter change cannot be served
    an array produced under the previous parameters (F50)."""
    payload = json.dumps(
        {"kind": spec.kind, "rounds": spec.n_rounds, "params": spec.resolved(config.SEED)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def load_spec(kind: Kind, x: pd.DataFrame, y: pd.Series, folds: list, tuned: bool) -> ModelSpec:
    """Tuned parameters when a search has been run for this family, defaults otherwise."""
    path = config.ROOT / f"params_{kind}.json"
    if tuned and path.exists():
        best = json.loads(path.read_text())
        return ModelSpec(kind=kind, n_rounds=int(best["rounds"]), params=best["params"])
    return ModelSpec(kind=kind, n_rounds=estimate_rounds(kind, x, y, folds[:3]))


def search_weights(
    oof: dict[Kind, np.ndarray], y: pd.Series, steps: int = 5
) -> tuple[dict[Kind, float], dict[str, float]]:
    """Coarse simplex search; ties are broken toward the flatter weight vector."""
    kinds = list(oof)
    best_weights: tuple[int, ...] = ()
    best_score = -1.0
    for weights in itertools.product(range(steps), repeat=len(kinds)):
        if sum(weights) == 0:
            continue
        blended = rank_average([oof[k] for k in kinds], [float(w) for w in weights])
        value = validate.score(y, blended)["pr_auc"]
        if value > best_score:
            best_weights, best_score = weights, value
    total = sum(best_weights)
    chosen = {k: w / total for k, w in zip(kinds, best_weights, strict=True)}
    blended = rank_average([oof[k] for k in kinds], [chosen[k] for k in kinds])
    return chosen, validate.score(y, blended)


def family_oof(
    kinds: tuple[Kind, ...], seeds: int, tuned: bool, refresh: bool
) -> tuple[dict[Kind, np.ndarray], pd.Series, dict[Kind, ModelSpec]]:
    dataset = build_dataset()
    x, y = dataset.x_train, dataset.y_train
    folds = list(
        StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y)
    )
    tag = "tuned" if tuned else "default"
    oof: dict[Kind, np.ndarray] = {}
    specs: dict[Kind, ModelSpec] = {}
    for kind in kinds:
        spec = load_spec(kind, x, y, folds, tuned)
        specs[kind] = spec
        # Neither the column count nor the feature hash is a safe cache key on its own:
        # a feature block can be rewritten without changing the width, and the model
        # parameters can change without touching a feature at all. Both go in the name.
        cache = (
            config.CACHE_DIR
            / f"oof_{kind}_{tag}_{seeds}seeds_{x.shape[1]}f"
            f"_{feature_code_hash()}_{spec_hash(spec)}.npy"
        )
        if cache.exists() and not refresh:
            oof[kind] = np.load(cache)
            print(f"{kind}: cached {cache.name}")
            continue
        report = validate.repeated_cv(spec, x, y, n_seeds=seeds)
        oof[kind] = np.asarray(report["oof"])
        np.save(cache, oof[kind])
        print(validate.format_report(f"{kind} (rounds={spec.n_rounds})", report))
        validate.log_result(f"solo_{kind}_{tag}", report, {"kind": kind, "rounds": spec.n_rounds})
    return oof, y, specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    parser.add_argument("--tuned", action="store_true", help="use params_<kind>.json")
    parser.add_argument("--refresh", action="store_true", help="ignore cached OOF")
    parser.add_argument("--name", default="ens")
    args = parser.parse_args()

    oof, y, _ = family_oof(KINDS, args.seeds, args.tuned, args.refresh)

    rows: list[dict[str, object]] = []
    for size in (1, 2, 3):
        for subset in itertools.combinations(KINDS, size):
            part = {k: oof[k] for k in subset}
            weights, report = search_weights(part, y)
            rows.append(
                {
                    "combination": "+".join(subset),
                    "weights": " ".join(f"{k}:{weights[k]:.2f}" for k in subset),
                    **{k: round(v, 4) for k, v in report.items()},
                }
            )
            validate.log_result(
                f"{args.name}_{'_'.join(subset)}",
                {f"{k}_mean": v for k, v in report.items()},
                {"weights": weights, "tuned": args.tuned, "seeds": args.seeds},
            )
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
