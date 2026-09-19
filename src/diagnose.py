"""Where the model wins and loses: metrics split by cookie regime."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import ModelSpec, estimate_rounds
from .pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--kind", default="lgb")
    args = parser.parse_args()

    dataset = build_dataset()
    x, y = dataset.x_train, dataset.y_train
    folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y))
    spec = ModelSpec(args.kind, estimate_rounds(args.kind, x, y, folds))
    report = validate.repeated_cv(spec, x, y, n_seeds=args.seeds)
    oof = report["oof"]
    print(validate.format_report("full", report))

    regimes = {
        "has_pointer": x["pointer_coverage"].fillna(0) > 0,
        "no_pointer": x["pointer_coverage"].fillna(0) == 0,
        "mobile_major": x[["plat_android", "plat_ios"]].sum(axis=1) > 0.5,
        "web_major": x[["plat_web", "plat_desktop"]].sum(axis=1) > 0.5,
        "few_events": x["n_events"] <= 8,
        "many_events": x["n_events"] > 23,
    }
    rows = []
    for name, mask in regimes.items():
        mask = mask.to_numpy()
        if y[mask].sum() < 5:
            continue
        stats = validate.score(y[mask], oof[mask])
        rows.append({"regime": name, "n": int(mask.sum()), "bots": int(y[mask].sum()), **stats})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # Where do the misses sit? Bots ranked below the global 70%-recall threshold.
    order = np.argsort(-oof)
    cumulative = np.cumsum(y.to_numpy()[order])
    cutoff_rank = int(np.searchsorted(cumulative, 0.7 * y.sum()))
    threshold = oof[order][cutoff_rank]
    missed = (y.to_numpy() == 1) & (oof < threshold)
    caught = (y.to_numpy() == 1) & (oof >= threshold)
    compare = pd.DataFrame(
        {
            "missed_bot": x[missed].mean(numeric_only=True),
            "caught_bot": x[caught].mean(numeric_only=True),
            "human": x[y.to_numpy() == 0].mean(numeric_only=True),
        }
    )
    compare["gap"] = (compare["missed_bot"] - compare["caught_bot"]).abs() / (
        compare["caught_bot"].abs() + 1e-6
    )
    print("\ntop separating features between caught and missed bots:")
    print(compare.sort_values("gap", ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
