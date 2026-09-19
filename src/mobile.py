"""Mobile specialist and the two ways of folding it into one global ranking.

The metric is computed on a single ranking over all cookies, but the population is not
one population. Web cookies carry a cursor trace and are close to solved (P@R70 ~0.87
inside that subgroup); android/ios cookies carry no cursor at all and are where the
score is actually lost (P@R70 ~0.36). A model fitted on everything spends its capacity
where the rows are, which is not where the errors are.

This module fits a second model on the mobile rows alone and offers two ways of using
it, both evaluated by the usual gate:

* **reorder** — keep every mobile cookie's global score value, but permute those values
  among the mobile cookies according to the blended ranking. Mobile cookies keep their
  collective position against web cookies, so only the part the specialist actually
  knows about can change. Conservative and impossible to make worse by accident.
* **stack** — a small second-level model over the two scores plus the subgroup flag,
  which can also move the subgroup as a whole. Stronger, and it has to earn it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config
from .model import Kind, ModelSpec, fit_predict

# A column this sparse inside the subgroup describes almost nobody in it.
MIN_SUBGROUP_COVERAGE = 0.02


def mobile_mask(x: pd.DataFrame) -> pd.Series:
    """Cookies whose events are mostly android/ios."""
    return x[["plat_android", "plat_ios"]].sum(axis=1) > 0.5


def subgroup_columns(x: pd.DataFrame, mask: pd.Series) -> list[str]:
    """Drop columns that are empty or constant inside the subgroup.

    Cursor geometry is entirely absent on mobile; carrying those columns only dilutes
    the specialist's column sampling.
    """
    inside = x.loc[mask.to_numpy()]
    coverage = inside.notna().mean()
    varying = inside.nunique(dropna=True) > 1
    return [c for c in x.columns if coverage[c] >= MIN_SUBGROUP_COVERAGE and varying[c]]


def _ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


def specialist_oof(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
    mask: pd.Series,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> np.ndarray:
    """Out-of-fold specialist scores, NaN outside the subgroup.

    The folds are stratified inside the subgroup, so every fold carries bots even
    though the subgroup holds only about 6% positives.
    """
    inside = mask.to_numpy()
    columns = subgroup_columns(x, mask)
    x_in, y_in = x.loc[inside, columns], y.loc[inside]
    scores = np.full(len(x), np.nan)
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x_in, y_in)
    part = np.zeros(len(x_in))
    for train_idx, valid_idx in folds:
        part[valid_idx] = fit_predict(
            spec, x_in.iloc[train_idx], y_in.iloc[train_idx], x_in.iloc[valid_idx], seed=seed
        )
    scores[inside] = part
    return scores


def reorder(global_scores: np.ndarray, specialist: np.ndarray, weight: float) -> np.ndarray:
    """Permute the subgroup's own score values by the blended within-subgroup ranking.

    `weight` is the specialist's share of the blend, 0 leaving the ranking untouched
    and 1 handing the subgroup entirely to the specialist.
    """
    inside = ~np.isnan(specialist)
    if not inside.any():
        return global_scores
    out = global_scores.copy()
    blended = (1 - weight) * _ranks(global_scores[inside]) + weight * _ranks(specialist[inside])
    values = np.sort(global_scores[inside])
    out[inside] = values[np.argsort(np.argsort(blended))]
    return out


def stack_features(
    global_scores: np.ndarray, specialist: np.ndarray, x: pd.DataFrame
) -> pd.DataFrame:
    """Second-level inputs: both scores as ranks, the subgroup flag and its size cue."""
    inside = ~np.isnan(specialist)
    specialist_rank = np.full(len(x), np.nan)
    specialist_rank[inside] = _ranks(specialist[inside])
    return pd.DataFrame(
        {
            "global_rank": _ranks(global_scores),
            "specialist_rank": specialist_rank,
            "is_mobile": inside.astype(float),
            "n_events_log": x["n_events_log"].to_numpy(),
        }
    )


def stack_oof(
    features: pd.DataFrame,
    y: pd.Series,
    kind: Kind = "lgb",
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED + 100,
) -> np.ndarray:
    """Cross-fitted second level, deliberately tiny — four inputs cannot carry more."""
    spec = ModelSpec(
        kind=kind,
        n_rounds=200,
        params={"num_leaves": 7, "learning_rate": 0.05, "min_data_in_leaf": 200},
    )
    scores = np.zeros(len(features))
    for train_idx, valid_idx in StratifiedKFold(
        n_folds, shuffle=True, random_state=seed
    ).split(features, y):
        scores[valid_idx] = fit_predict(
            spec,
            features.iloc[train_idx],
            y.iloc[train_idx],
            features.iloc[valid_idx],
            seed=seed,
        )
    return scores


def _report(name: str, y: pd.Series, scores: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    from . import validate

    overall = validate.score(y, scores)
    inside = validate.score(y[mask], scores[mask])
    print(
        f"{name:28s} P@R70 {overall['p_at_r70']:.4f}  PR {overall['pr_auc']:.4f}  "
        f"ROC {overall['roc_auc']:.4f}  | mobile P@R70 {inside['p_at_r70']:.4f}  "
        f"PR {inside['pr_auc']:.4f}"
    )
    return {f"{k}_mean": v for k, v in overall.items()} | {
        f"mobile_{k}": v for k, v in inside.items()
    }


def main() -> None:
    import argparse

    from . import validate
    from .model import estimate_rounds
    from .pipeline import build_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="M1_mobile")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS)
    args = parser.parse_args()

    dataset = build_dataset()
    x, y = dataset.x_train, dataset.y_train
    mask = mobile_mask(x)
    inside = mask.to_numpy()
    print(f"mobile rows {inside.sum()} of {len(x)}, bots {int(y[inside].sum())}")

    folds = list(
        StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y)
    )
    base = ModelSpec(kind="lgb", n_rounds=estimate_rounds("lgb", x, y, folds[:3]))
    global_report = validate.repeated_cv(base, x, y, n_seeds=args.seeds)
    global_scores = np.asarray(global_report["oof"])
    _report("global", y, global_scores, inside)

    columns = subgroup_columns(x, mask)
    print(f"specialist columns {len(columns)} of {x.shape[1]}")
    sub_folds = list(
        StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(
            x.loc[inside, columns], y[inside]
        )
    )
    sub_spec = ModelSpec(
        kind="lgb",
        n_rounds=estimate_rounds("lgb", x.loc[inside, columns], y[inside], sub_folds[:3]),
    )
    specialist = np.nanmean(
        [
            specialist_oof(sub_spec, x, y, mask, seed=config.SEED + offset)
            for offset in range(args.seeds)
        ],
        axis=0,
    )
    print(
        "specialist alone inside mobile: "
        + " ".join(f"{k} {v:.4f}" for k, v in validate.score(y[inside], specialist[inside]).items())
    )

    for weight in (0.25, 0.5, 0.75, 1.0):
        blended = reorder(global_scores, specialist, weight)
        report = _report(f"reorder w={weight}", y, blended, inside)
        validate.log_result(f"{args.name}_reorder{weight}", report, {"weight": weight})

    stacked = stack_oof(stack_features(global_scores, specialist, x), y)
    report = _report("stack", y, stacked, inside)
    validate.log_result(f"{args.name}_stack", report, {})


if __name__ == "__main__":
    main()
