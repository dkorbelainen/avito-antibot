"""Hyper-parameter search on PR-AUC — P@R70 is too noisy to optimise directly.

The budget is set by how long one trial costs, not by how many trials look thorough.
A trial is three folds on a capped round count with a low-learning-rate floor: an
earlier version allowed rates down to 0.015 with a 2500-round early-stopping cap and
spent half an hour per XGBoost trial, which buys nothing the blend does not.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .model import ModelSpec, estimate_rounds, fit_predict
from .pipeline import build_dataset

SPACES = {
    "lgb": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "num_leaves": t.suggest_int("num_leaves", 15, 96),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 10, 120),
        "feature_fraction": t.suggest_float("feature_fraction", 0.3, 0.95),
        "bagging_fraction": t.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": t.suggest_float("lambda_l2", 1e-2, 50.0, log=True),
        "min_gain_to_split": t.suggest_float("min_gain_to_split", 0.0, 1.0),
    },
    "cat": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "depth": t.suggest_int("depth", 4, 8),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.5, 5.0),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 2.0),
    },
    "xgb": lambda t: {
        "eta": t.suggest_float("eta", 0.03, 0.12, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 9),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.95),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="lgb", choices=list(SPACES))
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=900)
    args = parser.parse_args()

    dataset = build_dataset()
    x, y = dataset.x_train, dataset.y_train
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=config.SEED).split(x, y))

    def objective(trial: optuna.Trial) -> float:
        params = SPACES[args.kind](trial)
        rounds = estimate_rounds(
            args.kind,
            x,
            y,
            folds[:1],
            max_rounds=args.max_rounds,
            patience=60,
            params_override=params,
        )
        spec = ModelSpec(kind=args.kind, n_rounds=rounds, params=params)
        oof = np.zeros(len(x))
        for train_idx, valid_idx in folds:
            oof[valid_idx] = fit_predict(
                spec, x.iloc[train_idx], y.iloc[train_idx], x.iloc[valid_idx]
            )
        trial.set_user_attr("rounds", rounds)
        return float(average_precision_score(y, oof))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = (trial.datetime_complete - trial.datetime_start).total_seconds()
        print(
            f"trial {trial.number:3d} pr_auc {trial.value:.4f} best {study.best_value:.4f} "
            f"rounds {trial.user_attrs['rounds']:4d} {elapsed:.0f}s",
            flush=True,
        )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=False, callbacks=[log_trial])

    best = {"params": study.best_params, "rounds": study.best_trial.user_attrs["rounds"], "pr_auc": study.best_value}
    path = config.ROOT / f"params_{args.kind}.json"
    path.write_text(json.dumps(best, indent=2))
    print(f"{args.kind}: PR-AUC {study.best_value:.4f} rounds {best['rounds']}")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
