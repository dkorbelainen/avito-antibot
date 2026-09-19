"""Render results.jsonl as the experiment table used in the notebook."""

from __future__ import annotations

import json

import pandas as pd

from . import config

COLUMNS = {
    "name": "experiment",
    "n_features": "features",
    "p_at_r70_mean": "P@R70",
    "p_at_r70_std": "+/-",
    "roc_auc_mean": "ROC-AUC",
    "pr_auc_mean": "PR-AUC",
    "recall_at_fpr1_mean": "R@FPR1%",
    "fc_p_at_r70_mean": "fc P@R70",
}


def load(path=None) -> pd.DataFrame:
    path = path or config.RESULTS_PATH
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    missing = [c for c in COLUMNS if c not in frame.columns]
    for column in missing:
        frame[column] = pd.NA
    return frame[list(COLUMNS)].rename(columns=COLUMNS)


def table(names: list[str] | None = None) -> pd.DataFrame:
    frame = load()
    if names:
        frame = frame[frame["experiment"].isin(names)]
        frame = frame.set_index("experiment").loc[[n for n in names if n in set(frame["experiment"])]].reset_index()
    return frame.round(4)


if __name__ == "__main__":
    print(table().to_string(index=False))
