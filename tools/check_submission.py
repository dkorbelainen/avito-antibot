"""Format checks on submission.csv, the ones the scoring side will make."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main() -> int:
    submission = pd.read_csv(config.SUBMISSION_PATH)
    test = pd.read_csv(config.TEST_PATH)
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"{'ok  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    check(list(submission.columns) == list(sample.columns), f"columns {list(sample.columns)}")
    check(len(submission) == len(test), f"{len(test)} rows")
    check(not submission["cookie_id"].duplicated().any(), "no duplicate cookie_id")
    check(set(submission["cookie_id"]) == set(test["cookie_id"]), "cookie_id set matches test.csv")
    check(submission["score"].notna().all(), "no missing score")
    check(submission["score"].between(0, 1).all(), "every score inside [0, 1]")
    check(submission["score"].nunique() > len(submission) * 0.9, "scores are not degenerate")

    digest = hashlib.md5(config.SUBMISSION_PATH.read_bytes()).hexdigest()
    print(f"\nrows {len(submission)}  distinct scores {submission['score'].nunique()}  md5 {digest}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
