"""Keep a copy of a submission that might turn out to be the best one.

Seven uploads are allowed in total, so every candidate that was ever considered
shippable is archived under `docs/submissions/<name>/` together with the numbers it was
accepted on. The directory is outside the tracked tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

from src import config

SNAPSHOT_DIR = config.ROOT / "docs" / "submissions"
INDEX_PATH = SNAPSHOT_DIR / "index.jsonl"

METRIC_KEYS = (
    "p_at_r70_mean",
    "p_at_r70_std",
    "roc_auc_mean",
    "pr_auc_mean",
    "recall_at_fpr1_mean",
    "blend_p_at_r70",
    "fc_p_at_r70_mean",
    "n_features",
    "kind",
    "rounds",
)


def latest_result(name: str) -> dict[str, object]:
    """The most recent `results.jsonl` row with this experiment name."""
    rows = [json.loads(line) for line in config.RESULTS_PATH.read_text().splitlines()]
    matching = [row for row in rows if row["name"] == name]
    if not matching:
        raise SystemExit(f"no result named {name!r} in results.jsonl")
    return matching[-1]


def head_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=config.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="snapshot directory name")
    parser.add_argument("--result", default="final", help="results.jsonl row to attach")
    parser.add_argument("--note", default="", help="why this candidate was kept")
    parser.add_argument("--submitted", action="store_true", help="mark as actually uploaded")
    args = parser.parse_args()

    target = SNAPSHOT_DIR / args.name
    if target.exists():
        raise SystemExit(f"{target} already exists; pick another name")
    target.mkdir(parents=True)

    shutil.copy2(config.SUBMISSION_PATH, target / "submission.csv")
    oof_path = config.CACHE_DIR / f"oof_{args.result}.npy"
    if oof_path.exists():
        shutil.copy2(oof_path, target / "oof.npy")

    result = latest_result(args.result)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": args.name,
        "md5": hashlib.md5((target / "submission.csv").read_bytes()).hexdigest(),
        "commit": head_commit(),
        "result": args.result,
        "submitted": args.submitted,
        "note": args.note,
        **{k: result[k] for k in METRIC_KEYS if k in result},
    }
    (target / "meta.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    with INDEX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
