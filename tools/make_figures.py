"""Render the figures the README embeds.

Reads the shipped out-of-fold scores from `artifacts/`, so it does not retrain
anything; run `python -m src.submit --name final_pct` first if the file is missing.
"""

from __future__ import annotations

import numpy as np

from src import config, plots, report
from src.pipeline import build_dataset

ASSETS = config.ROOT / "assets"
OOF_PATH = config.CACHE_DIR / "oof_final_pct.npy"

TRACK = {
    "B0_baseline_simple": "baseline, 47",
    "ablation_full": "ядро, 178",
    "A3_pointer_deep_lgb": "+ курсор, 353",
    "A7_core_nofit_leak": "без утечки в подгонке, 423",
    "A12_core": "+ временной ряд, 457",
    "A15_ua_clients": "+ User-Agent, 468",
    "P7_pop_pct": "+ популярность, 479",
}


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    plots.use_style()

    dataset = build_dataset()
    oof = np.load(OOF_PATH)
    plots.pr_curve(dataset.y_train, oof).savefig(ASSETS / "pr_curve.png", dpi=150)

    timeline = report.table(list(TRACK)).assign(шаг=lambda f: f.experiment.map(TRACK))
    plots.progression(timeline.dropna(subset=["P@R70"])).savefig(
        ASSETS / "progression.png", dpi=150
    )
    print(f"wrote {ASSETS / 'pr_curve.png'} and {ASSETS / 'progression.png'}")


if __name__ == "__main__":
    main()
