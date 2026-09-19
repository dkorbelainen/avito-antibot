"""Leave-one-block-out ablation: what each feature family is actually worth."""

from __future__ import annotations

import argparse

from sklearn.model_selection import StratifiedKFold

from . import config, validate
from .model import ModelSpec, estimate_rounds
from .pipeline import build_dataset

BLOCKS: dict[str, str] = {
    "popularity": r"^pop_",
    "crowd_relative": r"^crowd_",
    "volume_mix": r"^(n_events|cnt_|rate_|ratio_|n_active_hours|events_per_active_hour|hour_entropy|night_rate)",
    "transitions": r"^(bg_|n_bigrams_uniq|bigram_entropy|self_loop_rate)",
    "timing": r"^(dt_|span_h|events_per_h)",
    "dt_granularity": r"^dtg_",
    "sessions": r"^(sess_|n_sessions)",
    "content": r"^(item_|search_query_|seller_|locations_per_category|items_per_category)",
    "pagination": r"^(page_|queries_nuniq|pages_per_query|query_repeat_rate|query_len_mean)",
    "pointer": r"^(ptr_|pointer_coverage)",
    "pointer_deep": r"^ptrd_",
    "client": r"^(plat_|n_platforms|platform_entropy|n_user_agents|ua_|uafam_|dominant_platform)",
    "catalog_shares": r"^(shloc_|shcat_|shq_)",
    "id_structure": r"^idst_",
    "dwell": r"^dwell_",
    "navigation": r"^(nav_|funnel_|swipe_run_max|item_revisit_max)",
    "platform_scoped": r"^(mob_|web_|webp_)",
    "timeseries": r"^(b1m_|b5m_|b30m_|cmp_|tod_|gap_)",
    "peer_rank": r"^rel_plat_",
    "window": r"^(first_event_offset_h|last_event_offset_h|window_coverage)",
    "cookie_meta": r"^(cookie_|window_dow)",
    "sequence_svd": r"^(seqo_|seqp_)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--kind", default="lgb")
    args = parser.parse_args()

    dataset = build_dataset()
    x, y = dataset.x_train, dataset.y_train
    folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(x, y))

    rounds = estimate_rounds(args.kind, x, y, folds[:3])
    full = validate.repeated_cv(ModelSpec(args.kind, rounds), x, y, n_seeds=args.seeds)
    print(validate.format_report("full", full))
    validate.log_result("ablation_full", full, {"kind": args.kind, "rounds": rounds})

    covered = set().union(*(set(x.filter(regex=pattern).columns) for pattern in BLOCKS.values()))
    if missing := sorted(set(x.columns) - covered):
        print(f"warning: columns in no block: {missing}")

    for name, pattern in BLOCKS.items():
        kept = x.drop(columns=x.filter(regex=pattern).columns)
        report = validate.repeated_cv(
            ModelSpec(args.kind, estimate_rounds(args.kind, kept, y, folds[:3])), kept, y, n_seeds=args.seeds
        )
        delta = report["p_at_r70_mean"] - full["p_at_r70_mean"]
        delta_pr = report["pr_auc_mean"] - full["pr_auc_mean"]
        print(
            f"-{name:<14} n={x.shape[1] - kept.shape[1]:<3} "
            f"P@R70 {report['p_at_r70_mean']:.4f} ({delta:+.4f})  "
            f"PR {report['pr_auc_mean']:.4f} ({delta_pr:+.4f})"
        )
        validate.log_result(f"ablation_drop_{name}", report, {"kind": args.kind, "dropped": int(x.shape[1] - kept.shape[1])})


if __name__ == "__main__":
    main()
