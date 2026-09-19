"""Figures for the solution notebook.

One palette, one style, defined once. Categorical hues are assigned in fixed order
and never cycled; magnitude charts use a single hue.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e3e0"
SURFACE = "#fcfcfb"


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9,
            "figure.dpi": 120,
        }
    )


def _bar_labels(ax, bars, values, fmt="{:.3f}", pad=0.004) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
        )


def window_timeline(train: pd.DataFrame, test: pd.DataFrame):
    """Train and test windows are disjoint in time — the reason validation is temporal."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for frame, colour, label in ((train, BLUE, "train"), (test, ORANGE, "test")):
        counts = frame["window_start_ts"].dt.date.value_counts().sort_index()
        ax.bar(counts.index, counts.to_numpy(), color=colour, label=label, width=0.72)
    ax.set_title("Окна наблюдения: train и test не пересекаются во времени")
    ax.set_ylabel("cookie")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def metric_noise(per_fold: list[float], per_day: list[float]):
    """P@R70 spread across resampling — why no single number decides anything."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    series = [("случайные фолды", per_fold, BLUE), ("дни (forward-chain)", per_day, ORANGE)]
    for row, (label, values, colour) in enumerate(series):
        ax.scatter(values, np.full(len(values), row), s=54, color=colour, zorder=3, label=label)
        ax.hlines(row, min(values), max(values), color=colour, linewidth=2, alpha=0.35, zorder=2)
        ax.text(
            max(values) + 0.012,
            row,
            f"размах {max(values) - min(values):.2f}",
            va="center",
            color=INK_SOFT,
            fontsize=8.5,
        )
    ax.set_yticks(range(len(series)), [s[0] for s in series])
    ax.set_xlabel("P@R70")
    ax.set_title("Разброс целевой метрики при одной и той же модели")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(min(min(per_fold), min(per_day)) - 0.03, max(max(per_fold), max(per_day)) + 0.12)
    fig.tight_layout()
    return fig


def ablation(deltas: pd.Series, full_score: float):
    """Cost of removing each feature family — magnitude, so a single hue."""
    ordered = deltas.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(ordered) + 1.4))
    bars = ax.barh(ordered.index, ordered.to_numpy(), color=BLUE, height=0.62)
    _bar_labels(ax, bars, ordered.to_numpy(), fmt="{:+.3f}", pad=-0.001)
    ax.set_xlabel(f"изменение P@R70 при удалении блока (полная модель {full_score:.3f})")
    ax.set_title("Вклад блоков признаков")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def regimes(frame: pd.DataFrame):
    """Where the model is strong and where it is not."""
    ordered = frame.sort_values("p_at_r70")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(ordered) + 1.3))
    bars = ax.barh(ordered["regime"], ordered["p_at_r70"], color=BLUE, height=0.6)
    _bar_labels(ax, bars, ordered["p_at_r70"].to_numpy())
    for bar, n, bots in zip(bars, ordered["n"], ordered["bots"], strict=True):
        ax.text(0.006, bar.get_y() + bar.get_height() / 2, f"n={n}, ботов {bots}",
                va="center", ha="left", color=SURFACE, fontsize=8)
    ax.set_xlabel("P@R70 внутри подгруппы")
    ax.set_title("Качество по режимам: метрику решают mobile и короткие сессии")
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def progression(frame: pd.DataFrame):
    """Experiment history: the number that actually moved."""
    fig, ax = plt.subplots(figsize=(8, 3.1))
    x = np.arange(len(frame))
    ax.plot(x, frame["P@R70"], color=BLUE, linewidth=2, marker="o", markersize=7, label="P@R70 (CV)")
    ax.fill_between(
        x,
        frame["P@R70"] - frame["+/-"],
        frame["P@R70"] + frame["+/-"],
        color=BLUE,
        alpha=0.13,
        linewidth=0,
    )
    if frame["fc P@R70"].notna().any():
        ax.plot(x, frame["fc P@R70"], color=ORANGE, linewidth=2, marker="s",
                markersize=6, label="P@R70 (forward-chain)")
    for xi, value in zip(x, frame["P@R70"], strict=True):
        ax.annotate(f"{value:.3f}", (xi, value), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, frame["experiment"], rotation=30, ha="right")
    ax.set_ylabel("P@R70")
    ax.set_title("История экспериментов")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_curve(y_true, scores, target_recall: float = 0.70):
    """Precision-recall with the admissible region the metric actually scores."""
    from .validate import METRIC

    precision, recall = METRIC.pr_curve(y_true, scores)
    best = float(precision[recall >= target_recall].max())
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axvspan(target_recall, 1.0, color=AQUA, alpha=0.1, linewidth=0)
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    ax.axhline(best, color=ORANGE, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"P@R70 = {best:.3f}",
        (target_recall, best),
        textcoords="offset points",
        xytext=(10, 10),
        color=INK,
        fontsize=9.5,
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("PR-кривая OOF; зелёная зона — допустимые пороги (recall ≥ 0.70)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def importances(gains: pd.Series, top: int = 20):
    ordered = gains.head(top).iloc[::-1]
    share = ordered / gains.sum()
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1.2))
    bars = ax.barh(ordered.index, share.to_numpy(), color=BLUE, height=0.62)
    _bar_labels(ax, bars, share.to_numpy() * 100, fmt="{:.1f}%", pad=share.max() * 0.02)
    ax.set_xlabel("доля суммарного gain")
    ax.set_title(f"Топ-{top} признаков по вкладу")
    ax.set_xlim(0, share.max() * 1.18)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
