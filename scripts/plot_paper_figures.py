from __future__ import annotations

"""Generate the publication figure set for the hard-budget alert study.

All quantitative panels are derived from frozen result files.  Figure 7 replays
the frozen UAD-HBAC controller for one real station-month with trace_all_steps.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


# Nature-figure contract: editable text and a consistent sans-serif stack.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


COLORS = {
    "ink": "#352A26",
    "muted": "#786A64",
    "grid": "#E9E3DF",
    "paper": "#FFFDFC",
    "cream": "#FFF8F2",
    "white": "#FFFFFF",
    "duo1_light": "#FAECA8",
    "duo1_dark": "#F18C25",
    "duo2_light": "#FDD5C0",
    "duo2_dark": "#DB614F",
    "duo3_light": "#F6C0CC",
    "duo3_dark": "#E64825",
    "duo4_light": "#CAC0E1",
    "duo4_dark": "#715EA9",
    "duo5_light": "#ABDAEC",
    "duo5_dark": "#6A9ACE",
    "duo6_light": "#97D1A0",
    "duo6_dark": "#1E803D",
    "g4_blue": "#91ABD2",
    "g4_pink": "#F59694",
    "g4_purple": "#BCA6CD",
    "g4_orange": "#FBD3A3",
    "g5_purple": "#BD7ED9",
    "g5_blue": "#90BFF9",
    "g5_green": "#5DD959",
    "g5_orange": "#FFC387",
    "g5_olive": "#C5C515",
    # Backward-compatible semantic aliases used by labels and neutral accents.
    "sand": "#FAECA8",
    "peach": "#FDD5C0",
    "amber": "#F18C25",
    "terracotta": "#DB614F",
    "brick": "#E64825",
    "rose": "#F6C0CC",
    "plum": "#715EA9",
    "olive": "#1E803D",
    "sage": "#97D1A0",
}

DUO_PALETTES = [
    ("#FAECA8", "#F18C25"),
    ("#FDD5C0", "#DB614F"),
    ("#F6C0CC", "#E64825"),
    ("#CAC0E1", "#715EA9"),
    ("#ABDAEC", "#6A9ACE"),
    ("#97D1A0", "#1E803D"),
]
FOUR_GROUP = ["#91ABD2", "#F59694", "#BCA6CD", "#FBD3A3"]
FIVE_GROUP = ["#BD7ED9", "#90BFF9", "#5DD959", "#FFC387", "#C5C515"]


EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "dynamic_hard_budget" / "experiments"
TRAJECTORY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dynamic_hard_budget"
    / "local_weather_hazard_trajectory"
)
SPATIAL_ROOT = PROJECT_ROOT / "results" / "recurrence_modeling" / "spatial_generalization"
EVENT_TABLE = (
    PROJECT_ROOT
    / "results"
    / "recurrence_modeling"
    / "seasonal_recurrence"
    / "event_global_seasonal_mapping.csv"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "rig_hazard_dynamic_hard_budget.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "visualization" / "manuscript_figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.tick_params(axis="both", colors=COLORS["muted"], labelsize=7.2, length=3, width=0.7)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.title.set_color(COLORS["ink"])
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.75, zorder=0)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="bottom",
    )


def add_badge(ax: plt.Axes, text: str, x: float, y: float, color: str = "olive") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.12",
            facecolor=COLORS[color],
            edgecolor="none",
            alpha=0.22,
        ),
    )


def export_figure(fig: plt.Figure, output_base: Path, dpi: int = 600) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for extension in ("svg", "pdf", "png", "tiff"):
        target = output_base.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = dpi if extension == "tiff" else 300
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(target, **kwargs)
        outputs[extension] = str(target.relative_to(PROJECT_ROOT)).replace("\\", "/")
    plt.close(fig)
    return outputs


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "ink",
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
    mutation_scale: float = 11,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=1.15,
            color=COLORS[color],
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            shrinkA=3,
            shrinkB=3,
        )
    )


def rounded_node(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str = "brick",
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor=COLORS[facecolor],
            edgecolor=COLORS[edgecolor],
            linewidth=1.15,
        )
    )
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["ink"])
    ax.text(x + width / 2, y + height * 0.29, subtitle, ha="center", va="center", fontsize=6.2, color=COLORS["muted"], linespacing=1.15)


def plot_figure2(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Delayed-feedback state machine and causal accounting timeline."""

    fig = plt.figure(figsize=(7.25, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.28], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    for ax in (ax_a, ax_b):
        ax.set_facecolor(COLORS["paper"])
        ax.set_axis_off()

    add_panel_label(ax_a, "a", x=-0.03, y=1.01)
    add_panel_label(ax_b, "b", x=-0.02, y=1.01)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.text(0.02, 0.96, "Delayed-feedback capacity state machine", fontsize=9, fontweight="bold", color=COLORS["ink"], va="top")

    rounded_node(ax_a, (0.34, 0.61), 0.34, 0.15, "P", "pending / unsettled\nreserved immediately", "g5_blue", "duo5_dark")
    rounded_node(ax_a, (0.06, 0.29), 0.30, 0.14, "F", "confirmed\nfalse-alarm capacity", "g5_orange", "duo1_dark")
    rounded_node(ax_a, (0.63, 0.29), 0.30, 0.14, "L", "true-alert capacity\nlocked by policy", "g5_purple", "duo4_dark")

    ax_a.text(0.11, 0.83, "alert issued", fontsize=7.2, color=COLORS["ink"], fontweight="bold")
    arrow(ax_a, (0.20, 0.80), (0.37, 0.72), "duo2_dark")
    arrow(ax_a, (0.39, 0.60), (0.28, 0.43), "duo1_dark")
    ax_a.text(0.13, 0.49, "6 h follow-up complete\nno matched event", fontsize=6.1, color=COLORS["muted"], ha="center")
    arrow(ax_a, (0.64, 0.60), (0.72, 0.43), "duo4_dark")
    ax_a.text(0.79, 0.50, "event observed\nlock reserve", fontsize=6.1, color=COLORS["muted"], ha="center")

    arrow(ax_a, (0.66, 0.69), (0.91, 0.68), "duo6_dark")
    ax_a.text(0.81, 0.77, "event observed\nrelease reserve", fontsize=6.1, color=COLORS["muted"], ha="center")
    ax_a.add_patch(Circle((0.93, 0.68), 0.035, facecolor=COLORS["g5_green"], edgecolor=COLORS["duo6_dark"], linewidth=1.0))
    ax_a.text(0.93, 0.68, "R", ha="center", va="center", fontsize=7, fontweight="bold", color=COLORS["ink"])

    arrow(ax_a, (0.52, 0.77), (0.68, 0.83), "g5_olive", connectionstyle="arc3,rad=-0.55")
    arrow(ax_a, (0.68, 0.83), (0.58, 0.77), "g5_olive", connectionstyle="arc3,rad=-0.55")
    ax_a.text(0.72, 0.88, "right-censored\nremains charged", fontsize=6.1, color=COLORS["muted"], ha="center")

    ax_a.add_patch(FancyBboxPatch((0.08, 0.08), 0.84, 0.11, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=COLORS["duo1_light"], edgecolor=COLORS["duo1_dark"], linewidth=1.0))
    ax_a.text(0.50, 0.145, r"$O_{qk}(t)=F_{qk}(t)+P_{qk}(t)+L_{qk}(t)\leq C_k$", fontsize=9.2, color=COLORS["duo3_dark"], fontweight="bold", ha="center", va="center")
    ax_a.text(0.50, 0.035, "Every bin stays on the alert's issue-month ledger", fontsize=6.6, color=COLORS["muted"], ha="center")

    ax_b.set_xlim(-24.8, 7.0)
    ax_b.set_ylim(0, 1)
    ax_b.text(-24.3, 0.96, "Event matching and issue-month accounting", fontsize=9, fontweight="bold", color=COLORS["ink"], va="top")
    y = 0.48
    ax_b.plot([-24, 6], [y, y], color=COLORS["ink"], lw=1.2)
    for tick in np.arange(-24, 6.01, 1 / 6):
        height = 0.024 if abs(tick * 6 - round(tick * 6)) < 1e-6 else 0.014
        ax_b.plot([tick, tick], [y - height, y + height], color=COLORS["muted"], lw=0.38)

    ax_b.add_patch(Rectangle((-24, 0.54), 24, 0.12, facecolor=COLORS["duo5_light"], edgecolor="none", alpha=0.95))
    ax_b.text(-12, 0.60, "past 24 h input", ha="center", va="center", fontsize=7, color=COLORS["ink"], fontweight="bold")
    ax_b.add_patch(Rectangle((0, 0.54), 6, 0.12, facecolor=COLORS["duo2_light"], edgecolor="none", alpha=0.95))
    ax_b.text(3, 0.60, "future 6 h · 36 forecasts", ha="center", va="center", fontsize=7, color=COLORS["ink"], fontweight="bold")

    alert_t = 0.0
    month_t = 1 + 4 / 6
    anchor_t = 2.0
    onset_t = 2 + 8 / 60
    settle_t = 6.0
    ax_b.scatter([alert_t], [y], s=40, color=COLORS["duo3_dark"], edgecolor="white", linewidth=0.7, zorder=5)
    ax_b.text(alert_t, 0.37, "alert issued\n31 Dec 22:20", fontsize=6.3, ha="center", color=COLORS["ink"])
    ax_b.axvline(month_t, ymin=0.21, ymax=0.81, color=COLORS["duo1_dark"], lw=1.0, ls=(0, (3, 2)))
    ax_b.text(month_t - 0.28, 0.77, "month boundary\n1 Jan 00:00", fontsize=6.2, ha="right", color=COLORS["duo1_dark"], fontweight="bold")
    ax_b.scatter([anchor_t], [y], marker="s", s=30, facecolor=COLORS["paper"], edgecolor=COLORS["duo2_dark"], linewidth=1.1, zorder=5)
    ax_b.text(anchor_t - 0.12, 0.22, "evaluation anchor\nfloor(onset − 1 ns)\n00:20", fontsize=6.1, ha="right", color=COLORS["muted"])
    ax_b.scatter([onset_t], [y], marker="D", s=46, facecolor=COLORS["duo3_light"], edgecolor=COLORS["duo3_dark"], linewidth=0.9, zorder=6)
    ax_b.text(onset_t + 0.55, 0.70, "observed onset\n00:28 (off-grid)", fontsize=6.2, ha="left", color=COLORS["duo3_dark"], fontweight="bold")
    ax_b.scatter([settle_t], [y], marker="o", s=38, color=COLORS["duo6_dark"], edgecolor="white", linewidth=0.7, zorder=5)
    ax_b.text(settle_t, 0.37, "settlement\n04:20", fontsize=6.3, ha="center", color=COLORS["ink"])

    ax_b.annotate("one alert ↔ at most one event", xy=(onset_t, 0.50), xytext=(4.3, 0.86), fontsize=6.5, color=COLORS["duo4_dark"], ha="center", arrowprops=dict(arrowstyle="-|>", color=COLORS["duo4_dark"], lw=0.8))
    ax_b.annotate("capacity remains in December", xy=(month_t, 0.18), xytext=(-2.0, 0.10), fontsize=6.5, color=COLORS["duo3_dark"], ha="center", arrowprops=dict(arrowstyle="-|>", color=COLORS["duo3_dark"], lw=0.8))
    ax_b.text(-24, 0.43, "−24 h", fontsize=6.5, ha="center", color=COLORS["muted"])
    ax_b.text(0, 0.43, "$t$", fontsize=7, ha="center", color=COLORS["ink"], fontweight="bold")
    ax_b.text(6, 0.43, "+6 h", fontsize=6.5, ha="center", color=COLORS["muted"])

    fig.subplots_adjust(left=0.025, right=0.99, top=0.97, bottom=0.04)
    outputs = export_figure(fig, output / "fig2_delayed_feedback_accounting")
    contract = {
        "core_conclusion": "Immediate reservation plus causal settlement keeps delayed-feedback occupancy within capacity and on the alert's issue-month ledger.",
        "archetype": "schematic-led composite",
        "panels": {"a": "capacity state machine", "b": "event matching and issue-month timeline"},
        "source_data": "conceptual; no synthetic quantitative observations",
    }
    return outputs, contract


def load_figure4_data(source_dir: Path) -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "frozen_main_results.csv"
    data = pd.read_csv(path)
    data = data.loc[data["method"].eq("uadhbac")].copy()
    data = data.sort_values(["year", "budget_hours"]).reset_index(drop=True)
    assert data.shape[0] == 8
    assert set(data["budget_hours"].astype(float)) == {2.0, 5.0, 10.0, 20.0}
    assert data["online_hard_budget_violations"].fillna(0).eq(0).all()
    assert data["nesting_violations"].fillna(0).eq(0).all()
    assert (data["maximum_reserved_alarm_hours"] <= data["budget_hours"] + 1e-9).all()
    keep = [
        "year",
        "budget_hours",
        "operational_hit_events",
        "operational_evaluable_events",
        "operational_event_hit_rate",
        "operational_lead_utility_hours",
        "maximum_false_alarm_hours",
        "maximum_reserved_alarm_hours",
        "online_hard_budget_violations",
        "nesting_violations",
    ]
    out = data[keep].copy()
    out.to_csv(source_dir / "fig4_budget_performance.csv", index=False)
    return out


def plot_figure4(output: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = load_figure4_data(source_dir)
    budgets = np.array([2.0, 5.0, 10.0, 20.0])
    x = np.arange(budgets.size)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), sharex=True)
    budget_colors = FOUR_GROUP

    for column, year in enumerate((2023, 2024)):
        part = data.loc[data["year"].eq(year)].sort_values("budget_hours")
        rates = 100 * part["operational_event_hit_rate"].to_numpy(float)
        utility = part["operational_lead_utility_hours"].to_numpy(float)
        hits = part["operational_hit_events"].to_numpy(int)
        totals = part["operational_evaluable_events"].to_numpy(int)

        ax = axes[0, column]
        bars = ax.bar(x, rates, width=0.64, color=budget_colors, edgecolor=COLORS["muted"], linewidth=0.75, zorder=2)
        ax.plot(x, rates, color=COLORS["ink"], lw=1.35, zorder=3)
        ax.scatter(x, rates, s=30, c=budget_colors, edgecolor=COLORS["white"], linewidth=0.55, zorder=4)
        for index, (bar, rate, hit, total) in enumerate(zip(bars, rates, hits, totals)):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.5, f"{rate:.1f}%", ha="center", va="bottom", fontsize=7, color=COLORS["ink"], fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, max(4, rate * 0.52), f"{hit}/{total}", ha="center", va="center", fontsize=7.2, color=COLORS["ink"], fontweight="bold")
        ax.set_ylim(0, 96)
        ax.set_ylabel("Operational event hit rate (%)" if column == 0 else "")
        ax.set_title(f"{year} frozen evaluation", fontsize=8.6, fontweight="bold", pad=7)
        style_axis(ax, grid=True)
        add_panel_label(ax, "a" if column == 0 else "b")
        add_badge(ax, "all B: FAH ≤ B · HBV = 0 · NC = 0", 0.50, 0.975, "duo6_light")

        ax = axes[1, column]
        ax.fill_between(x, utility, color=COLORS["cream"], alpha=0.88, zorder=1)
        ax.plot(x, utility, color=COLORS["ink"], lw=1.55, zorder=3)
        ax.scatter(x, utility, s=36, c=budget_colors, edgecolor=COLORS["white"], linewidth=0.55, zorder=4)
        for xi, value in zip(x, utility):
            ax.text(xi, value + 0.14, f"{value:.2f} h", ha="center", va="bottom", fontsize=7, color=COLORS["ink"], fontweight="bold")
        ax.set_ylim(0, 4.25)
        ax.set_ylabel("Lead-time utility (h)" if column == 0 else "")
        ax.set_xlabel(r"Monthly budget  $B$  (h · station$^{-1}$ · month$^{-1}$)")
        ax.set_xticks(x, ["2", "5", "10", "20"])
        style_axis(ax, grid=True)
        add_panel_label(ax, "c" if column == 0 else "d")

    fig.text(0.5, 0.985, "More budget increases event coverage while preserving every hard constraint", ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.952, "UAD-HBAC · operational-realistic queue · parameters selected on 2022 pooled OOF", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.12, wspace=0.23, hspace=0.28)
    outputs = export_figure(fig, output / "fig4_budget_performance")
    contract = {
        "core_conclusion": "Increasing the monthly alert budget raises real-world event coverage and lead-time utility without hard-budget or nesting violations.",
        "archetype": "quantitative grid",
        "panels": {"a": "2023 hit rate", "b": "2024 hit rate", "c": "2023 utility", "d": "2024 utility"},
        "statistics": "frozen point estimates; operational-realistic event queue",
        "source_data": "source_data/fig4_budget_performance.csv",
    }
    return outputs, contract


def load_figure5_data(source_dir: Path) -> pd.DataFrame:
    original = pd.read_csv(EXPERIMENT_ROOT / "paired_station_cluster_bootstrap.csv")
    budget_safe = original.loc[
        original["method_a"].eq("uadhbac")
        & original["method_b"].eq("budget_safe")
        & original["queue"].eq("operational")
        & original["metric"].eq("lead_utility")
        & original["budget_hours"].isin([5.0, 10.0])
    ].copy()
    standard = pd.read_csv(EXPERIMENT_ROOT / "standard_baseline_station_cluster_bootstrap.csv")
    standard = standard.loc[
        standard["method_a"].eq("uadhbac")
        & standard["queue"].eq("operational")
        & standard["metric"].eq("lead_utility")
        & standard["budget_hours"].isin([5.0, 10.0])
    ].copy()
    data = pd.concat([budget_safe, standard], ignore_index=True)
    labels = {
        "budget_safe": "Budget-safe",
        "dual_mirror_descent_hard_guard": "Dual mirror descent",
        "switch_over_knapsack_hard_guard": "Switch-over knapsack",
    }
    data["baseline"] = data["method_b"].map(labels)
    data = data.sort_values(["baseline", "year", "budget_hours"]).reset_index(drop=True)
    assert data.shape[0] == 12
    assert data["difference"].gt(0).all()
    keep = [
        "baseline",
        "year",
        "budget_hours",
        "difference",
        "ci_lower",
        "ci_upper",
        "p_value",
        "holm_p_value",
        "bootstrap_samples",
        "stations",
        "paired_events",
        "queue",
        "metric",
    ]
    data[keep].to_csv(source_dir / "fig5_utility_gain_forest.csv", index=False)
    return data[keep]


def p_label(value: float) -> str:
    if value < 0.001:
        return r"$P_{Holm}<0.001$"
    return rf"$P_{{Holm}}={value:.3f}$"


def plot_figure5(output: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = load_figure5_data(source_dir)
    baseline_order = ["Budget-safe", "Dual mirror descent", "Switch-over knapsack"]
    baseline_colors = {
        "Budget-safe": COLORS["duo1_dark"],
        "Dual mirror descent": COLORS["duo4_dark"],
        "Switch-over knapsack": COLORS["duo5_dark"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.15), sharex=True, sharey=True)
    row_order = [(2023, 5.0), (2023, 10.0), (2024, 5.0), (2024, 10.0)]
    y = np.arange(4)[::-1]

    for panel, (ax, baseline) in enumerate(zip(axes, baseline_order)):
        part = data.loc[data["baseline"].eq(baseline)].copy()
        part["row_key"] = list(zip(part["year"].astype(int), part["budget_hours"].astype(float)))
        part = part.set_index("row_key").loc[row_order].reset_index()
        estimates = part["difference"].to_numpy(float)
        low = part["ci_lower"].to_numpy(float)
        high = part["ci_upper"].to_numpy(float)
        pvals = part["holm_p_value"].to_numpy(float)
        color = baseline_colors[baseline]

        for yi in y:
            ax.axhspan(yi - 0.46, yi + 0.46, color=COLORS["cream"] if yi % 2 else COLORS["paper"], zorder=0)
        for yi, estimate, lo, hi, pval in zip(y, estimates, low, high, pvals):
            significant = pval < 0.05
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.65, solid_capstyle="round", zorder=3)
            ax.plot([lo, lo], [yi - 0.08, yi + 0.08], color=color, lw=1.0, zorder=3)
            ax.plot([hi, hi], [yi - 0.08, yi + 0.08], color=color, lw=1.0, zorder=3)
            ax.scatter([estimate], [yi], s=40, marker="o", facecolor=color if significant else COLORS["white"], edgecolor=color, linewidth=1.2, zorder=4)
            ax.text(1.60, yi, p_label(pval), ha="left", va="center", fontsize=6.0, color=COLORS["ink"] if significant else COLORS["muted"])
            ax.text(estimate, yi + 0.20, f"{estimate:+.2f}", ha="center", va="bottom", fontsize=6.3, color=color, fontweight="bold")

        ax.axvline(0, color=COLORS["muted"], lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.14, 2.18)
        ax.set_ylim(-0.65, 3.65)
        ax.set_title(baseline, fontsize=8.2, fontweight="bold", color=color, pad=9)
        ax.set_xlabel(r"$\Delta U$  (h)")
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels(["2023 · 5 h", "2023 · 10 h", "2024 · 5 h", "2024 · 10 h"])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        style_axis(ax, grid=False)
        add_panel_label(ax, chr(ord("a") + panel), x=-0.10 if panel == 0 else -0.05)
        stations = sorted(set(part["stations"].astype(int)))
        ax.text(0.98, 0.02, "5000 station-cluster resamples\n" + " / ".join(f"n={value}" for value in stations) + " stations", transform=ax.transAxes, fontsize=5.8, color=COLORS["muted"], ha="right", va="bottom")

    fig.text(0.5, 0.985, "UAD-HBAC improves lead-time utility over three frozen baselines", ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.948, "Operational-realistic queue · paired station-cluster bootstrap · 95% CI", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    fig.text(0.5, 0.025, "Filled markers: Holm-adjusted P < 0.05   ·   Hollow marker: positive estimate, not significant after multiplicity correction", ha="center", va="bottom", fontsize=6.2, color=COLORS["muted"])
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.17, wspace=0.14)
    outputs = export_figure(fig, output / "fig5_utility_gain_forest")
    contract = {
        "core_conclusion": "UAD-HBAC has positive frozen lead-time utility gains over all three baselines, with seven of eight standard-baseline comparisons remaining significant after Holm correction.",
        "archetype": "clinical-triptych forest plot",
        "panels": {"a": "budget-safe", "b": "dual mirror descent", "c": "switch-over knapsack"},
        "statistics": "5000 paired station-cluster bootstrap samples; 95% CI; Holm-adjusted P values",
        "source_data": "source_data/fig5_utility_gain_forest.csv",
    }
    return outputs, contract


def load_figure6_data(source_dir: Path) -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "nesting_utility_cost.csv"
    data = pd.read_csv(path)
    rows = []
    for year, group in data.groupby("year", sort=True):
        non_nested = int(group["non_nested_adjacent_pair_violations"].max())
        nested = int(group["nested_adjacent_pair_violations"].max())
        rows.extend(
            [
                {"year": int(year), "method": "Without budget coupling", "conflicts": non_nested},
                {"year": int(year), "method": "UAD-HBAC", "conflicts": nested},
            ]
        )
    out = pd.DataFrame(rows)
    assert out.loc[out["method"].eq("Without budget coupling"), "conflicts"].tolist() == [2574, 2952]
    assert out.loc[out["method"].eq("UAD-HBAC"), "conflicts"].eq(0).all()
    out.to_csv(source_dir / "fig6_nesting_conflicts.csv", index=False)
    return out


def plot_figure6(output: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = load_figure6_data(source_dir)
    fig = plt.figure(figsize=(7.25, 4.30))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    add_panel_label(ax_a, "a", x=-0.03)
    ax_a.text(0.02, 0.99, "Nested alarm sets", fontsize=9, fontweight="bold", color=COLORS["ink"], va="top")

    boxes = [
        (0.05, 0.08, 0.88, 0.76, "20 h", FOUR_GROUP[3]),
        (0.14, 0.17, 0.70, 0.58, "10 h", FOUR_GROUP[2]),
        (0.23, 0.26, 0.52, 0.40, "5 h", FOUR_GROUP[1]),
        (0.32, 0.35, 0.34, 0.22, "2 h", FOUR_GROUP[0]),
    ]
    for x, y, width, height, label, color in boxes:
        ax_a.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01,rounding_size=0.035", facecolor=color, edgecolor=COLORS["ink"], linewidth=0.8, alpha=0.94))
        ax_a.text(x + width - 0.035, y + height - 0.035, rf"$A({label})$", ha="right", va="top", fontsize=7.5, color=COLORS["ink"], fontweight="bold")
    ax_a.text(0.49, 0.43, r"$A(2)\subseteq A(5)$" + "\n" + r"$\subseteq A(10)\subseteq A(20)$", ha="center", va="center", fontsize=9.2, color=COLORS["white"], fontweight="bold", linespacing=1.28)
    arrow(ax_a, (0.06, 0.86), (0.90, 0.86), "duo1_dark")
    ax_a.text(0.48, 0.885, "more capacity adds alerts; it never withdraws existing alerts", ha="center", va="bottom", fontsize=6.0, color=COLORS["muted"])
    ax_a.text(0.49, 0.025, "Joint capacity guard + candidate inheritance", ha="center", va="center", fontsize=6.5, color=COLORS["duo3_dark"], fontweight="bold")

    add_panel_label(ax_b, "b", x=-0.10)
    ax_b.set_title("Adjacent-budget conflicts", fontsize=9, fontweight="bold", pad=8)
    years = [2023, 2024]
    x = np.arange(2)
    width = 0.34
    uncoupled = [int(data.loc[(data["year"].eq(year)) & data["method"].eq("Without budget coupling"), "conflicts"].iloc[0]) for year in years]
    coupled = [int(data.loc[(data["year"].eq(year)) & data["method"].eq("UAD-HBAC"), "conflicts"].iloc[0]) for year in years]
    baseline_light, controller_dark = DUO_PALETTES[4]
    bars = ax_b.bar(x - width / 2, uncoupled, width, color=baseline_light, edgecolor=controller_dark, linewidth=1.1, label="Without coupling", zorder=2)
    ax_b.bar(x + width / 2, coupled, width, color=controller_dark, edgecolor=controller_dark, linewidth=1.1, label="UAD-HBAC", zorder=2)
    for bar, value in zip(bars, uncoupled):
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + 90, f"{value:,}", ha="center", va="bottom", fontsize=8, color=controller_dark, fontweight="bold")
    for xi in x + width / 2:
        ax_b.scatter([xi], [0], s=48, facecolor=controller_dark, edgecolor=COLORS["ink"], linewidth=0.8, zorder=4)
        ax_b.text(xi, 115, "0", ha="center", va="bottom", fontsize=8, color=controller_dark, fontweight="bold")
    ax_b.set_xticks(x, ["2023", "2024"])
    ax_b.set_ylabel("Conflicting 10-min decisions")
    ax_b.set_ylim(-150, 3400)
    style_axis(ax_b, grid=True)
    ax_b.legend(loc="upper left", fontsize=6.7)
    ax_b.text(0.33, 2180, "Conflicts removed\n100%", ha="center", va="center", fontsize=6.5, color=controller_dark, fontweight="bold", linespacing=1.25)
    ax_b.text(0.33, 1580, "Utility-cost point estimate\n2.91%", ha="center", va="center", fontsize=6.5, color=COLORS["ink"], fontweight="bold", linespacing=1.25)
    ax_b.text(0.33, 1220, "Holm-significant only\n2023 · 10 h", ha="center", va="center", fontsize=5.9, color=COLORS["muted"], linespacing=1.25)

    fig.text(0.5, 0.985, "Cross-budget coupling removes configuration contradictions", ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["ink"])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.15)
    outputs = export_figure(fig, output / "fig6_cross_budget_nesting")
    contract = {
        "core_conclusion": "Joint budget coupling enforces set inclusion and removes all observed adjacent-budget conflicts at a modest frozen point-estimate utility cost.",
        "archetype": "schematic-led composite",
        "panels": {"a": "nested-set semantics", "b": "conflicts before and after coupling"},
        "statistics": "conflict counts are deterministic frozen replay results; 2.91% is an aggregate point estimate",
        "source_data": "source_data/fig6_nesting_conflicts.csv",
    }
    return outputs, contract


def load_figure8_data(source_dir: Path) -> pd.DataFrame:
    path = SPATIAL_ROOT / "paired_generalization_gap_bootstrap.csv"
    data = pd.read_csv(path)
    data = data.loc[
        data["split"].isin(["2022_spatial_holdout", "2023_spatial_temporal"])
        & data["metric"].eq("delta_pr_auc")
        & data["model_a"].eq("station_group_cv")
        & data["model_b"].eq("all_station_reference")
    ].copy()
    assert data.shape[0] == 2
    labels = {
        "2022_spatial_holdout": "2022 station-group holdout",
        "2023_spatial_temporal": "2023 spatial–temporal evaluation",
    }
    data["experiment"] = data["split"].map(labels)
    keep = ["experiment", "split", "estimate", "ci95_low", "ci95_high", "probability_model_a_better", "stations", "bootstrap_samples"]
    data[keep].to_csv(source_dir / "fig8_unseen_station_generalization.csv", index=False)
    return data[keep]


def plot_figure8(output: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = load_figure8_data(source_dir)
    order = ["2022 station-group holdout", "2023 spatial–temporal evaluation"]
    data = data.set_index("experiment").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.25, 2.80))
    y = np.array([1.0, 0.0])
    colors = [DUO_PALETTES[3][1], DUO_PALETTES[3][0]]
    markers = ["o", "s"]
    for yi, (_, row), color, marker in zip(y, data.iterrows(), colors, markers):
        ax.axhspan(yi - 0.39, yi + 0.39, color=COLORS["cream"] if yi > 0 else COLORS["paper"], zorder=0)
        ax.plot([row.ci95_low, row.ci95_high], [yi, yi], color=color, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([row.ci95_low, row.ci95_low], [yi - 0.07, yi + 0.07], color=color, lw=1.0)
        ax.plot([row.ci95_high, row.ci95_high], [yi - 0.07, yi + 0.07], color=color, lw=1.0)
        ax.scatter([row.estimate], [yi], s=52, color=color, marker=marker, edgecolor=COLORS["ink"] if color == DUO_PALETTES[3][0] else COLORS["white"], linewidth=0.8, zorder=4)
        label_color = DUO_PALETTES[3][1]
        ax.text(row.estimate, yi + 0.21, f"{row.estimate:+.4f}", ha="center", va="bottom", fontsize=7.2, color=label_color, fontweight="bold")
        ax.text(-0.0125, yi - 0.16, f"95% CI [{row.ci95_low:.4f}, {row.ci95_high:.4f}]\nPr(Δ>0)={row.probability_model_a_better:.4f} · n={int(row.stations)}", ha="right", va="top", fontsize=6.2, color=COLORS["muted"])
    ax.axvline(0, color=COLORS["ink"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(-0.14, 0.018)
    ax.set_ylim(-0.65, 1.65)
    ax.set_yticks(y, order)
    ax.set_xlabel(r"Change in PR-AUC relative to the all-station reference  ($\Delta$PR-AUC)")
    style_axis(ax, grid=False)
    add_panel_label(ax, "a", x=-0.04, y=1.05)
    ax.text(0.01, 1.02, "No pooled estimate", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=COLORS["duo4_dark"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor=COLORS["duo4_light"], edgecolor="none", alpha=0.65))
    fig.text(0.5, 0.985, "Performance declines at unseen stations under two distinct contracts", ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.935, "Separate 5000-sample station-cluster bootstrap experiments; effects are not pooled", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.24)
    outputs = export_figure(fig, output / "fig8_unseen_station_generalization")
    contract = {
        "core_conclusion": "Both spatial contracts show a PR-AUC loss at unseen stations, but the two estimates remain separate experiments and are not pooled.",
        "archetype": "supplementary forest plot",
        "panels": {"a": "two distinct spatial generalization gaps"},
        "statistics": "5000 station-cluster bootstrap samples; 95% CI; n=27 stations",
        "source_data": "source_data/fig8_unseen_station_generalization.csv",
    }
    return outputs, contract


def prepare_figure7_data(source_dir: Path, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trace_path = source_dir / "fig7_F5260_2024-03_trace.csv.gz"
    events_path = source_dir / "fig7_F5260_2024-03_events.csv"
    summary_path = source_dir / "fig7_F5260_2024-03_summary.json"
    if trace_path.exists() and events_path.exists() and summary_path.exists() and not force:
        trace = pd.read_csv(trace_path, parse_dates=["issue_time"])
        events = pd.read_csv(events_path, parse_dates=["onset_time"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return trace, events, summary

    from rig_hazard.dynamic_hard_budget import DynamicBudgetConfig, apply_dynamic_hard_budget

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selected_path = EXPERIMENT_ROOT / "selected_controller_2022.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    timeline_manifest = (
        PROJECT_ROOT
        / config["cache_root"]
        / "timeline_manifest.json"
    )
    manifest = json.loads(timeline_manifest.read_text(encoding="utf-8"))
    file_ids = {
        int(row["file_id"])
        for row in manifest["files"]
        if str(row["station_code"]) == "F5260" and int(row.get("year", 2024)) == 2024
    }
    if not file_ids:
        raise RuntimeError("No 2024 frozen trajectory file found for station F5260")

    trajectory_path = TRAJECTORY_ROOT / "2024_final_time.npz"
    with np.load(trajectory_path, allow_pickle=False) as payload:
        file_id = np.asarray(payload["file_id"], dtype=np.int64)
        mask = np.isin(file_id, list(file_ids))
        issue_time = pd.to_datetime(np.asarray(payload["issue_time_ns"])[mask], unit="ns")
        hazard = np.asarray(payload["h_trajectory"])[mask].astype(np.float64, copy=False)
        observed = np.asarray(payload["observed_6h"])[mask].astype(np.int8, copy=False)
        risk = np.asarray(payload["risk_6h"])[mask].astype(np.float64, copy=False)

    frame = pd.DataFrame(
        {
            "station_code": "F5260",
            "issue_time": issue_time,
            "observed_6h": observed,
            "risk_6h": risk,
        }
    )
    order = frame.sort_values("issue_time").index.to_numpy()
    frame = frame.iloc[order].reset_index(drop=True)
    hazard = hazard[order]

    event_table = pd.read_csv(EVENT_TABLE, low_memory=False)
    event_table["station_code"] = event_table["station_code"].astype(str)
    event_table["onset_time"] = pd.to_datetime(event_table["onset_time"], errors="coerce")
    event_table = event_table.loc[
        event_table["station_code"].eq("F5260")
        & event_table["onset_time"].dt.year.eq(2024)
        & pd.to_numeric(event_table["valid_target_event"], errors="coerce").fillna(0).eq(1)
    ].copy()

    controller_cfg = config["controller"]
    controller = DynamicBudgetConfig(
        step_minutes=int(config["step_minutes"]),
        horizon_steps=int(config["horizon_steps"]),
        method="uadhbac",
        utility=str(controller_cfg.get("utility", "linear")),
        score_threshold=float(selected["utility_score_threshold"]),
        price_initial=float(controller_cfg.get("price_initial", 0.0)),
        price_learning_rate=float(selected["price_learning_rate"]),
        pending_pressure=float(selected["pending_pressure"]),
        pacing_slack_bins=int(controller_cfg.get("pacing_slack_bins", 1)),
        deduplication_bins=int(controller_cfg.get("deduplication_bins", 2)),
        couple_budgets=bool(controller_cfg.get("couple_budgets", True)),
    )
    budgets = [float(value) for value in config["budgets_hours"]]
    controlled, monthly, trace = apply_dynamic_hard_budget(
        frame,
        hazard,
        budgets,
        controller,
        events=event_table,
        trace_all_steps=True,
    )
    observed_lookup = controlled[["issue_time", "observed_6h"]].drop_duplicates("issue_time")
    trace = trace.merge(observed_lookup, on="issue_time", how="left", validate="many_to_one")
    trace["locked_true_bins"] = (
        trace["occupied_bins"] - trace["confirmed_false_bins"] - trace["pending_bins"]
    ).astype(int)
    trace["station_month"] = trace["station_code"].astype(str) + "|" + trace["issue_time"].dt.to_period("M").astype(str)
    case_trace = trace.loc[trace["station_month"].eq("F5260|2024-03")].copy()
    if case_trace.empty:
        raise RuntimeError("Frozen replay produced no March 2024 trace for F5260")

    records = pd.read_csv(EXPERIMENT_ROOT / "frozen_event_records.csv.gz", low_memory=False)
    records["onset_time"] = pd.to_datetime(records["onset_time"], errors="coerce")
    records = records.loc[
        records["method"].eq("uadhbac")
        & records["budget_hours"].eq(10.0)
        & records["station_code"].astype(str).eq("F5260")
        & records["onset_time"].dt.to_period("M").eq(pd.Period("2024-03"))
    ][["event_id", "hit", "effective_lead_hours", "operational_evaluable"]]
    case_events = event_table.loc[event_table["onset_time"].dt.to_period("M").eq(pd.Period("2024-03"))][
        ["event_id", "station_code", "station_name", "onset_time", "end_time"]
    ].merge(records, on="event_id", how="left", validate="one_to_one")
    case_events = case_events.sort_values("onset_time").reset_index(drop=True)
    case_events["event_short"] = [f"E{index + 1}" for index in range(case_events.shape[0])]

    case_trace.to_csv(trace_path, index=False, compression="gzip")
    case_events.to_csv(events_path, index=False)
    month_rows = monthly.loc[monthly["station_month"].eq("F5260|2024-03")].sort_values("budget_hours")
    month_rows.to_csv(source_dir / "fig7_F5260_2024-03_month_state.csv", index=False)

    summary = {
        "station_code": "F5260",
        "station_name": str(case_events["station_name"].dropna().iloc[0]) if case_events["station_name"].notna().any() else "",
        "station_month": "2024-03",
        "events": int(case_events.shape[0]),
        "hits_at_10h": int(pd.to_numeric(case_events["hit"], errors="coerce").fillna(0).sum()),
        "misses_at_10h": int(case_events.shape[0] - pd.to_numeric(case_events["hit"], errors="coerce").fillna(0).sum()),
        "contains_false_alarm_capacity": bool(month_rows["confirmed_false_bins"].gt(0).any()),
        "contains_unsettled_capacity": bool(month_rows["unsettled_bins"].gt(0).any()),
        "contains_true_release": bool(month_rows["released_true_bins"].gt(0).any()),
        "locked_true_policy": bool(not controller.release_true_reserve),
        "selected_parameter_sha256": sha256_file(selected_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "event_table_sha256": sha256_file(EVENT_TABLE),
        "controller_contract_version": str(controlled["controller_contract_version"].iloc[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_trace, case_events, summary


def plot_figure7(
    output: Path, source_dir: Path, force_case: bool = False
) -> tuple[dict[str, str], dict[str, Any]]:
    trace, events, summary = prepare_figure7_data(source_dir, force=force_case)
    events = events.sort_values("onset_time").reset_index(drop=True)
    window_start = events["onset_time"].min() - pd.Timedelta(hours=7)
    window_end = events["onset_time"].max() + pd.Timedelta(hours=7)
    trace = trace.loc[trace["issue_time"].between(window_start, window_end)].copy()
    if trace.empty:
        raise RuntimeError("Figure 7 display window is empty")

    fig = plt.figure(figsize=(7.25, 6.10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.72, 1.18], hspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)

    budget_trace = trace.loc[trace["budget_hours"].eq(10.0)].sort_values("issue_time")
    times = pd.to_datetime(budget_trace["issue_time"])
    values = budget_trace["utility_value"].to_numpy(float)
    thresholds = budget_trace["decision_threshold"].to_numpy(float)
    alarms = budget_trace["alarm"].to_numpy(int).astype(bool)
    ax_a.plot(times, values, color=FIVE_GROUP[1], lw=1.55, label=r"risk value  $v_t$", zorder=3)
    ax_a.plot(times, thresholds, color=FIVE_GROUP[0], lw=1.25, ls=(0, (4, 2)), label=r"dynamic threshold  $\tau_t$", zorder=3)
    ax_a.fill_between(times, thresholds, values, where=values >= thresholds, color=FIVE_GROUP[3], alpha=0.38, interpolate=True, zorder=1)
    ax_a.scatter(times[alarms], values[alarms], s=11, color=FIVE_GROUP[4], edgecolor=COLORS["white"], linewidth=0.35, zorder=4, label="accepted at 10 h")
    ax_a.set_ylabel("Lead-time value (h)")
    ax_a.set_ylim(0, max(float(np.nanmax(values)), float(np.nanmax(thresholds))) * 1.22)
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 1.005), ncol=3, fontsize=6.3, handlelength=2.4, frameon=True, facecolor=COLORS["white"], edgecolor="none", framealpha=0.94)
    style_axis(ax_a, grid=True)
    add_panel_label(ax_a, "a", x=-0.045)
    ax_a.set_title(r"Value, dynamic threshold and observed icing onset  ($B=10$ h)", fontsize=8.5, fontweight="bold", loc="left", pad=7)

    budget_order = [2.0, 5.0, 10.0, 20.0]
    budget_colors = FOUR_GROUP
    for yi, budget, color in zip(range(4), budget_order, budget_colors):
        part = trace.loc[trace["budget_hours"].eq(budget)].sort_values("issue_time")
        accepted = part.loc[part["alarm"].eq(1)]
        settled = accepted.loc[pd.to_numeric(accepted["observed_6h"], errors="coerce").fillna(0).ge(0.5)]
        censored = accepted.loc[pd.to_numeric(accepted["observed_6h"], errors="coerce").fillna(0).lt(0.5)]
        ax_b.scatter(settled["issue_time"], np.full(settled.shape[0], yi), marker="s", s=15, color=color, edgecolor="none", zorder=3)
        ax_b.scatter(censored["issue_time"], np.full(censored.shape[0], yi), marker="D", s=18, facecolor=COLORS["white"], edgecolor=color, linewidth=0.9, zorder=4)
    ax_b.set_yticks(range(4), ["2 h", "5 h", "10 h", "20 h"])
    ax_b.set_ylim(-0.65, 3.65)
    ax_b.set_ylabel("Budget")
    style_axis(ax_b, grid=False)
    add_panel_label(ax_b, "b", x=-0.045)
    ax_b.set_title("Nested 10-min alert actions", fontsize=8.5, fontweight="bold", loc="left", pad=6)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["muted"], markeredgecolor="none", markersize=5, label="complete follow-up"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["white"], markeredgecolor=COLORS["muted"], markersize=5, label="right-censored / unsettled"),
        ],
        loc="upper right",
        ncol=2,
        fontsize=6.2,
        frameon=True,
        facecolor=COLORS["white"],
        edgecolor="none",
        framealpha=0.94,
    )

    f_hours = budget_trace["confirmed_false_bins"].to_numpy(float) / 6.0
    p_hours = budget_trace["pending_bins"].to_numpy(float) / 6.0
    l_hours = budget_trace["locked_true_bins"].to_numpy(float) / 6.0
    o_hours = budget_trace["occupied_bins"].to_numpy(float) / 6.0
    ax_c.stackplot(times, f_hours, p_hours, l_hours, colors=[FIVE_GROUP[3], FIVE_GROUP[1], FIVE_GROUP[0]], labels=["F · confirmed false", "P · pending/unsettled", "L · locked true"], alpha=0.90, zorder=1)
    ax_c.plot(times, o_hours, color=FIVE_GROUP[2], lw=1.70, label="O · total occupied", zorder=3)
    ax_c.axhline(10.0, color=FIVE_GROUP[4], lw=1.25, ls=(0, (4, 2)), label=r"capacity  $C=10$ h", zorder=2)
    ax_c.set_ylim(0, 11.5)
    ax_c.set_ylabel("Capacity charged (h)")
    ax_c.set_xlabel("Frozen issue time (10-min grid)")
    style_axis(ax_c, grid=True)
    add_panel_label(ax_c, "c", x=-0.045)
    ax_c.set_title(r"Issue-month capacity state  $O=F+P+L\leq C$  ($B=10$ h)", fontsize=8.5, fontweight="bold", loc="left", pad=6)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0, 0.985), ncol=5, fontsize=5.8, handlelength=1.5, columnspacing=1.1, frameon=True, facecolor=COLORS["white"], edgecolor="none", framealpha=0.94)
    ax_c.text(0.99, 0.08, "L = 0 under the frozen release-on-hit policy", transform=ax_c.transAxes, ha="right", va="center", fontsize=6.0, color=COLORS["muted"])

    for _, event in events.iterrows():
        onset = pd.Timestamp(event["onset_time"])
        if onset < window_start or onset > window_end:
            continue
        hit = int(pd.to_numeric(pd.Series([event.get("hit", 0)]), errors="coerce").fillna(0).iloc[0])
        event_color = COLORS["duo6_dark"] if hit else COLORS["duo3_dark"]
        for ax in (ax_a, ax_b, ax_c):
            ax.axvline(onset, color=event_color, lw=0.85, ls=(0, (2, 2)), alpha=0.78, zorder=2)
        ax_a.text(onset, ax_a.get_ylim()[1] * 0.86, f"{event['event_short']} {'hit' if hit else 'miss'}", rotation=90, ha="right", va="top", fontsize=5.8, color=event_color, fontweight="bold", bbox=dict(facecolor=COLORS["white"], edgecolor="none", alpha=0.78, pad=0.4))

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_c.xaxis.set_major_locator(locator)
    ax_c.xaxis.set_major_formatter(formatter)
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", labelbottom=False)
    ax_c.set_xlim(window_start, window_end)

    fig.text(0.5, 0.992, "A real frozen station-month shows risk, nested actions and delayed capacity settlement", ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.963, f"Station {summary['station_code']} · March 2024 · 4 observed events · 3 hits / 1 miss at B=10 h", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.91, bottom=0.10)
    outputs = export_figure(fig, output / "fig7_real_station_month_case")
    contract = {
        "core_conclusion": "A real frozen replay demonstrates how dynamic value thresholds, nested actions, event-triggered release and unsettled reservations interact within a station-month capacity limit.",
        "archetype": "asymmetric multi-track time-series figure",
        "panels": {"a": "risk value and threshold", "b": "four nested budget actions", "c": "F/P/L/O capacity state at 10 h"},
        "case": summary,
        "source_data": [
            "source_data/fig7_F5260_2024-03_trace.csv.gz",
            "source_data/fig7_F5260_2024-03_events.csv",
            "source_data/fig7_F5260_2024-03_month_state.csv",
        ],
    }
    return outputs, contract


def write_captions(output: Path) -> None:
    captions = """Fig. 2 | Delayed-feedback capacity accounting and causal event matching. a, Every accepted alert immediately enters pending capacity P. A fully observed non-event transfers capacity from P to F; an observed event either releases the reservation or transfers it to L according to policy; right-censored alerts remain charged. The issue-month invariant is O=F+P+L≤C. b, A 10-min decision grid combines a 24-h input history with 36 future hazard steps. The evaluation anchor is floor(onset−1 ns), so an onset at 00:28 is anchored at 00:20. Capacity remains assigned to the alert's issue month across the month boundary, and one alert segment matches at most one event.

Fig. 4 | Frozen mountain-icing performance across monthly alert budgets. a,b, Operational-realistic event hit rates in 2023 and 2024; labels give hit events over evaluable events. c,d, Lead-time utility for the same frozen runs. All points satisfy maximum reserved false-alarm hours no greater than budget, zero online hard-budget violations and zero cross-budget nesting conflicts. Model and controller parameters were selected using 2022 pooled out-of-fold predictions only.

Fig. 5 | Lead-time utility gain over frozen baselines. Differences are UAD-HBAC minus the named baseline for the operational-realistic queue at 5-h and 10-h monthly budgets. Points show paired estimates and lines show 95% confidence intervals from 5,000 station-cluster bootstrap samples (21 stations in 2023; 28 in 2024). Filled points remain significant after Holm correction; the hollow DMD point at 2024, 10 h has an unadjusted positive interval but Holm-adjusted P=0.143.

Fig. 6 | Cross-budget nesting and conflict elimination. a, Coupled decisions enforce A(2 h)⊆A(5 h)⊆A(10 h)⊆A(20 h), so more capacity can add but cannot retract lower-budget alerts. b, Removing coupling yields 2,574 and 2,952 adjacent-budget conflicts in 2023 and 2024, whereas UAD-HBAC yields none. The aggregate frozen point-estimate utility cost is 2.91%; only the 2023 10-h operational cost remains significant after Holm correction.

Fig. 7 | Real frozen controller replay at station F5260 in March 2024. a, Lead-time value, dynamic 10-h threshold, accepted decisions and observed icing onsets. b, Ten-minute alert actions across four nested budgets; hollow diamonds mark alerts whose complete six-hour follow-up is unavailable and therefore remains unsettled. c, Confirmed false, pending and locked capacity at 10 h, with total occupied capacity O and the monthly limit C. The release-on-hit frozen policy gives L=0. Four events occur in the displayed station-month; three are hit and one is missed at 10 h.

Fig. 8 | Unseen-station generalization under two distinct evaluation contracts. PR-AUC changes relative to the all-station reference are shown separately for the 2022 station-group holdout and the 2023 spatial–temporal evaluation. Lines show 95% station-cluster bootstrap intervals from 5,000 samples over 27 stations. No pooled estimate is calculated because the experiments use different temporal and spatial contracts.
"""
    (output / "figure_captions.txt").write_text(captions, encoding="utf-8")


def build_manifest(
    output: Path,
    contracts: dict[str, dict[str, Any]],
    files: dict[str, dict[str, str]],
) -> None:
    source_dir = output / "source_data"
    source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "Python / Matplotlib only",
        "visual_style": "user-specified grouped palettes: six two-color pairs, one four-group set, and one five-group set",
        "formats": ["svg", "pdf", "png", "tiff"],
        "svg_editable_text": True,
        "figures": files,
        "contracts": contracts,
        "source_data_sha256": {
            str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
            for path in source_files
        },
        "primary_input_sha256": {
            "frozen_main_results.csv": sha256_file(EXPERIMENT_ROOT / "frozen_main_results.csv"),
            "paired_station_cluster_bootstrap.csv": sha256_file(EXPERIMENT_ROOT / "paired_station_cluster_bootstrap.csv"),
            "standard_baseline_station_cluster_bootstrap.csv": sha256_file(EXPERIMENT_ROOT / "standard_baseline_station_cluster_bootstrap.csv"),
            "nesting_utility_cost.csv": sha256_file(EXPERIMENT_ROOT / "nesting_utility_cost.csv"),
            "paired_generalization_gap_bootstrap.csv": sha256_file(SPATIAL_ROOT / "paired_generalization_gap_bootstrap.csv"),
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures 2 and 4–8.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force-case", action="store_true", help="Recompute the real Figure 7 frozen trace")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("fig2", "fig4", "fig5", "fig6", "fig7", "fig8"),
        default=None,
    )
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.only or ("fig2", "fig4", "fig5", "fig6", "fig7", "fig8"))
    files: dict[str, dict[str, str]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    plotters = {
        "fig2": lambda: plot_figure2(output),
        "fig4": lambda: plot_figure4(output, source_dir),
        "fig5": lambda: plot_figure5(output, source_dir),
        "fig6": lambda: plot_figure6(output, source_dir),
        "fig7": lambda: plot_figure7(output, source_dir, force_case=args.force_case),
        "fig8": lambda: plot_figure8(output, source_dir),
    }
    for name in ("fig2", "fig4", "fig5", "fig6", "fig7", "fig8"):
        if name not in requested:
            continue
        figure_files, contract = plotters[name]()
        files[name] = figure_files
        contracts[name] = contract
        print(f"Generated {name}: {figure_files['svg']}", flush=True)

    write_captions(output)
    build_manifest(output, contracts, files)
    (output / ".complete_manuscript_figures").touch()
    print(f"Figure package complete: {output}", flush=True)


if __name__ == "__main__":
    main()
