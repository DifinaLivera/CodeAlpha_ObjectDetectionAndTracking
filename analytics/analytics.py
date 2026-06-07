# ─────────────────────────────────────────────────────────────────────────────
# analytics/analytics.py
#
# Reads the in-memory event log (list of dicts) or a saved CSV and produces
# three publication-quality graphs:
#   • entries_per_hour.png
#   • exits_per_hour.png
#   • occupancy_over_time.png
#
# Uses Pandas for aggregation and Matplotlib for rendering.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in threads
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Colour palette (dark theme) ───────────────────────────────────────────────
STYLE      = "dark_background"
CLR_ENTRY  = "#00FF88"
CLR_EXIT   = "#FF4455"
CLR_OCC    = "#FFAA00"
CLR_GRID   = "#333333"
FONT_TITLE = {"fontsize": 14, "fontweight": "bold", "color": "white"}
FONT_AXIS  = {"fontsize": 10, "color": "#AAAAAA"}


def _load_df(event_log: List[Dict]) -> pd.DataFrame:
    """Convert the list-of-dicts event log into a tidy DataFrame."""
    if not event_log:
        return pd.DataFrame(columns=["timestamp", "event", "track_id", "occupancy"])

    df = pd.DataFrame(event_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")
    df["minute"] = df["timestamp"].dt.floor("min")
    return df


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Graph saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Individual graph generators
# ─────────────────────────────────────────────────────────────────────────────

def plot_entries_per_hour(
    event_log: List[Dict],
    out_path: Path | str,
) -> Optional[Path]:
    """Bar chart: entries per hour."""
    out_path = Path(out_path)
    df = _load_df(event_log)
    entries = df[df["event"] == "ENTRY"]

    if entries.empty:
        logger.warning("No ENTRY events – skipping entries_per_hour graph.")
        return None

    counts = entries.groupby("hour").size().reset_index(name="count")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        bars = ax.bar(
            counts["hour"], counts["count"],
            color=CLR_ENTRY, width=0.03, edgecolor="none", alpha=0.85,
        )

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.1,
                str(int(h)), ha="center", va="bottom",
                fontsize=9, color="white",
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        fig.autofmt_xdate()

        ax.set_title("Entries per Hour", **FONT_TITLE, pad=14)
        ax.set_xlabel("Hour", **FONT_AXIS)
        ax.set_ylabel("People", **FONT_AXIS)
        ax.tick_params(colors="#AAAAAA")
        ax.grid(axis="y", color=CLR_GRID, linewidth=0.6)
        ax.spines[:].set_color(CLR_GRID)

        _save(fig, out_path)
    return out_path


def plot_exits_per_hour(
    event_log: List[Dict],
    out_path: Path | str,
) -> Optional[Path]:
    """Bar chart: exits per hour."""
    out_path = Path(out_path)
    df = _load_df(event_log)
    exits = df[df["event"] == "EXIT"]

    if exits.empty:
        logger.warning("No EXIT events – skipping exits_per_hour graph.")
        return None

    counts = exits.groupby("hour").size().reset_index(name="count")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        bars = ax.bar(
            counts["hour"], counts["count"],
            color=CLR_EXIT, width=0.03, edgecolor="none", alpha=0.85,
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.1,
                str(int(h)), ha="center", va="bottom",
                fontsize=9, color="white",
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        fig.autofmt_xdate()

        ax.set_title("Exits per Hour", **FONT_TITLE, pad=14)
        ax.set_xlabel("Hour", **FONT_AXIS)
        ax.set_ylabel("People", **FONT_AXIS)
        ax.tick_params(colors="#AAAAAA")
        ax.grid(axis="y", color=CLR_GRID, linewidth=0.6)
        ax.spines[:].set_color(CLR_GRID)

        _save(fig, out_path)
    return out_path


def plot_occupancy_over_time(
    event_log: List[Dict],
    out_path: Path | str,
) -> Optional[Path]:
    """Line chart: occupancy value over time (per-event granularity)."""
    out_path = Path(out_path)
    df = _load_df(event_log)

    if df.empty:
        logger.warning("No events – skipping occupancy_over_time graph.")
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        ax.fill_between(
            df["timestamp"], df["occupancy"],
            alpha=0.25, color=CLR_OCC,
        )
        ax.plot(
            df["timestamp"], df["occupancy"],
            color=CLR_OCC, linewidth=1.8, label="Occupancy",
        )

        # Mark entry / exit events
        entry_df = df[df["event"] == "ENTRY"]
        exit_df  = df[df["event"] == "EXIT"]
        if not entry_df.empty:
            ax.scatter(
                entry_df["timestamp"], entry_df["occupancy"],
                color=CLR_ENTRY, s=22, zorder=5, label="Entry", marker="^",
            )
        if not exit_df.empty:
            ax.scatter(
                exit_df["timestamp"], exit_df["occupancy"],
                color=CLR_EXIT, s=22, zorder=5, label="Exit", marker="v",
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

        ax.set_title("Occupancy Over Time", **FONT_TITLE, pad=14)
        ax.set_xlabel("Time", **FONT_AXIS)
        ax.set_ylabel("Occupancy (people)", **FONT_AXIS)
        ax.tick_params(colors="#AAAAAA")
        ax.grid(color=CLR_GRID, linewidth=0.6)
        ax.spines[:].set_color(CLR_GRID)
        ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white")

        _save(fig, out_path)
    return out_path


def plot_combined_dashboard(
    event_log: List[Dict],
    out_path: Path | str,
) -> Optional[Path]:
    """
    A single 2×2 dashboard figure combining all three charts plus a
    summary table – useful for embedding in the PDF report.
    """
    out_path = Path(out_path)
    df = _load_df(event_log)

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(16, 10), facecolor="#111111")
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        # ── 1. Entries per hour ──────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#111111")
        entries = df[df["event"] == "ENTRY"]
        if not entries.empty:
            c = entries.groupby("hour").size().reset_index(name="count")
            ax1.bar(c["hour"], c["count"], color=CLR_ENTRY, width=0.03, alpha=0.85)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.set_title("Entries / Hour", **FONT_TITLE)
        ax1.tick_params(colors="#AAAAAA", rotation=30)
        ax1.grid(axis="y", color=CLR_GRID, lw=0.6)
        ax1.spines[:].set_color(CLR_GRID)

        # ── 2. Exits per hour ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#111111")
        exits = df[df["event"] == "EXIT"]
        if not exits.empty:
            c2 = exits.groupby("hour").size().reset_index(name="count")
            ax2.bar(c2["hour"], c2["count"], color=CLR_EXIT, width=0.03, alpha=0.85)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.set_title("Exits / Hour", **FONT_TITLE)
        ax2.tick_params(colors="#AAAAAA", rotation=30)
        ax2.grid(axis="y", color=CLR_GRID, lw=0.6)
        ax2.spines[:].set_color(CLR_GRID)

        # ── 3. Occupancy over time ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor("#111111")
        if not df.empty:
            ax3.fill_between(df["timestamp"], df["occupancy"], alpha=0.2, color=CLR_OCC)
            ax3.plot(df["timestamp"], df["occupancy"], color=CLR_OCC, lw=1.8, label="Occupancy")
            if not entries.empty:
                ax3.scatter(entries["timestamp"], entries["occupancy"], color=CLR_ENTRY, s=20, zorder=5, label="Entry", marker="^")
            if not exits.empty:
                ax3.scatter(exits["timestamp"], exits["occupancy"], color=CLR_EXIT, s=20, zorder=5, label="Exit", marker="v")
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax3.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white")
        ax3.set_title("Occupancy Over Time", **FONT_TITLE)
        ax3.tick_params(colors="#AAAAAA", rotation=30)
        ax3.grid(color=CLR_GRID, lw=0.6)
        ax3.spines[:].set_color(CLR_GRID)

        fig.suptitle(
            "Occupancy Monitoring – Session Analytics",
            fontsize=16, fontweight="bold", color="white", y=1.01,
        )

        _save(fig, out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate ALL graphs at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_graphs(
    event_log: List[Dict],
    graphs_dir: Path | str = "graphs",
) -> Dict[str, Optional[Path]]:
    """
    Generate all three individual graphs + the combined dashboard.

    Returns a dict mapping graph name → saved Path (or None if skipped).
    """
    d = Path(graphs_dir)
    return {
        "entries_per_hour":    plot_entries_per_hour(event_log,    d / "entries_per_hour.png"),
        "exits_per_hour":      plot_exits_per_hour(event_log,      d / "exits_per_hour.png"),
        "occupancy_over_time": plot_occupancy_over_time(event_log, d / "occupancy_over_time.png"),
        "dashboard":           plot_combined_dashboard(event_log,  d / "dashboard.png"),
    }