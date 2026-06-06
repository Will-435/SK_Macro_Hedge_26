"""
skhynix_kospi_returns_overlay.py

Investigation-3 visual pipeline. Overlays the one-year cumulative return paths
of SK Hynix (000660.KS) and the wider KOSPI index (^KS11) on a single chart.

Both series are rebased to a common base of 100 at the start of the trailing
one-year window, so the chart reads as a direct comparison of total return
over the horizon: the gap between the two lines is the relative performance of
SK Hynix against its home market, and the difference in amplitude shows SK
Hynix's higher beta to the index.

Both instruments are quoted in KRW, so the comparison is apples-to-apples with
no FX leg. Note that SK Hynix is itself a KOSPI constituent, so part of the
index path is the stock's own move; the overlay is best read as SK Hynix
against the rest of the market.

Run:
    python skhynix_kospi_returns_overlay.py --end 2026-05-01 --horizon-days 252
"""

# Imports
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# Tickers (friendly name to Yahoo Finance symbol).
SKHYNIX_NAME = "skhynix"
SKHYNIX_SYMBOL = "000660.KS"
KOSPI_NAME = "kospi"
KOSPI_SYMBOL = "^KS11"

# Horizon. One trading year is 252 trading days. A generous calendar buffer is
# fetched so that 252 aligned trading rows are available after weekends and
# Korean market holidays are dropped.
HORIZON_TRADING_DAYS = 252
FETCH_BUFFER_CALENDAR_DAYS = 420

# Rebasing. Both series start the window at this common base so the lines are
# directly comparable as cumulative return.
REBASE_BASE = 100.0

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: figure size.
OVERLAY_FIGSIZE = (12, 6)

# Plotting: colours. SK Hynix on red, KOSPI on blue, base line in neutral dark.
SKHYNIX_LINE_COLOUR = "#C1272D"
KOSPI_LINE_COLOUR = "#1F4E79"
REFERENCE_AXIS_COLOUR = "#222222"
CAPTION_COLOUR = "dimgray"

# Plotting: line widths, dpi, style.
RETURN_LINE_LINEWIDTH = 1.6
REFERENCE_AXIS_LINEWIDTH = 0.6
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Plotting: caption layout.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18

# Filesystem layout. One sub-directory per pipeline keeps these outputs
# separate inside investigation-3.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "skhynix_kospi_returns_overlay"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / SCRIPT_SUBDIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" / SCRIPT_SUBDIR
VISUALS_DIR = PROJECT_ROOT / "visuals" / SCRIPT_SUBDIR


# Logging.
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("skhynix-kospi-returns-overlay")


# Persistence helpers.

def save_parquet(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to Parquet under the supplied directory, creating it if
    needed. Parquet preserves dtypes and reloads faster than CSV, so it is the
    default on-disk format for downstream reuse.

    INPUTS:
        * frame      : DataFrame to write
        * directory  : target directory
        * name_stem  : file name without extension

    OUTPUTS:
        * Path to the written Parquet file.
    """
    directory.mkdir(parents = True, exist_ok = True)
    target_path = directory / f"{name_stem}.parquet"
    frame.to_parquet(target_path)
    return target_path


# Data acquisition.

def download_one_ticker(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download one Yahoo Finance ticker and flatten any MultiIndex columns that
    newer yfinance versions return. Raises if no data comes back so downstream
    alignment is not silently broken.

    INPUTS:
        * symbol      : Yahoo Finance ticker symbol
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * DataFrame of OHLCV-like columns indexed by date.
    """
    raw_frame = yf.download(
        symbol,
        start = start_date,
        end = end_date,
        progress = False,
        auto_adjust = True,
    )
    if raw_frame is None or raw_frame.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(raw_frame.columns, pd.MultiIndex):
        raw_frame.columns = raw_frame.columns.get_level_values(0)
    return raw_frame


def extract_close_series(raw_frame: pd.DataFrame, friendly_name: str) -> pd.Series:
    """
    Pull the close-price column out of a raw yfinance frame and rename it to
    the friendly identifier so downstream panel columns are easy to identify.

    INPUTS:
        * raw_frame      : DataFrame returned by yf.download
        * friendly_name  : output Series name

    OUTPUTS:
        * Close-price Series renamed to friendly_name.
    """
    close_col = "Close" if "Close" in raw_frame.columns else raw_frame.columns[0]
    close_series = raw_frame[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series.rename(friendly_name)


def fetch_pair_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download SK Hynix and KOSPI close prices and align them on a forward-filled
    common trading calendar. Raw frames are cached as Parquet under
    DATA_RAW_DIR so re-runs can inspect the un-processed inputs.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Two-column DataFrame indexed by date with columns 'skhynix' and
          'kospi' carrying daily close prices.
    """
    log.info("Downloading %s (%s) and %s (%s)",
             SKHYNIX_NAME, SKHYNIX_SYMBOL, KOSPI_NAME, KOSPI_SYMBOL)

    skhynix_raw = download_one_ticker(SKHYNIX_SYMBOL, start_date, end_date)
    save_parquet(skhynix_raw, DATA_RAW_DIR, f"raw_{SKHYNIX_NAME}")
    skhynix_close = extract_close_series(skhynix_raw, SKHYNIX_NAME)

    kospi_raw = download_one_ticker(KOSPI_SYMBOL, start_date, end_date)
    save_parquet(kospi_raw, DATA_RAW_DIR, f"raw_{KOSPI_NAME}")
    kospi_close = extract_close_series(kospi_raw, KOSPI_NAME)

    panel = pd.concat([skhynix_close, kospi_close], axis = 1)
    panel = panel.dropna(how = "all")
    panel = panel.ffill(limit = PANEL_FFILL_LIMIT).dropna()

    log.info(
        "Aligned panel: %d rows (%s to %s)",
        panel.shape[0],
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel


# Returns.

def slice_to_horizon(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Keep only the final horizon_days trading rows of the aligned price panel so
    the overlay covers exactly the trailing one-year window.

    INPUTS:
        * prices        : aligned two-column price panel
        * horizon_days  : number of trailing trading rows to keep

    OUTPUTS:
        * Trimmed price panel with at most horizon_days rows.
    """
    return prices.iloc[-horizon_days:]


def rebase_to_base(prices: pd.DataFrame, base: float = REBASE_BASE) -> pd.DataFrame:
    """
    Rebase each price column to a common base at the first row of the window so
    the two series can be compared directly as cumulative return. An indexed
    value of 120 means a 20 per cent gain since the window start.

    INPUTS:
        * prices  : trimmed price panel covering the horizon
        * base    : common starting value for every column

    OUTPUTS:
        * DataFrame of indexed series, every column starting at base.
    """
    return prices / prices.iloc[0] * base


# Plotting helpers.

def add_caption(fig: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption underneath the figure so the saved PNG carries
    its own self-contained explanation of the axes and any caveats.

    INPUTS:
        * fig            : matplotlib Figure
        * caption_text   : caption string

    OUTPUTS:
        * None. Mutates the figure in place.
    """
    fig.subplots_adjust(bottom = CAPTION_BOTTOM_PAD)
    fig.text(
        CAPTION_X_POSITION, CAPTION_Y_POSITION, caption_text,
        ha = "center", va = "bottom",
        fontsize = TEXT_FONT, color = CAPTION_COLOUR, wrap = True,
    )


def save_figure(fig: plt.Figure, file_name: str) -> Path:
    """
    Save a matplotlib figure into VISUALS_DIR at the project default resolution
    and close it.

    INPUTS:
        * fig        : matplotlib Figure
        * file_name  : output file name including extension

    OUTPUTS:
        * Path to the written PNG.
    """
    VISUALS_DIR.mkdir(parents = True, exist_ok = True)
    output_path = VISUALS_DIR / file_name
    fig.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(fig)
    return output_path


def plot_returns_overlay(indexed: pd.DataFrame) -> Path:
    """
    Render the two rebased return paths on one axes. The final cumulative
    return of each series is written into its legend label so the relative
    performance is legible without reading the axis.

    INPUTS:
        * indexed  : DataFrame with 'skhynix' and 'kospi' columns, both
                     starting at the common base

    OUTPUTS:
        * Path to the saved PNG.
    """
    skhynix_indexed = indexed[SKHYNIX_NAME]
    kospi_indexed = indexed[KOSPI_NAME]
    skhynix_total_return = (skhynix_indexed.iloc[-1] / REBASE_BASE - 1.0) * 100.0
    kospi_total_return = (kospi_indexed.iloc[-1] / REBASE_BASE - 1.0) * 100.0

    fig, ax = plt.subplots(figsize = OVERLAY_FIGSIZE)

    # SK Hynix path.
    ax.plot(
        skhynix_indexed.index, skhynix_indexed.values,
        color = SKHYNIX_LINE_COLOUR, lw = RETURN_LINE_LINEWIDTH,
        label = f"SK Hynix ({skhynix_total_return:+.1f}%)",
    )

    # KOSPI path.
    ax.plot(
        kospi_indexed.index, kospi_indexed.values,
        color = KOSPI_LINE_COLOUR, lw = RETURN_LINE_LINEWIDTH,
        label = f"KOSPI ({kospi_total_return:+.1f}%)",
    )

    ax.axhline(
        REBASE_BASE, color = REFERENCE_AXIS_COLOUR, lw = REFERENCE_AXIS_LINEWIDTH,
    )
    ax.set_title("SK Hynix vs KOSPI: one-year indexed return (start = 100)")
    ax.set_ylabel("Indexed return (start = 100)")
    ax.legend(loc = "upper left")
    fig.tight_layout()
    add_caption(
        fig,
        "One-year cumulative return paths of SK Hynix (000660.KS) and the "
        "KOSPI index (^KS11), both rebased to 100 at the window start and both "
        "quoted in KRW. The gap between the lines is SK Hynix's relative "
        "performance against its home market; the wider amplitude reflects its "
        "higher beta to the index. SK Hynix is itself a KOSPI constituent, so "
        "the overlay is best read as SK Hynix against the rest of the market.",
    )
    return save_figure(fig, "skhynix_kospi_returns_overlay.png")


# Plot styling.

def configure_plot_style() -> None:
    """
    Apply the seaborn whitegrid background and the project font sizes. Line
    colours are set explicitly in the plot so only the two palette constants
    are needed here.

    INPUTS:
        * None

    OUTPUTS:
        * None. Mutates plt.rcParams.
    """
    sns.set_style(SEABORN_STYLE)
    plt.rcParams.update({
        "figure.titlesize": TITLE_FONT,
        "axes.titlesize":   TITLE_FONT,
        "axes.labelsize":   TEXT_FONT,
        "xtick.labelsize":  TEXT_FONT,
        "ytick.labelsize":  TEXT_FONT,
        "legend.fontsize":  TEXT_FONT,
    })


# Orchestration.

def run_pipeline(end_date: str, horizon_days: int) -> Dict[str, object]:
    """
    Execute the end-to-end overlay: fetch SK Hynix and KOSPI over a buffered
    window ending at end_date, align them, trim to the trailing horizon_days
    trading rows, rebase both to a common base, persist the indexed series and
    render the overlay chart.

    INPUTS:
        * end_date      : ISO date string, exclusive upper bound on the window
        * horizon_days  : number of trailing trading days to plot

    OUTPUTS:
        * Dictionary summary of the run, also written to JSON.
    """
    for required_dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALS_DIR):
        required_dir.mkdir(parents = True, exist_ok = True)
    configure_plot_style()

    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    start_datetime = end_datetime - timedelta(days = FETCH_BUFFER_CALENDAR_DAYS)
    start_date = start_datetime.strftime("%Y-%m-%d")

    prices = fetch_pair_panel(start_date, end_date)
    horizon_prices = slice_to_horizon(prices, horizon_days)
    save_parquet(horizon_prices, DATA_PROCESSED_DIR, "horizon_prices")

    indexed = rebase_to_base(horizon_prices)
    save_parquet(indexed, DATA_PROCESSED_DIR, "indexed_returns")

    log.info("Plotting the one-year returns overlay")
    plot_returns_overlay(indexed)

    skhynix_total_return = float(indexed[SKHYNIX_NAME].iloc[-1] / REBASE_BASE - 1.0)
    kospi_total_return = float(indexed[KOSPI_NAME].iloc[-1] / REBASE_BASE - 1.0)
    summary: Dict[str, object] = {
        "run_at": datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        "window_start": str(indexed.index.min().date()),
        "window_end": str(indexed.index.max().date()),
        "n_trading_days": int(len(indexed)),
        "skhynix_total_return": skhynix_total_return,
        "kospi_total_return": kospi_total_return,
        "relative_return": skhynix_total_return - kospi_total_return,
    }
    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent = 2))
    log.info("Wrote summary to %s", summary_path)
    return summary


# Command line interface.

def parse_args() -> argparse.Namespace:
    """
    Parse the end-date and horizon command line arguments.

    INPUTS:
        * None (reads sys.argv via argparse)

    OUTPUTS:
        * argparse Namespace with .end and .horizon_days attributes.
    """
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--end", default = datetime.utcnow().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--horizon-days", type = int, default = HORIZON_TRADING_DAYS,
    )
    return parser.parse_args()


# Main.

def main() -> None:
    """
    Entry point. Runs the overlay pipeline and prints a one-screen summary of
    the window, the two total returns and SK Hynix's relative performance.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: figure in VISUALS_DIR, Parquet tables in
          DATA_PROCESSED_DIR, raw caches in DATA_RAW_DIR, JSON summary in
          DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.end, cli_args.horizon_days)

    print("\nPipeline summary")
    print(f"Window           : {summary['window_start']} -> {summary['window_end']} "
          f"(n = {summary['n_trading_days']})")
    print(f"SK Hynix return  : {summary['skhynix_total_return'] * 100:+.1f}%")
    print(f"KOSPI return     : {summary['kospi_total_return'] * 100:+.1f}%")
    print(f"Relative (SKH-KOSPI): {summary['relative_return'] * 100:+.1f}%")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
