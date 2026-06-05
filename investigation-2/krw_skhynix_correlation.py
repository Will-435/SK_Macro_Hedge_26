"""
krw_skhynix_correlation.py

Focused two-asset correlation pipeline. Extracts the Pearson and Spearman
rank correlations between the daily log returns of the USD/KRW exchange
rate and SK Hynix (000660.KS).

The full-sample Pearson and Spearman are written to a single-row CSV (and
Parquet). A 252-day rolling Pearson and rolling Spearman are written to a
time-series Parquet and rendered onto one line chart for visual inspection.

Sign convention:
    USD/KRW is quoted as KRW per USD. A positive daily log return on
    'r_krw' therefore corresponds to KRW depreciation against the dollar.
    SK Hynix is a KOSPI-listed equity quoted in KRW; its log return is in
    local currency terms.

Run:
    python krw_skhynix_correlation.py (start 2015-01-01, end 2026-05-01)
"""

# Imports
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# Tickers (friendly name to Yahoo Finance symbol).
KRW_NAME = "krw"
KRW_SYMBOL = "KRW=X"
SKHYNIX_NAME = "skhynix"
SKHYNIX_SYMBOL = "000660.KS" # Because this is an asset on the korean stock exchange

# Calendar and window sizes.
TRADING_DAYS_PER_YEAR = 252
ROLLING_CORR_WINDOW = 252
ROLLING_CORR_MIN_PERIODS = 200

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: figure size.
ROLLING_CORR_FIGSIZE = (12, 5)

# Plotting: colours. Pearson on red, Spearman on blue, zero reference in
# neutral dark. Red and blue chosen because the two methods are the
# headline comparison in this file.
PEARSON_LINE_COLOUR = "#C1272D"
SPEARMAN_LINE_COLOUR = "#1F4E79"
REFERENCE_AXIS_COLOUR = "#222222"
CAPTION_COLOUR = "dimgray"

# Plotting: line widths, alphas, dpi.
ROLLING_LINE_LINEWIDTH = 1.4
REFERENCE_AXIS_LINEWIDTH = 0.5
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Plotting: caption layout.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18

# Filesystem layout. One sub-directory per script keeps each pipeline's
# outputs cleanly separated from any other pipeline in this investigation.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "krw_skhynix_correlation"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / SCRIPT_SUBDIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" / SCRIPT_SUBDIR
VISUALS_DIR = PROJECT_ROOT / "visuals" / SCRIPT_SUBDIR

# CLI defaults.
DEFAULT_START_DATE = "2015-01-01"


# Logging.
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("krw-skhynix-correlation")


# Result containers (dataclasses).

@dataclass
class StaticCorrelationResult:
    """
    Full-sample Pearson and Spearman rank correlation between two daily
    log-return series. Written to the single-row CSV that is this file's
    primary deliverable.

    INPUTS:
        * asset_a
        * asset_b
        * n_obs
        * pearson
        * spearman

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    asset_a: str
    asset_b: str
    n_obs: int
    pearson: float
    spearman: float


@dataclass
class PipelineSummary:
    """
    Top-level run summary, persisted as JSON in the processed-data
    directory. Records the date window, observation count and the
    headline static correlation numbers.

    INPUTS:
        * run_at
        * start
        * end
        * n_obs
        * static_correlation
        * notes

    OUTPUTS:
        * Dataclass aggregating the run metadata.
    """
    run_at: str
    start: str
    end: str
    n_obs: int
    static_correlation: Optional[StaticCorrelationResult] = None
    notes: List[str] = field(default_factory = list)


# Persistence helpers.

def save_parquet(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to Parquet under the supplied directory, creating
    the directory if it does not yet exist. Parquet is the preferred
    on-disk format because it preserves dtypes and is faster to reload
    than CSV.

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


def save_csv(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to CSV under the supplied directory. The static
    correlation table is also written to CSV because it is the headline
    deliverable that downstream notebooks and write-ups will read directly.

    INPUTS:
        * frame      : DataFrame to write
        * directory  : target directory
        * name_stem  : file name without extension

    OUTPUTS:
        * Path to the written CSV file.
    """
    directory.mkdir(parents = True, exist_ok = True)
    target_path = directory / f"{name_stem}.csv"
    frame.to_csv(target_path, index = False)
    return target_path


# Data acquisition.

def download_one_ticker(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download one Yahoo Finance ticker and flatten any MultiIndex columns
    that newer yfinance versions return. Raises if no data comes back so
    that downstream alignment is not silently broken.

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
    Pull the close-price column out of a raw yfinance frame and rename it
    to the friendly identifier so downstream panel columns are easy to
    identify.

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
    Download USD/KRW and SK Hynix close prices and align them on a
    forward-filled common trading calendar. Raw frames are cached as
    Parquet under DATA_RAW_DIR so re-runs can inspect the un-processed
    inputs without refetching.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Two-column DataFrame indexed by date with columns 'krw' and
          'skhynix' carrying daily close prices.
    """
    log.info("Downloading %s (%s) and %s (%s)",
             KRW_NAME, KRW_SYMBOL, SKHYNIX_NAME, SKHYNIX_SYMBOL)

    krw_raw = download_one_ticker(KRW_SYMBOL, start_date, end_date)
    save_parquet(krw_raw, DATA_RAW_DIR, f"raw_{KRW_NAME}")
    krw_close = extract_close_series(krw_raw, KRW_NAME)

    skhynix_raw = download_one_ticker(SKHYNIX_SYMBOL, start_date, end_date)
    save_parquet(skhynix_raw, DATA_RAW_DIR, f"raw_{SKHYNIX_NAME}")
    skhynix_close = extract_close_series(skhynix_raw, SKHYNIX_NAME)

    panel = pd.concat([krw_close, skhynix_close], axis = 1)
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

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price panel to daily log returns and prefix each column
    name with 'r_'.

    INPUTS:
        * prices  : DataFrame of close prices

    OUTPUTS:
        * DataFrame of daily log returns; one column per input column.
    """
    returns = np.log(prices).diff().dropna()
    new_columns: List[str] = []
    for column_name in returns.columns:
        new_columns.append(f"r_{column_name}")
    returns.columns = new_columns
    return returns


# Correlations.

def compute_static_correlations(returns: pd.DataFrame) -> StaticCorrelationResult:
    """
    Compute the full-sample Pearson and Spearman rank correlations
    between the two columns of the returns frame. Pearson captures linear
    co-movement; Spearman captures monotonic co-movement and is robust to
    the magnitude of any individual day.

    INPUTS:
        * returns  : two-column DataFrame of daily log returns

    OUTPUTS:
        * Populated StaticCorrelationResult record.
    """
    if returns.shape[1] != 2:
        raise RuntimeError(
            f"compute_static_correlations expects exactly two columns, "
            f"got {returns.shape[1]}"
        )
    col_a, col_b = returns.columns
    pearson_value = float(returns[col_a].corr(returns[col_b], method = "pearson"))
    spearman_value = float(returns[col_a].corr(returns[col_b], method = "spearman"))
    return StaticCorrelationResult(
        asset_a = col_a,
        asset_b = col_b,
        n_obs = int(len(returns)),
        pearson = pearson_value,
        spearman = spearman_value,
    )


def compute_rolling_correlations(
    returns: pd.DataFrame,
    window: int = ROLLING_CORR_WINDOW,
    min_periods: int = ROLLING_CORR_MIN_PERIODS,
) -> pd.DataFrame:
    """
    Compute a rolling Pearson and rolling Spearman correlation between
    the two columns of the returns frame. Both lines share the same
    window so they are directly comparable.

    INPUTS:
        * returns      : two-column DataFrame of daily log returns
        * window       : rolling window length in trading days
        * min_periods  : minimum window observations required for a point

    OUTPUTS:
        * DataFrame indexed by date with two columns: 'pearson' and
          'spearman'.
    """
    if returns.shape[1] != 2:
        raise RuntimeError(
            f"compute_rolling_correlations expects exactly two columns, "
            f"got {returns.shape[1]}"
        )
    col_a, col_b = returns.columns
    series_a = returns[col_a]
    series_b = returns[col_b]
    rolling_pearson = series_a.rolling(window = window, min_periods = min_periods).corr(series_b)
    rolling_spearman = (
        series_a.rank(pct = True)
        .rolling(window = window, min_periods = min_periods)
        .corr(series_b.rank(pct = True))
    )
    return pd.DataFrame({
        "pearson":  rolling_pearson,
        "spearman": rolling_spearman,
    })


# Plotting helpers.

def add_caption(fig: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption underneath the figure so the saved PNG
    carries its own self-contained explanation of the axes and any
    symbols used.

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
    Save a matplotlib figure into VISUALS_DIR at the project default
    resolution and close it.

    INPUTS:
        * fig        : matplotlib Figure
        * file_name  : output file name including extension

    OUTPUTS:
        * Path to the written PNG.
    """
    output_path = VISUALS_DIR / file_name
    fig.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(fig)
    return output_path


def plot_rolling_correlation(rolling_frame: pd.DataFrame) -> Path:
    """
    Render the rolling Pearson and rolling Spearman correlations as two
    lines on a single axes. The chart is the headline visual for this
    file because it shows whether the static correlation reported in the
    CSV reflects a stable relationship or is averaging across regime
    flips.

    INPUTS:
        * rolling_frame  : DataFrame with 'pearson' and 'spearman' columns
                           indexed by date

    OUTPUTS:
        * Path to the saved PNG.
    """
    pearson_series = rolling_frame["pearson"].dropna()
    spearman_series = rolling_frame["spearman"].dropna()

    fig, ax = plt.subplots(figsize = ROLLING_CORR_FIGSIZE)
    ax.plot(
        pearson_series.index, pearson_series.values,
        color = PEARSON_LINE_COLOUR, lw = ROLLING_LINE_LINEWIDTH,
        label = "Pearson",
    )
    ax.plot(
        spearman_series.index, spearman_series.values,
        color = SPEARMAN_LINE_COLOUR, lw = ROLLING_LINE_LINEWIDTH,
        label = "Spearman rank",
    )
    ax.axhline(
        0, color = REFERENCE_AXIS_COLOUR, lw = REFERENCE_AXIS_LINEWIDTH,
    )
    ax.set_title(
        f"Rolling {ROLLING_CORR_WINDOW} day correlation: KRW vs SK Hynix"
    )
    ax.set_ylabel("Correlation coefficient")
    ax.legend(loc = "upper left")
    fig.tight_layout()
    add_caption(
        fig,
        "Rolling 252 day Pearson and Spearman-rank correlation between "
        "the daily log return of USD/KRW and SK Hynix (000660.KS). "
        "Positive 'r_krw' means KRW depreciation against the dollar, so a "
        "positive correlation here means SK Hynix tends to rise on KRW "
        "weakening days. Each point is stamped at the window end-date.",
    )
    return save_figure(fig, "rolling_correlation.png")


# JSON serialisation helper.

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert dataclasses, dicts, lists and numpy scalars into
    JSON-friendly Python types so the pipeline summary can be written
    directly with json.dumps.

    INPUTS:
        * obj  : any Python object that may contain dataclasses or numpy
                 scalars

    OUTPUTS:
        * JSON-friendly equivalent (dict, list, float, int or str).
    """
    if hasattr(obj, "__dataclass_fields__"):
        result_dict: Dict[str, Any] = {}
        for field_name, field_value in asdict(obj).items():
            result_dict[field_name] = to_jsonable(field_value)
        return result_dict
    if isinstance(obj, dict):
        out_dict: Dict[str, Any] = {}
        for dict_key, dict_value in obj.items():
            out_dict[str(dict_key)] = to_jsonable(dict_value)
        return out_dict
    if isinstance(obj, (list, tuple)):
        out_list: List[Any] = []
        for list_item in obj:
            out_list.append(to_jsonable(list_item))
        return out_list
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


# Plot styling.

def configure_plot_style() -> None:
    """
    Apply project-wide matplotlib defaults: seaborn whitegrid background
    and consistent global font sizes. Line colours are set explicitly per
    plot rather than through a cycler so this file only needs the two
    palette constants defined above.

    INPUTS:
        * None

    OUTPUTS:
        * None. This just applies the global variables defined in the preamble.
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

def run_pipeline(start_date: str, end_date: str) -> PipelineSummary:
    """
    Execute the end-to-end pipeline: fetch the price pair, compute daily
    log returns, extract the full-sample Pearson and Spearman rank
    correlations, compute the 252-day rolling versions of both, persist
    every table as Parquet (and the headline static correlation as CSV
    as well), render the rolling-correlation visual, and write a JSON
    run summary.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Populated PipelineSummary, also persisted as JSON in
          DATA_PROCESSED_DIR.
    """
    for required_dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALS_DIR):
        required_dir.mkdir(parents = True, exist_ok = True)
    configure_plot_style()

    prices = fetch_pair_panel(start_date, end_date)
    save_parquet(prices, DATA_PROCESSED_DIR, "prices")

    returns = to_log_returns(prices)
    save_parquet(returns, DATA_PROCESSED_DIR, "log_returns")

    log.info("Computing full-sample Pearson and Spearman correlations")
    static_result = compute_static_correlations(returns)
    static_frame = pd.DataFrame([{
        "asset_a":  static_result.asset_a,
        "asset_b":  static_result.asset_b,
        "n_obs":    static_result.n_obs,
        "pearson":  static_result.pearson,
        "spearman": static_result.spearman,
    }])
    save_csv(static_frame, DATA_PROCESSED_DIR, "static_correlations")
    save_parquet(static_frame, DATA_PROCESSED_DIR, "static_correlations")

    log.info("Computing rolling %dd Pearson and Spearman", ROLLING_CORR_WINDOW)
    rolling_frame = compute_rolling_correlations(returns)
    save_parquet(rolling_frame, DATA_PROCESSED_DIR, "rolling_correlations")

    log.info("Plotting rolling correlation chart")
    plot_rolling_correlation(rolling_frame)

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(prices.index.min().date()),
        end = str(prices.index.max().date()),
        n_obs = int(len(returns)),
        static_correlation = static_result,
    )
    summary.notes.append(
        "USD/KRW is quoted in KRW per USD; a positive r_krw therefore "
        "means KRW depreciation against the dollar. SK Hynix is KOSPI-"
        "listed and quoted in KRW, so its log return is in local currency "
        "terms."
    )
    summary.notes.append(
        "Spearman is computed via the rank-then-rolling-correlation "
        "approach so the same 252-day window and minimum-periods rule "
        "applies to both methods. Direct .corr(method='spearman') is not "
        "available inside a rolling object in current pandas."
    )

    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent = 2))
    log.info("Wrote summary to %s", summary_path)
    return summary


# Command line interface.

def parse_args() -> argparse.Namespace:
    """
    Parse the start- and end-date command line arguments.

    INPUTS:
        * None (reads sys.argv via argparse)

    OUTPUTS:
        * argparse Namespace with .start and .end string attributes.
    """
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--start", default = DEFAULT_START_DATE)
    parser.add_argument(
        "--end", default = datetime.utcnow().strftime("%Y-%m-%d"),
    )
    return parser.parse_args()


# Main.

def main() -> None:
    """
    Entry point. Runs the full pipeline and prints a concise one-screen
    summary covering the date window, observation count and the two
    headline correlation numbers.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: figure in VISUALS_DIR, Parquet and CSV
          tables in DATA_PROCESSED_DIR, raw caches in DATA_RAW_DIR, JSON
          summary in DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\nPipeline summary")
    print(f"Window  : {summary.start} -> {summary.end}  (n = {summary.n_obs})")
    if summary.static_correlation is not None:
        result = summary.static_correlation
        print(
            f"\nStatic correlation ({result.asset_a} vs {result.asset_b}):"
        )
        print(f"  Pearson  : {result.pearson:+.4f}")
        print(f"  Spearman : {result.spearman:+.4f}")
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")
    print(f"Raw in     : {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
