"""
ftse_skhynix_pearson_correlation.py

Investigation-3 focused two-asset pipeline. Extracts the Pearson linear
correlation between the daily log returns of the FTSE 100 (^FTSE) and SK Hynix
(000660.KS).

The full-sample Pearson is written to a single-row CSV (and Parquet). A 252-day
rolling Pearson is written to a time-series Parquet and rendered onto one line
chart for visual inspection of whether the static number reflects a stable
relationship or averages across regime flips.

NON-OVERLAPPING SESSIONS

The FTSE 100 trades in London and SK Hynix trades in Seoul. The two sessions do
not overlap: Seoul (roughly 09:00 to 15:30 KST) closes before London (roughly
08:00 to 16:30 GMT) opens. A same-calendar-day FTSE return therefore reflects
information that arrived after SK Hynix had already closed for that day, so the
contemporaneous Pearson understates the true linkage; SK Hynix tends to absorb
that information at its next open. The number below is the naive same-day
Pearson and should be read with this lag in mind.

Sign convention:
    Both series are local-currency price indices, FTSE 100 in GBP and SK Hynix
    in KRW. A positive daily log return is a price rise in local terms. No FX
    leg is modelled here.

Run:
    python ftse_skhynix_pearson_correlation.py --start 2015-01-01 --end 2026-05-01
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# Tickers (friendly name to Yahoo Finance symbol).
FTSE_NAME = "ftse100"
FTSE_SYMBOL = "^FTSE"
SKHYNIX_NAME = "skhynix"
SKHYNIX_SYMBOL = "000660.KS" # Korean stock exchange listing.

# Calendar and window sizes.
ROLLING_CORR_WINDOW = 252
ROLLING_CORR_MIN_PERIODS = 200

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: figure size.
ROLLING_CORR_FIGSIZE = (12, 5)

# Plotting: colours. Pearson on red, zero reference in neutral dark.
PEARSON_LINE_COLOUR = "#C1272D"
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

# Filesystem layout. One sub-directory per pipeline keeps these outputs
# separate inside investigation-3.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "ftse_skhynix_pearson_correlation"
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
log = logging.getLogger("ftse-skhynix-pearson-correlation")


# Result containers (dataclasses).

@dataclass
class StaticPearsonResult:
    """
    Full-sample Pearson linear correlation between two daily log-return series.
    Written to the single-row CSV that is this file's primary deliverable.

    INPUTS:
        * asset_a
        * asset_b
        * n_obs
        * pearson

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    asset_a: str
    asset_b: str
    n_obs: int
    pearson: float


@dataclass
class PipelineSummary:
    """
    Top-level run summary, persisted as JSON in the processed-data directory.
    Records the date window, observation count and the headline Pearson number.

    INPUTS:
        * run_at
        * start
        * end
        * n_obs
        * static_pearson
        * notes

    OUTPUTS:
        * Dataclass aggregating the run metadata.
    """
    run_at: str
    start: str
    end: str
    n_obs: int
    static_pearson: Optional[StaticPearsonResult] = None
    notes: List[str] = field(default_factory = list)


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


def save_csv(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to CSV under the supplied directory. The static
    correlation table is written to CSV because it is the headline deliverable
    that the write-up reads directly.

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
    Download the FTSE 100 and SK Hynix close prices and align them on a
    forward-filled common trading calendar. Raw frames are cached as Parquet
    under DATA_RAW_DIR so re-runs can inspect the un-processed inputs without
    refetching.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Two-column DataFrame indexed by date with columns 'ftse100' and
          'skhynix' carrying daily close prices.
    """
    log.info("Downloading %s (%s) and %s (%s)",
             FTSE_NAME, FTSE_SYMBOL, SKHYNIX_NAME, SKHYNIX_SYMBOL)

    ftse_raw = download_one_ticker(FTSE_SYMBOL, start_date, end_date)
    save_parquet(ftse_raw, DATA_RAW_DIR, f"raw_{FTSE_NAME}")
    ftse_close = extract_close_series(ftse_raw, FTSE_NAME)

    skhynix_raw = download_one_ticker(SKHYNIX_SYMBOL, start_date, end_date)
    save_parquet(skhynix_raw, DATA_RAW_DIR, f"raw_{SKHYNIX_NAME}")
    skhynix_close = extract_close_series(skhynix_raw, SKHYNIX_NAME)

    panel = pd.concat([ftse_close, skhynix_close], axis = 1)
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
    Convert a price panel to daily log returns and prefix each column name
    with 'r_'.

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

def compute_static_pearson(returns: pd.DataFrame) -> StaticPearsonResult:
    """
    Compute the full-sample Pearson linear correlation between the two columns
    of the returns frame. Pearson captures linear co-movement and is sensitive
    to the magnitude of any individual day.

    INPUTS:
        * returns  : two-column DataFrame of daily log returns

    OUTPUTS:
        * Populated StaticPearsonResult record.
    """
    if returns.shape[1] != 2:
        raise RuntimeError(
            f"compute_static_pearson expects exactly two columns, "
            f"got {returns.shape[1]}"
        )
    col_a, col_b = returns.columns
    pearson_value = float(returns[col_a].corr(returns[col_b], method = "pearson"))
    return StaticPearsonResult(
        asset_a = col_a,
        asset_b = col_b,
        n_obs = int(len(returns)),
        pearson = pearson_value,
    )


def compute_rolling_pearson(
    returns: pd.DataFrame,
    window: int = ROLLING_CORR_WINDOW,
    min_periods: int = ROLLING_CORR_MIN_PERIODS,
) -> pd.DataFrame:
    """
    Compute a rolling Pearson correlation between the two columns of the
    returns frame so the stability of the static number can be inspected.

    INPUTS:
        * returns      : two-column DataFrame of daily log returns
        * window       : rolling window length in trading days
        * min_periods  : minimum window observations required for a point

    OUTPUTS:
        * DataFrame indexed by date with a single 'pearson' column.
    """
    if returns.shape[1] != 2:
        raise RuntimeError(
            f"compute_rolling_pearson expects exactly two columns, "
            f"got {returns.shape[1]}"
        )
    col_a, col_b = returns.columns
    rolling_pearson = returns[col_a].rolling(
        window = window, min_periods = min_periods,
    ).corr(returns[col_b])
    return pd.DataFrame({"pearson": rolling_pearson})


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


def plot_rolling_pearson(rolling_frame: pd.DataFrame) -> Path:
    """
    Render the rolling Pearson correlation as one line. The chart is the
    headline visual for this file because it shows whether the static Pearson
    reported in the CSV reflects a stable relationship or averages across
    regime flips.

    INPUTS:
        * rolling_frame  : DataFrame with a 'pearson' column indexed by date

    OUTPUTS:
        * Path to the saved PNG.
    """
    pearson_series = rolling_frame["pearson"].dropna()

    fig, ax = plt.subplots(figsize = ROLLING_CORR_FIGSIZE)
    ax.plot(
        pearson_series.index, pearson_series.values,
        color = PEARSON_LINE_COLOUR, lw = ROLLING_LINE_LINEWIDTH,
        label = "Pearson",
    )
    ax.axhline(
        0, color = REFERENCE_AXIS_COLOUR, lw = REFERENCE_AXIS_LINEWIDTH,
    )
    ax.set_title(
        f"Rolling {ROLLING_CORR_WINDOW}-day Pearson correlation: "
        "FTSE 100 vs SK Hynix"
    )
    ax.set_ylabel("Correlation coefficient")
    ax.legend(loc = "upper left")
    fig.tight_layout()
    add_caption(
        fig,
        "Rolling 252-day Pearson correlation between the daily log return of "
        "the FTSE 100 (^FTSE) and SK Hynix (000660.KS). London and Seoul "
        "sessions do not overlap, so the same-day return pairs a Seoul close "
        "with a later London close and the contemporaneous correlation "
        "understates the true linkage. Each point is stamped at the window "
        "end-date.",
    )
    return save_figure(fig, "rolling_pearson_correlation.png")


# JSON serialisation helper.

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert dataclasses, dicts, lists and numpy scalars into
    JSON-friendly Python types so the pipeline summary can be written directly
    with json.dumps.

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
    Apply the seaborn whitegrid background and the project font sizes. The
    line colour is set explicitly in the plot so only the single palette
    constant is needed here.

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

def run_pipeline(start_date: str, end_date: str) -> PipelineSummary:
    """
    Execute the end-to-end pipeline: fetch the price pair, compute daily log
    returns, extract the full-sample Pearson, compute the 252-day rolling
    Pearson, persist every table as Parquet (and the headline Pearson as CSV),
    render the rolling-Pearson visual, and write a JSON run summary.

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

    log.info("Computing full-sample Pearson correlation")
    static_result = compute_static_pearson(returns)
    static_frame = pd.DataFrame([{
        "asset_a":  static_result.asset_a,
        "asset_b":  static_result.asset_b,
        "n_obs":    static_result.n_obs,
        "pearson":  static_result.pearson,
    }])
    save_csv(static_frame, DATA_PROCESSED_DIR, "static_pearson")
    save_parquet(static_frame, DATA_PROCESSED_DIR, "static_pearson")

    log.info("Computing rolling %dd Pearson", ROLLING_CORR_WINDOW)
    rolling_frame = compute_rolling_pearson(returns)
    save_parquet(rolling_frame, DATA_PROCESSED_DIR, "rolling_pearson")

    log.info("Plotting rolling Pearson chart")
    plot_rolling_pearson(rolling_frame)

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(prices.index.min().date()),
        end = str(prices.index.max().date()),
        n_obs = int(len(returns)),
        static_pearson = static_result,
    )
    summary.notes.append(
        "London and Seoul sessions do not overlap: Seoul closes before London "
        "opens, so a same-calendar-day FTSE return reflects information SK "
        "Hynix had not yet traded on. The contemporaneous Pearson therefore "
        "understates the true linkage, which SK Hynix absorbs at its next open."
    )
    summary.notes.append(
        "Both series are local-currency price indices, FTSE 100 in GBP and SK "
        "Hynix in KRW; returns are in local terms and no FX leg is modelled."
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
    Entry point. Runs the full pipeline and prints a concise one-screen summary
    covering the date window, observation count and the headline Pearson number.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: figure in VISUALS_DIR, Parquet and CSV tables in
          DATA_PROCESSED_DIR, raw caches in DATA_RAW_DIR, JSON summary in
          DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\nPipeline summary")
    print(f"Window  : {summary.start} -> {summary.end}  (n = {summary.n_obs})")
    if summary.static_pearson is not None:
        result = summary.static_pearson
        print(f"\nStatic Pearson ({result.asset_a} vs {result.asset_b}):")
        print(f"  Pearson  : {result.pearson:+.4f}")
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")
    print(f"Raw in     : {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
