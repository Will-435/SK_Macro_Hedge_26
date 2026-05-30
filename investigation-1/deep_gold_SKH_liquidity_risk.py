"""
deep_gold_SKH_liquidity_risk.py

Single-ticker deep dive on Gold's behaviour during SK Hynix bearish, high-
realised-volatility regimes. This file is the SK-Hynix-specific companion to
the broader gold time series analysis; it deliberately drops VIX as an input
and works only with the stock's own realised volatility.

Regime definition used throughout this file:
    A trading day is in the 'bearish high-vol' regime for SK Hynix when:
        1. The SK Hynix 4-week (20 trading day) log return is at or below its
           full-sample 10th percentile, AND
        2. The SK Hynix 21-day annualised realised volatility is at or above
           its full-sample 75th percentile.
    Both conditions must hold simultaneously.

The set of figures produced mirrors the multi-ticker pipeline but is adapted
for a single ticker: prices and realised vol panel, realised-vol regime
shading, realised vol time series, 4-week return distribution, trading
volume, conditional gold performance across four sub-samples for comparison,
rolling 252-day conditional correlation, and a 2x2 correlation matrix grid.

Run:
    python deep_gold_SKH_liquidity_risk.py --start 2015-01-01 --end 2026-05-01
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
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# Tickers (friendly name to Yahoo Finance symbol).
GOLD_NAME = "gold"
GOLD_SYMBOL = "GC=F"
TARGET_NAME = "skhynix"
TARGET_SYMBOL = "000660.KS"

# Calendar and window sizes.
TRADING_DAYS_PER_YEAR = 252
FOUR_WEEK_TRADING_DAYS = 20
REALISED_VOL_WINDOW = 21
ROLLING_CORR_WINDOW = 252
VOLUME_AVG_WINDOW = 20

# Quantile thresholds.
BEARISH_QUANTILE = 0.10
HIGH_VOL_QUANTILE = 0.75

# Minimum number of regime-filtered observations required to compute a
# rolling correlation point. Below this the window is skipped.
MIN_OBS_FOR_ROLLING_CORR = 15

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Scaling.
BPS_SCALAR = 1.0e4

# Plotting: global font sizes. Edit these to globally change plot text.
TITLE_FONT = 14
TEXT_FONT = 10
HEATMAP_ANNOT_SIZE = 9

# Plotting: figure sizes.
WIDE_FIGSIZE = (12, 5)
TALL_WIDE_FIGSIZE = (12, 5.5)
BAR_FIGSIZE = (9, 5)
HISTOGRAM_FIGSIZE = (7, 5)
MATRIX_GRID_FIGSIZE = (10, 9)

# Plotting: colours. Unconventional red, green and blue tones.
PRIMARY_RED = "#C1272D"
PRIMARY_GREEN = "#386641"
PRIMARY_BLUE = "#1F4E79"
ACCENT_RED = "#E63946"
ACCENT_GREEN = "#52B788"
ACCENT_BLUE = "#457B9D"
NEUTRAL_DARK = "#222222"
NEUTRAL_GREY = "#5C5C5C"
CAPTION_COLOUR = "dimgray"
HEATMAP_CMAP = "RdBu_r"

# Per-asset colour assignments.
TARGET_COLOUR = PRIMARY_RED
GOLD_COLOUR = PRIMARY_GREEN
REGIME_HIGHLIGHT_COLOUR = PRIMARY_BLUE

# Plotting: line widths, alphas, sizes.
REFERENCE_AXIS_LINEWIDTH = 0.5
SECONDARY_LINE_LINEWIDTH = 0.7
THRESHOLD_LINEWIDTH = 0.8
REGIME_LINE_LINEWIDTH = 1.0
REGIME_FILL_ALPHA = 0.30
SECONDARY_LINE_ALPHA = 0.45
HIT_RATE_REFERENCE_ALPHA = 0.4
HISTOGRAM_BINS = 60
BAR_WIDTH = 0.6
HIT_RATE_AXIS_MIN = 0.0
HIT_RATE_AXIS_MAX = 1.0
HIT_RATE_REFERENCE_VALUE = 0.5
HEATMAP_VMIN = -1.0
HEATMAP_VMAX = 1.0
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Plotting: caption layout.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18

# Sub-sample labels used in the conditional gold comparison plot.
SUBSAMPLE_FULL = "full_sample"
SUBSAMPLE_BEARISH_ONLY = "bearish_only"
SUBSAMPLE_HIGH_RV_ONLY = "high_rv_only"
SUBSAMPLE_REGIME = "bearish_and_high_rv"
SUBSAMPLE_LABELS: List[str] = [
    SUBSAMPLE_FULL,
    SUBSAMPLE_BEARISH_ONLY,
    SUBSAMPLE_HIGH_RV_ONLY,
    SUBSAMPLE_REGIME,
]
SUBSAMPLE_DISPLAY: Dict[str, str] = {
    SUBSAMPLE_FULL:         "Full sample",
    SUBSAMPLE_BEARISH_ONLY: "Bearish only (4w <= P10)",
    SUBSAMPLE_HIGH_RV_ONLY: "High RV only (RV >= P75)",
    SUBSAMPLE_REGIME:       "Bearish + high RV",
}
SUBSAMPLE_COLOURS: Dict[str, str] = {
    SUBSAMPLE_FULL:         NEUTRAL_GREY,
    SUBSAMPLE_BEARISH_ONLY: PRIMARY_BLUE,
    SUBSAMPLE_HIGH_RV_ONLY: PRIMARY_GREEN,
    SUBSAMPLE_REGIME:       PRIMARY_RED,
}

# Filesystem layout. One sub-directory per script keeps each pipeline's
# outputs cleanly separated from any other pipeline in this repository.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "deep_gold_SKH_liquidity_risk"
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
log = logging.getLogger("skh-liquidity-risk")


# Result containers (dataclasses).

@dataclass
class SubsampleGoldStats:
    """
    Summary of gold's behaviour over one sub-sample of the daily index;
    one record per sub-sample variant in the conditional comparison.

    INPUTS:
        * subsample
        * n_obs
        * gold_mean_bps
        * gold_median_bps
        * gold_std_bps
        * gold_hit_rate_positive
        * target_mean_bps
        * corr_pearson
        * corr_spearman

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    subsample: str
    n_obs: int
    gold_mean_bps: float
    gold_median_bps: float
    gold_std_bps: float
    gold_hit_rate_positive: float
    target_mean_bps: float
    corr_pearson: float
    corr_spearman: float


@dataclass
class PipelineSummary:
    """
    Top-level run summary, persisted as JSON in the processed-data directory.

    INPUTS:
        * run_at
        * start
        * end
        * n_obs
        * notes
        * subsample_stats

    OUTPUTS:
        * Dataclass aggregating the run metadata and the four sub-sample
          gold-performance records.
    """
    run_at: str
    start: str
    end: str
    n_obs: int
    notes: List[str] = field(default_factory = list)
    subsample_stats: List[SubsampleGoldStats] = field(default_factory = list)


# Persistence helpers.

def save_table(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a tabular object to Parquet, the preferred format on disk. Series
    inputs are coerced to a single-column DataFrame so the Parquet schema is
    always well defined.

    INPUTS:
        * frame      : pandas DataFrame or Series to write
        * directory  : target directory
        * name_stem  : file name without extension

    OUTPUTS:
        * Path to the written Parquet file.
    """
    directory.mkdir(parents = True, exist_ok = True)
    if isinstance(frame, pd.Series):
        out_frame = frame.to_frame()
    else:
        out_frame = frame
    target_path = directory / f"{name_stem}.parquet"
    out_frame.to_parquet(target_path)
    return target_path


# Data acquisition.

def download_single_ticker(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download one Yahoo Finance ticker and flatten any MultiIndex columns
    that newer yfinance versions return.

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


def fetch_panel(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Build the aligned price, volume and target panel used by the rest of
    the pipeline. Raw downloaded frames are cached as Parquet under
    DATA_RAW_DIR so that subsequent runs can inspect the un-processed
    inputs without refetching.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Dict with keys 'prices' (DataFrame: gold and target close prices)
          and 'volume' (Series: target trading volume), aligned on a
          forward-filled common trading calendar.
    """
    log.info("Downloading gold and %s", TARGET_NAME)

    gold_frame = download_single_ticker(GOLD_SYMBOL, start_date, end_date)
    save_table(gold_frame, DATA_RAW_DIR, f"raw_{GOLD_NAME}")

    target_frame = download_single_ticker(TARGET_SYMBOL, start_date, end_date)
    save_table(target_frame, DATA_RAW_DIR, f"raw_{TARGET_NAME}")

    gold_close_col = "Close" if "Close" in gold_frame.columns else gold_frame.columns[0]
    gold_close = gold_frame[gold_close_col]
    if isinstance(gold_close, pd.DataFrame):
        gold_close = gold_close.iloc[:, 0]
    gold_close = gold_close.rename(GOLD_NAME)

    target_close_col = "Close" if "Close" in target_frame.columns else target_frame.columns[0]
    target_close = target_frame[target_close_col]
    if isinstance(target_close, pd.DataFrame):
        target_close = target_close.iloc[:, 0]
    target_close = target_close.rename(TARGET_NAME)

    if "Volume" in target_frame.columns:
        target_volume = target_frame["Volume"]
        if isinstance(target_volume, pd.DataFrame):
            target_volume = target_volume.iloc[:, 0]
        target_volume = target_volume.rename(TARGET_NAME)
    else:
        target_volume = pd.Series(dtype = float, name = TARGET_NAME)

    prices = pd.concat([gold_close, target_close], axis = 1).dropna(how = "all")
    prices = prices.ffill(limit = PANEL_FFILL_LIMIT).dropna()
    target_volume = target_volume.reindex(prices.index).ffill()

    log.info(
        "Aligned panel: %d rows (%s to %s)",
        prices.shape[0],
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return {"prices": prices, "volume": target_volume}


# Returns, rolling windows, realised volatility.

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price panel to daily log returns and prefix each column name
    with 'r_'.

    INPUTS:
        * prices  : DataFrame of adjusted close prices

    OUTPUTS:
        * DataFrame of daily log returns; one column per input column.
    """
    returns = np.log(prices).diff().dropna()
    new_columns = []
    for col_name in returns.columns:
        new_columns.append(f"r_{col_name}")
    returns.columns = new_columns
    return returns


def rolling_four_week_return(prices: pd.DataFrame, ticker_name: str) -> pd.Series:
    """
    Compute the rolling 4-week (20 trading day) log return for one ticker.

    INPUTS:
        * prices       : DataFrame of adjusted close prices
        * ticker_name  : column name to operate on

    OUTPUTS:
        * Series of rolling 4-week log returns indexed by date.
    """
    log_prices = np.log(prices[ticker_name])
    four_w = log_prices - log_prices.shift(FOUR_WEEK_TRADING_DAYS)
    return four_w.dropna().rename(f"r4w_{ticker_name}")


def rolling_realised_volatility(returns: pd.DataFrame, ticker_name: str) -> pd.Series:
    """
    Compute the annualised rolling realised volatility for one ticker.

    INPUTS:
        * returns      : DataFrame of daily log returns (columns prefixed 'r_')
        * ticker_name  : ticker friendly name (input column is 'r_<ticker>')

    OUTPUTS:
        * Annualised realised-vol Series indexed by date.
    """
    target_returns = returns[f"r_{ticker_name}"]
    rolling_std = target_returns.rolling(REALISED_VOL_WINDOW).std()
    annualised = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annualised.rename(f"rv_{ticker_name}")


# Regime masks.

def bearish_mask_from_4w(four_week: pd.Series) -> pd.Series:
    """
    Identify days where the 4-week log return is in the bottom 10% of the
    full sample.

    INPUTS:
        * four_week  : 4-week return Series

    OUTPUTS:
        * Boolean Series aligned to the input index; True on bearish days.
    """
    threshold = four_week.quantile(BEARISH_QUANTILE)
    return four_week <= threshold


def high_vol_mask_from_rv(realised_vol: pd.Series) -> pd.Series:
    """
    Identify days where the realised volatility is in the top 25% of the
    full sample.

    INPUTS:
        * realised_vol  : annualised realised vol Series

    OUTPUTS:
        * Boolean Series aligned to the input index; True on high-vol days.
    """
    threshold = realised_vol.quantile(HIGH_VOL_QUANTILE)
    return realised_vol >= threshold


def build_regime_masks(four_week: pd.Series, realised_vol: pd.Series) -> Dict[str, pd.Series]:
    """
    Build the boolean masks for the four sub-samples used downstream: the
    full sample, bearish-only, high-realised-vol-only, and the conjunction
    of the two (which is the headline regime).

    INPUTS:
        * four_week     : 4-week return Series for the target ticker
        * realised_vol  : annualised realised-vol Series for the same ticker

    OUTPUTS:
        * Dict of boolean Series keyed by SUBSAMPLE_LABELS, all aligned on
          the intersection of the input indices.
    """
    common_index = four_week.dropna().index.intersection(realised_vol.dropna().index)
    fw_aligned = four_week.reindex(common_index)
    rv_aligned = realised_vol.reindex(common_index)

    full = pd.Series(True, index = common_index)
    bearish = bearish_mask_from_4w(fw_aligned)
    high_vol = high_vol_mask_from_rv(rv_aligned)
    regime = bearish & high_vol

    return {
        SUBSAMPLE_FULL:         full,
        SUBSAMPLE_BEARISH_ONLY: bearish,
        SUBSAMPLE_HIGH_RV_ONLY: high_vol,
        SUBSAMPLE_REGIME:       regime,
    }


# Conditional gold statistics.
def subsample_gold_stats(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    subsample_label: str,
) -> SubsampleGoldStats:
    """
    Compute the headline conditional gold statistics restricted to days
    that satisfy the mask, for one sub-sample variant.

    INPUTS:
        * daily_returns    : DataFrame of daily log returns (cols 'r_...')
        * mask             : boolean Series indexed by date
        * subsample_label  : one of SUBSAMPLE_LABELS

    OUTPUTS:
        * Populated SubsampleGoldStats record.
    """
    aligned_mask = mask.reindex(daily_returns.index).fillna(False).astype(bool)
    subsample_frame = daily_returns.loc[aligned_mask]
    n_subsample_obs = int(len(subsample_frame))

    if n_subsample_obs == 0:
        nan_value = float("nan")
        return SubsampleGoldStats(
            subsample = subsample_label, n_obs = 0,
            gold_mean_bps = nan_value, gold_median_bps = nan_value,
            gold_std_bps = nan_value, gold_hit_rate_positive = nan_value,
            target_mean_bps = nan_value,
            corr_pearson = nan_value, corr_spearman = nan_value,
        )

    gold_returns = subsample_frame[f"r_{GOLD_NAME}"]
    target_returns = subsample_frame[f"r_{TARGET_NAME}"]
    return SubsampleGoldStats(
        subsample = subsample_label,
        n_obs = n_subsample_obs,
        gold_mean_bps = float(gold_returns.mean() * BPS_SCALAR),
        gold_median_bps = float(gold_returns.median() * BPS_SCALAR),
        gold_std_bps = float(gold_returns.std() * BPS_SCALAR),
        gold_hit_rate_positive = float((gold_returns > 0).mean()),
        target_mean_bps = float(target_returns.mean() * BPS_SCALAR),
        corr_pearson = float(gold_returns.corr(target_returns, method = "pearson")),
        corr_spearman = float(gold_returns.corr(target_returns, method = "spearman")),
    )


# Rolling conditional correlation.

def rolling_conditional_correlation(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.Series:
    """
    Within each rolling window, restrict to the days where the regime mask
    is True and compute the Pearson correlation between gold and the
    target. Windows with fewer than MIN_OBS_FOR_ROLLING_CORR regime days
    are skipped, producing a sparse time series instead of NaN-filled
    output.

    INPUTS:
        * daily_returns  : DataFrame with r_<gold> and r_<target>
        * mask           : boolean Series indexed by date
        * window         : rolling window length in trading days

    OUTPUTS:
        * Series of correlations indexed by window end-date.
    """
    aligned_mask = mask.reindex(daily_returns.index).fillna(False).astype(bool).values
    gold_arr = daily_returns[f"r_{GOLD_NAME}"].values
    target_arr = daily_returns[f"r_{TARGET_NAME}"].values
    date_index = daily_returns.index

    rolling_points = []
    n_total = len(daily_returns)
    for window_end in range(window, n_total):
        slice_mask = aligned_mask[window_end - window:window_end]
        if slice_mask.sum() < MIN_OBS_FOR_ROLLING_CORR:
            continue
        gold_slice = gold_arr[window_end - window:window_end][slice_mask]
        target_slice = target_arr[window_end - window:window_end][slice_mask]
        if gold_slice.std() == 0 or target_slice.std() == 0:
            continue
        corr_val = float(np.corrcoef(gold_slice, target_slice)[0, 1])
        rolling_points.append((date_index[window_end], corr_val))

    if len(rolling_points) == 0:
        return pd.Series(dtype = float)
    indexed_series = pd.Series(dict(rolling_points))
    return indexed_series


# Regime-filtered correlation matrices.

def subsample_correlation_matrix(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    method: str,
) -> pd.DataFrame:
    """
    Build a correlation matrix over [r_<gold>, r_<target>] restricted to
    days that satisfy the supplied mask.

    INPUTS:
        * daily_returns  : DataFrame of daily log returns
        * mask           : boolean Series indexed by date
        * method         : 'pearson' or 'spearman'

    OUTPUTS:
        * 2x2 correlation matrix DataFrame, or an empty DataFrame if there
          are no observations.
    """
    aligned_mask = mask.reindex(daily_returns.index).fillna(False).astype(bool)
    subset_frame = daily_returns.loc[aligned_mask]
    if len(subset_frame) == 0:
        return pd.DataFrame()
    cols = [f"r_{GOLD_NAME}", f"r_{TARGET_NAME}"]
    return subset_frame[cols].corr(method = method)


# Plotting helpers.

def add_caption(fig: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption beneath the figure. Reserves a vertical
    margin so that the caption is not clipped on save.

    INPUTS:
        * fig            : matplotlib Figure
        * caption_text   : caption string; defines every symbol in the figure

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
    Save a matplotlib figure into VISUALS_DIR and close it.

    INPUTS:
        * fig        : matplotlib Figure
        * file_name  : file name with extension

    OUTPUTS:
        * Path to the written file.
    """
    output_path = VISUALS_DIR / file_name
    fig.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(fig)
    return output_path


def plot_prices_and_realised_vol(prices: pd.DataFrame, realised_vol: pd.Series) -> Path:
    """
    Render gold and SK Hynix prices rebased to 100, with the target's
    realised volatility on a secondary right axis.

    INPUTS:
        * prices        : DataFrame with gold and target close prices
        * realised_vol  : 21d annualised realised vol of the target

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax_left = plt.subplots(figsize = TALL_WIDE_FIGSIZE)
    normalised = prices / prices.iloc[0] * 100
    ax_left.plot(
        normalised.index, normalised[GOLD_NAME],
        color = GOLD_COLOUR, label = "gold",
    )
    ax_left.plot(
        normalised.index, normalised[TARGET_NAME],
        color = TARGET_COLOUR, label = TARGET_NAME,
    )
    ax_left.set_ylabel("Price index (start = 100)")
    ax_left.set_title("Normalised prices and SK Hynix realised vol")
    ax_left.legend(loc = "upper left")

    ax_right = ax_left.twinx()
    ax_right.plot(
        realised_vol.index, realised_vol.values,
        color = NEUTRAL_GREY, alpha = SECONDARY_LINE_ALPHA,
        lw = SECONDARY_LINE_LINEWIDTH, label = "realised vol",
    )
    ax_right.set_ylabel("Annualised realised vol")
    ax_right.legend(loc = "upper right")
    fig.tight_layout()
    add_caption(
        fig,
        "Daily close prices for gold (GC=F) and SK Hynix (000660.KS), "
        "rebased so each series starts at 100. The grey line on the right "
        "axis is the SK Hynix 21-day annualised realised volatility. "
        "RV_t = std(r_{t-20}, ..., r_t) x sqrt(252), where r_t is the daily "
        "log return.",
    )
    return save_figure(fig, "prices_and_realised_vol.png")


def plot_realised_vol_regime(realised_vol: pd.Series) -> Path:
    """
    Render the target ticker's realised volatility with the top-quartile
    high-vol regime shaded.

    INPUTS:
        * realised_vol  : 21d annualised realised vol Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    threshold = realised_vol.quantile(HIGH_VOL_QUANTILE)
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    ax.plot(
        realised_vol.index, realised_vol.values,
        color = PRIMARY_BLUE, lw = REGIME_LINE_LINEWIDTH,
    )
    ax.fill_between(
        realised_vol.index, realised_vol.values, threshold,
        where = (realised_vol.values >= threshold),
        color = PRIMARY_RED, alpha = REGIME_FILL_ALPHA,
        label = "RV in top quartile",
    )
    ax.axhline(
        threshold, color = PRIMARY_RED, lw = THRESHOLD_LINEWIDTH,
        linestyle = "--", label = f"75th pctile = {threshold:.2f}",
    )
    ax.set_title("SK Hynix realised volatility with high-vol regime highlighted")
    ax.set_ylabel("Annualised RV")
    ax.legend()
    fig.tight_layout()
    add_caption(
        fig,
        "Blue line: SK Hynix 21-day annualised realised volatility. "
        "Shaded red region: days where RV is at or above its full-sample "
        "75th percentile (the high-vol condition of the headline regime). "
        "RV_t = std(r_{t-20}, ..., r_t) x sqrt(252).",
    )
    return save_figure(fig, "realised_vol_regime.png")


def plot_realised_volatility(realised_vol: pd.Series) -> Path:
    """
    Render the target's realised volatility on its own panel, without any
    regime overlay.

    INPUTS:
        * realised_vol  : 21d annualised realised vol Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    ax.plot(
        realised_vol.index, realised_vol.values,
        color = TARGET_COLOUR, label = TARGET_NAME,
    )
    ax.set_title(f"{REALISED_VOL_WINDOW}d annualised realised volatility")
    ax.set_ylabel("Annualised RV")
    ax.legend()
    fig.tight_layout()
    add_caption(
        fig,
        "21-day rolling annualised realised volatility for SK Hynix. "
        "RV_t = std(r_{t-20}, ..., r_t) x sqrt(252), where r_t is the "
        "daily log return.",
    )
    return save_figure(fig, "realised_volatility.png")


def plot_four_week_return_distribution(four_week: pd.Series) -> Path:
    """
    Render the SK Hynix 4-week return histogram with the 10th-percentile
    bearish threshold marked.

    INPUTS:
        * four_week  : 4-week return Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    threshold = four_week.dropna().quantile(BEARISH_QUANTILE)
    fig, ax = plt.subplots(figsize = HISTOGRAM_FIGSIZE)
    sns.histplot(
        four_week.dropna(),
        bins = HISTOGRAM_BINS, kde = True, ax = ax,
        color = TARGET_COLOUR,
    )
    ax.axvline(
        threshold, color = PRIMARY_BLUE, linestyle = "--",
        label = f"P10 = {threshold:+.3f}",
    )
    ax.set_title("SK Hynix 4-week return distribution")
    ax.set_xlabel("r4w")
    ax.legend()
    fig.tight_layout()
    add_caption(
        fig,
        "Histogram of the SK Hynix 4-week (20 trading day) rolling log "
        "return. r4w = log(P_t / P_{t-20}). The dashed blue line marks P10, "
        "the 10th percentile of the distribution, which is the bearish "
        "threshold used in the headline regime.",
    )
    return save_figure(fig, "four_week_return_distribution.png")


def plot_target_volume(volume: pd.Series) -> Path:
    """
    Render the 20-day rolling mean daily trading volume of the target
    ticker.

    INPUTS:
        * volume  : daily trading volume Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    smoothed = volume.rolling(VOLUME_AVG_WINDOW).mean()
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    ax.plot(
        smoothed.index, smoothed.values,
        color = TARGET_COLOUR, label = TARGET_NAME,
    )
    ax.set_title(f"{VOLUME_AVG_WINDOW}d rolling mean trading volume")
    ax.set_ylabel("Shares")
    ax.legend()
    fig.tight_layout()
    add_caption(
        fig,
        "20-day rolling mean of SK Hynix daily trading volume (shares). "
        "Provided as background context only; trading volume is not part "
        "of the regime-filter logic.",
    )
    return save_figure(fig, "trading_volume.png")


def plot_conditional_gold_comparison(stats_list: List[SubsampleGoldStats]) -> Path:
    """
    Render a four-bar comparison of gold's mean daily return across the
    full sample, bearish-only, high-realised-vol-only, and the combined
    headline regime. Gold's hit-rate-positive is overlayed on a secondary
    right axis.

    INPUTS:
        * stats_list  : list of SubsampleGoldStats, one per sub-sample

    OUTPUTS:
        * Path to the saved PNG.
    """
    ordered_stats = []
    for label in SUBSAMPLE_LABELS:
        for stat in stats_list:
            if stat.subsample == label:
                ordered_stats.append(stat)
                break

    fig, ax_left = plt.subplots(figsize = BAR_FIGSIZE)
    labels = []
    means = []
    hit_rates = []
    n_obs_values = []
    bar_colours = []
    for stat in ordered_stats:
        labels.append(SUBSAMPLE_DISPLAY[stat.subsample])
        means.append(stat.gold_mean_bps)
        hit_rates.append(stat.gold_hit_rate_positive)
        n_obs_values.append(stat.n_obs)
        bar_colours.append(SUBSAMPLE_COLOURS[stat.subsample])

    bar_positions = np.arange(len(ordered_stats))
    ax_left.bar(
        bar_positions, means,
        width = BAR_WIDTH, color = bar_colours,
        label = "Gold mean (bps)",
    )
    ax_left.set_xticks(bar_positions)
    tick_labels = []
    for label, n_obs_value in zip(labels, n_obs_values):
        tick_labels.append(f"{label}\n(n = {n_obs_value})")
    ax_left.set_xticklabels(tick_labels, fontsize = TEXT_FONT)
    ax_left.axhline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax_left.set_ylabel("Mean daily gold return (bps)")
    ax_left.set_title("Gold performance across sub-samples")

    ax_right = ax_left.twinx()
    ax_right.plot(
        bar_positions, hit_rates,
        color = ACCENT_RED, marker = "o",
        label = "Gold hit-rate positive",
    )
    ax_right.set_ylim(HIT_RATE_AXIS_MIN, HIT_RATE_AXIS_MAX)
    ax_right.set_ylabel("Fraction of days with r_gold > 0")
    ax_right.axhline(
        HIT_RATE_REFERENCE_VALUE,
        color = ACCENT_RED, linestyle = ":",
        alpha = HIT_RATE_REFERENCE_ALPHA,
    )

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc = "upper left",
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Gold's mean daily log return (bars, left axis) and hit-rate-"
        "positive (red line, right axis) across four sub-samples of the "
        "SK Hynix index. bps = basis points = 1e-4. n = number of days in "
        "each sub-sample. The headline regime is the right-most bar "
        "(bearish 4w return AND high realised vol).",
    )
    return save_figure(fig, "conditional_gold_comparison.png")


def plot_rolling_conditional_correlation(rolling_series: pd.Series) -> Path:
    """
    Render the rolling 252-day Pearson correlation between gold and the
    target, restricted within each window to days that satisfy the
    headline regime mask.

    INPUTS:
        * rolling_series  : rolling correlation Series from
                            rolling_conditional_correlation

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    if rolling_series.empty:
        ax.text(0.5, 0.5, "no data", ha = "center", va = "center")
    else:
        ax.plot(
            rolling_series.index, rolling_series.values,
            color = PRIMARY_RED, label = f"gold vs {TARGET_NAME}",
        )
    ax.axhline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_title(
        f"Rolling {ROLLING_CORR_WINDOW}d conditional corr (gold vs SK Hynix)\n"
        "[bearish 4w return + high realised vol]"
    )
    ax.set_ylabel("Conditional rho")
    ax.legend()
    fig.tight_layout()
    add_caption(
        fig,
        "Rolling 252-day Pearson correlation between r_gold and r_skhynix, "
        "restricted within each window to days that satisfy the headline "
        "regime mask. rho = Pearson correlation. r_X = daily log return of "
        "asset X. Windows with fewer than 15 regime days are skipped.",
    )
    return save_figure(fig, "rolling_corr_regime.png")


def plot_correlation_matrices_grid(
    matrices_by_subsample_and_method: Dict[Tuple[str, str], pd.DataFrame],
) -> Path:
    """
    Render a 2x2 grid of correlation matrices over [r_gold, r_skhynix]:
    rows compare the full sample with the headline regime; columns compare
    Pearson and Spearman.

    INPUTS:
        * matrices_by_subsample_and_method  : dict keyed by
          (subsample_label, method)

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize = MATRIX_GRID_FIGSIZE)
    rows = [SUBSAMPLE_FULL, SUBSAMPLE_REGIME]
    cols = ["pearson", "spearman"]
    for row_idx, subsample_label in enumerate(rows):
        for col_idx, method in enumerate(cols):
            ax = axes[row_idx, col_idx]
            matrix = matrices_by_subsample_and_method.get(
                (subsample_label, method)
            )
            if matrix is None or matrix.empty:
                ax.text(0.5, 0.5, "no data", ha = "center", va = "center")
                ax.set_axis_off()
                continue
            sns.heatmap(
                matrix, annot = True, fmt = ".2f", cmap = HEATMAP_CMAP,
                vmin = HEATMAP_VMIN, vmax = HEATMAP_VMAX,
                square = True, ax = ax, cbar = True,
                annot_kws = {"size": HEATMAP_ANNOT_SIZE},
            )
            ax.set_title(
                f"{method.capitalize()} | {SUBSAMPLE_DISPLAY[subsample_label]}",
                fontsize = TEXT_FONT,
            )
    fig.suptitle(
        "Correlation matrices: full sample vs headline regime",
        fontsize = TITLE_FONT,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Correlation matrices over [r_gold, r_skhynix]. Top row: full "
        "sample. Bottom row: headline regime days (bearish 4w + high RV). "
        "Left column: Pearson (linear). Right column: Spearman (rank-"
        "based, robust to outliers). r_X = daily log return of asset X.",
    )
    return save_figure(fig, "correlation_matrices_grid.png")


# JSON serialisation helper.

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert dataclasses, dicts, lists and numpy scalars into
    JSON-friendly Python types.

    INPUTS:
        * obj  : any Python object that may contain dataclasses or numpy
                 scalars

    OUTPUTS:
        * JSON-friendly equivalent.
    """
    if hasattr(obj, "__dataclass_fields__"):
        result_dict = {}
        for field_name, field_value in asdict(obj).items():
            result_dict[field_name] = to_jsonable(field_value)
        return result_dict
    if isinstance(obj, dict):
        result_dict = {}
        for dict_key, dict_value in obj.items():
            result_dict[str(dict_key)] = to_jsonable(dict_value)
        return result_dict
    if isinstance(obj, (list, tuple)):
        result_list = []
        for list_item in obj:
            result_list.append(to_jsonable(list_item))
        return result_list
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.round(4).to_dict()
    return obj


# Plot styling.

def configure_plot_style() -> None:
    """
    Apply the project-wide matplotlib style: seaborn whitegrid, global
    font sizes, and a red/green/blue colour cycler so that pandas .plot()
    calls pick up the project palette by default.

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
        "axes.prop_cycle":  mpl.cycler(color = [
            PRIMARY_RED, PRIMARY_GREEN, PRIMARY_BLUE,
            ACCENT_RED, ACCENT_GREEN, ACCENT_BLUE,
        ]),
    })


# Orchestration.

def run_pipeline(start_date: str, end_date: str) -> PipelineSummary:
    """
    Execute the end-to-end pipeline: fetch the price and volume panel,
    build daily and rolling returns, compute SK Hynix realised volatility,
    construct the four sub-sample masks, compute conditional gold
    statistics, the rolling conditional correlation, and the regime-
    filtered correlation matrices, render every figure and persist every
    table.

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

    panel = fetch_panel(start_date, end_date)
    prices = panel["prices"]
    volume = panel["volume"]
    save_table(prices, DATA_PROCESSED_DIR, "prices")
    save_table(volume, DATA_PROCESSED_DIR, "volume")

    daily_returns = to_log_returns(prices)
    four_week = rolling_four_week_return(prices, TARGET_NAME)
    realised_vol = rolling_realised_volatility(daily_returns, TARGET_NAME)
    save_table(daily_returns, DATA_PROCESSED_DIR, "log_returns")
    save_table(four_week.to_frame(), DATA_PROCESSED_DIR, "four_week_returns")
    save_table(realised_vol.to_frame(), DATA_PROCESSED_DIR, "realised_volatility")

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(prices.index.min().date()),
        end = str(prices.index.max().date()),
        n_obs = int(len(daily_returns)),
    )
    summary.notes.append(
        "Realised-volatility-only deep dive on SK Hynix. The headline regime "
        "uses 4-week return at or below the 10th percentile AND 21d realised "
        "vol at or above the 75th percentile."
    )

    log.info("Generating context plots")
    plot_prices_and_realised_vol(prices, realised_vol)
    plot_realised_vol_regime(realised_vol)
    plot_realised_volatility(realised_vol)
    plot_four_week_return_distribution(four_week)
    if not volume.empty:
        plot_target_volume(volume)

    log.info("Building sub-sample masks and conditional statistics")
    subsample_masks = build_regime_masks(four_week, realised_vol)
    for subsample_label in SUBSAMPLE_LABELS:
        mask = subsample_masks[subsample_label]
        summary.subsample_stats.append(
            subsample_gold_stats(daily_returns, mask, subsample_label)
        )

    log.info("Plotting conditional gold comparison")
    plot_conditional_gold_comparison(summary.subsample_stats)

    log.info("Computing rolling 252d conditional correlation")
    regime_mask = subsample_masks[SUBSAMPLE_REGIME]
    rolling_corr_series = rolling_conditional_correlation(
        daily_returns, regime_mask, ROLLING_CORR_WINDOW,
    )
    save_table(
        rolling_corr_series.to_frame(name = "corr"),
        DATA_PROCESSED_DIR,
        "rolling_corr_regime",
    )
    plot_rolling_conditional_correlation(rolling_corr_series)

    log.info("Computing correlation matrices: full sample and headline regime")
    matrices_by_subsample_and_method: Dict[Tuple[str, str], pd.DataFrame] = {}
    for subsample_label in [SUBSAMPLE_FULL, SUBSAMPLE_REGIME]:
        mask = subsample_masks[subsample_label]
        for method in ["pearson", "spearman"]:
            matrix = subsample_correlation_matrix(daily_returns, mask, method)
            matrices_by_subsample_and_method[(subsample_label, method)] = matrix
            save_table(
                matrix,
                DATA_PROCESSED_DIR,
                f"corr_matrix_{method}_{subsample_label}",
            )
    plot_correlation_matrices_grid(matrices_by_subsample_and_method)

    matrices_json = {}
    for (subsample_label, method), matrix in matrices_by_subsample_and_method.items():
        matrices_json[f"{subsample_label}__{method}"] = matrix.round(4).to_dict()
    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_payload = to_jsonable(summary)
    summary_payload["correlation_matrices"] = matrices_json
    summary_path.write_text(json.dumps(summary_payload, indent = 2))
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
    Entry point. Runs the full pipeline and prints a concise summary table.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: figures in VISUALS_DIR, Parquet tables in
          DATA_PROCESSED_DIR, raw caches in DATA_RAW_DIR, JSON summary in
          DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\nPipeline summary")
    print(f"Window  : {summary.start} -> {summary.end}  (n = {summary.n_obs})")
    print("\nSub-sample gold statistics:")
    header = (
        f"  {'subsample':22s}  {'n':>5s}  "
        f"{'gold_mu_bps':>11s}  {'gold_med_bps':>12s}  "
        f"{'hit_rate':>8s}  {'rho_p':>6s}  {'rho_s':>6s}"
    )
    print(header)
    for stat in summary.subsample_stats:
        print(
            f"  {stat.subsample:22s}  {stat.n_obs:5d}  "
            f"{stat.gold_mean_bps:+11.2f}  {stat.gold_median_bps:+12.2f}  "
            f"{stat.gold_hit_rate_positive:8.3f}  "
            f"{stat.corr_pearson:+6.3f}  {stat.corr_spearman:+6.3f}"
        )
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")
    print(f"Raw in     : {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
