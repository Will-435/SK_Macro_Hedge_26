"""
Gold-time-series-analysis.py

Conditional gold-performance analysis against South Korean semiconductor
producers (Samsung Electronics, SK Hynix) and the global semiconductor ETF
(SOXX), focusing on negative-volatile regimes.

Research question:
    When semiconductor equities suffer left-tail 4-week price moves, AND when
    volatility regimes are elevated, how does gold behave? Is it a diversifier,
    a hedge, or does it co-move with the stress?

Volatility proxies:
    Historical per-stock implied volatility is not available on free data
    sources (yfinance returns only current option snapshots). This pipeline
    therefore uses two complementary proxies, run in parallel for every
    IV-dependent output:
        1. CBOE VIX (ticker ^VIX), the global equity-vol regime indicator.
        2. Stock-specific 21-day rolling realised volatility, a local IV proxy.

    Every IV-dependent output is produced in three variants:
        a) NO_VOL_FILTER  : left-quartile 4w semi return only.
        b) HIGH_VIX       : left-quartile 4w semi return AND VIX in top quartile.
        c) HIGH_RV        : left-quartile 4w semi return AND the stock's own
                            21-day realised vol in its top quartile.

Pipeline stages:
    1. Data acquisition (yfinance) for prices, trading volume and VIX. Raw
       responses are cached as Parquet under data/raw/.
    2. Daily log returns and 4-week (20 trading day) rolling returns.
    3. Stock-specific 21-day annualised realised volatility.
    4. Regime masks: left-quartile 4w semi returns, high VIX, high realised vol.
    5. Conditional gold-performance summary per regime per semi.
    6. Rolling 252-day conditional correlation between gold and each semi.
    7. Full-period regime-filtered correlation matrices (Pearson and Spearman).
    8. Context plots: prices+VIX, VIX regime, realised vol, 4w return
       distributions, trading volume.

Run:
    python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01
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

# Price tickers (friendly name to Yahoo Finance symbol).
PRICE_TICKERS: Dict[str, str] = {
    "gold":    "GC=F",
    "samsung": "005930.KS",
    "skhynix": "000660.KS",
    "soxx":    "SOXX",
}

# Tickers for which trading volume is recorded. Gold is a future, so volume is
# omitted there.
VOLUME_TICKERS: List[str] = ["samsung", "skhynix", "soxx"]

# Semi tickers used in the conditional analysis.
SEMI_TICKERS: List[str] = ["samsung", "skhynix", "soxx"]

# VIX as the global vol-regime proxy.
VIX_TICKER = "^VIX"

# Calendar and window sizes.
TRADING_DAYS_PER_YEAR = 252
FOUR_WEEK_TRADING_DAYS = 20
REALISED_VOL_WINDOW = 21
ROLLING_CORR_WINDOW = 252
VOLUME_AVG_WINDOW = 20

# Quantile thresholds.
LEFT_QUARTILE_THRESHOLD = 0.25
HIGH_VOL_QUANTILE = 0.75

# Minimum number of regime-filtered observations required to compute a
# rolling correlation point. Below this the window is skipped.
MIN_OBS_FOR_ROLLING_CORR = 15

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Scaling factors.
BPS_SCALAR = 1.0e4

# Regime labels (used as suffixes on output filenames).
REGIME_NO_FILTER = "no_vol_filter"
REGIME_HIGH_VIX = "high_vix"
REGIME_HIGH_RV = "high_realised_vol"
REGIME_LABELS: List[str] = [REGIME_NO_FILTER, REGIME_HIGH_VIX, REGIME_HIGH_RV]
REGIME_DISPLAY: Dict[str, str] = {
    REGIME_NO_FILTER: "Left-quartile 4w return only",
    REGIME_HIGH_VIX:  "Left-quartile 4w return + high VIX",
    REGIME_HIGH_RV:   "Left-quartile 4w return + high realised vol",
}

# Plotting: global font sizes. Edit these to globally change plot text.
TITLE_FONT = 14
TEXT_FONT = 10
HEATMAP_ANNOT_SIZE = 9

# Plotting: figure sizes.
WIDE_FIGSIZE = (12, 5)
TALL_WIDE_FIGSIZE = (12, 5.5)
BAR_FIGSIZE = (8, 5)
HISTOGRAM_PANEL_WIDTH_PER_TICKER = 5
HISTOGRAM_PANEL_HEIGHT = 4
MATRIX_GRID_WIDTH = 11
MATRIX_GRID_ROW_HEIGHT = 4.2

# Plotting: colours. Unconventional red, green and blue tones, used in priority
# order across each figure. Matplotlib's `color=` keyword is part of the
# library API, so the British spelling 'colour' is only used in our own
# identifiers.
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

# Per-ticker colour assignments (deliberately red, green and blue).
SEMI_COLOURS: Dict[str, str] = {
    "samsung": PRIMARY_BLUE,
    "skhynix": PRIMARY_RED,
    "soxx":    PRIMARY_GREEN,
}
GOLD_COLOUR = PRIMARY_RED
VIX_LINE_COLOUR = PRIMARY_BLUE

# Plotting: line widths, alphas, sizes.
REFERENCE_AXIS_LINEWIDTH = 0.5
VIX_LINE_LINEWIDTH = 0.7
VIX_REGIME_LINEWIDTH = 0.8
THRESHOLD_LINEWIDTH = 0.8
VIX_OVERLAY_ALPHA = 0.35
REGIME_FILL_ALPHA = 0.30
HIT_RATE_REFERENCE_ALPHA = 0.4
HISTOGRAM_BINS = 60
BAR_WIDTH = 0.4
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

# Filesystem layout. One sub-directory per script keeps each pipeline's
# outputs cleanly separated from any other pipeline in this repository.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "gold_time_series_analysis"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / SCRIPT_SUBDIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" / SCRIPT_SUBDIR
VISUALS_DIR = PROJECT_ROOT / "visuals" / SCRIPT_SUBDIR

# CLI defaults.
DEFAULT_START_DATE = "2015-01-01"


# Result containers (dataclasses).

@dataclass
class ConditionalGoldStats:
    """
    Summary of gold's behaviour under a single regime for a single semi
    ticker; one record per (ticker, regime) pair.

    INPUTS:
        * ticker
        * regime
        * n_obs
        * gold_mean_bps
        * gold_median_bps
        * gold_std_bps
        * gold_hit_rate_positive
        * semi_mean_bps
        * corr_pearson
        * corr_spearman

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    ticker: str
    regime: str
    n_obs: int
    gold_mean_bps: float
    gold_median_bps: float
    gold_std_bps: float
    gold_hit_rate_positive: float
    semi_mean_bps: float
    corr_pearson: float
    corr_spearman: float


@dataclass
class PipelineSummary:
    """
    Top-level run summary; persisted as JSON in the processed-data directory.

    INPUTS:
        * run_at
        * start
        * end
        * n_obs
        * notes
        * conditional_stats

    OUTPUTS:
        * Dataclass instance aggregating the above into one record.
    """
    run_at: str
    start: str
    end: str
    n_obs: int
    notes: List[str] = field(default_factory = list)
    conditional_stats: List[ConditionalGoldStats] = field(default_factory = list)


# Persistence helpers.

def save_table(frame: pd.DataFrame | pd.Series, directory: Path, name_stem: str) -> Path:
    """
    Persist a tabular object as Parquet, the preferred on-disk format for the
    pipeline. Series inputs are coerced to a single-column DataFrame so that
    the Parquet schema is well defined.

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

def _download_one(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download one Yahoo Finance ticker and flatten any MultiIndex columns that
    newer yfinance versions return.

    INPUTS:
        * symbol      : Yahoo Finance ticker symbol
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * DataFrame of OHLCV columns indexed by date.
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
    Build the aligned price, volume and VIX panel used by the rest of the
    pipeline. Raw downloaded frames are cached as Parquet under data/raw/ so
    that subsequent runs can inspect the un-processed inputs.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Dict with keys 'prices' (DataFrame), 'volumes' (DataFrame),
          'vix' (Series), all aligned on a forward-filled common trading
          calendar.
    """
    log.info("Downloading %d price tickers plus VIX", len(PRICE_TICKERS))
    price_frames: List[pd.Series] = []
    volume_frames: List[pd.Series] = []

    for friendly_name, yahoo_symbol in PRICE_TICKERS.items():
        raw_frame = _download_one(yahoo_symbol, start_date, end_date)
        save_table(raw_frame, DATA_RAW_DIR, f"raw_{friendly_name}")

        close_col = "Close" if "Close" in raw_frame.columns else raw_frame.columns[0]
        price_series = raw_frame[close_col]
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        price_frames.append(price_series.rename(friendly_name))

        if friendly_name in VOLUME_TICKERS and "Volume" in raw_frame.columns:
            volume_series = raw_frame["Volume"]
            if isinstance(volume_series, pd.DataFrame):
                volume_series = volume_series.iloc[:, 0]
            volume_frames.append(volume_series.rename(friendly_name))

    raw_vix = _download_one(VIX_TICKER, start_date, end_date)
    save_table(raw_vix, DATA_RAW_DIR, "raw_vix")
    vix_close_col = "Close" if "Close" in raw_vix.columns else raw_vix.columns[0]
    vix_series = raw_vix[vix_close_col]
    if isinstance(vix_series, pd.DataFrame):
        vix_series = vix_series.iloc[:, 0]
    vix_series = vix_series.rename("vix")

    prices = pd.concat(price_frames, axis = 1).dropna(how = "all")
    prices = prices.ffill(limit = PANEL_FFILL_LIMIT).dropna()
    volumes = pd.concat(volume_frames, axis = 1).reindex(prices.index).ffill()
    vix_aligned = vix_series.reindex(prices.index).ffill()

    log.info(
        "Aligned panel: %d rows (%s to %s)",
        prices.shape[0],
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return {"prices": prices, "volumes": volumes, "vix": vix_aligned}


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
    returns.columns = [f"r_{col_name}" for col_name in returns.columns]
    return returns


def rolling_four_week_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the rolling 4-week (20 trading day) log return per series.

    INPUTS:
        * prices  : DataFrame of adjusted close prices

    OUTPUTS:
        * DataFrame of rolling 4-week log returns; one column per input
          column, prefixed 'r4w_'.
    """
    log_prices = np.log(prices)
    four_w = log_prices - log_prices.shift(FOUR_WEEK_TRADING_DAYS)
    four_w.columns = [f"r4w_{col_name}" for col_name in four_w.columns]
    return four_w.dropna(how = "all")


def rolling_realised_vol(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised rolling realised volatility per series. Used as the
    stock-specific implied-volatility proxy.

    INPUTS:
        * returns  : DataFrame of daily log returns (columns prefixed 'r_')

    OUTPUTS:
        * DataFrame of annualised realised vol; columns prefixed 'rv_'.
    """
    rv_frame = returns.rolling(REALISED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rv_frame.columns = [f"rv_{col_name[2:]}" for col_name in returns.columns]
    return rv_frame


# Regime masks.

def left_quartile_mask(four_week: pd.Series) -> pd.Series:
    """
    Identify days where the 4-week return is in its bottom quartile.

    INPUTS:
        * four_week  : 4-week rolling return Series for one ticker

    OUTPUTS:
        * Boolean Series aligned to the input index; True on regime days.
    """
    threshold = four_week.quantile(LEFT_QUARTILE_THRESHOLD)
    return four_week <= threshold


def high_vix_mask(vix: pd.Series) -> pd.Series:
    """
    Identify days where VIX is in its top quartile.

    INPUTS:
        * vix  : VIX Series

    OUTPUTS:
        * Boolean Series aligned to the input index; True on regime days.
    """
    threshold = vix.quantile(HIGH_VOL_QUANTILE)
    return vix >= threshold


def high_realised_vol_mask(realised_vol: pd.Series) -> pd.Series:
    """
    Identify days where the ticker's own realised volatility is in its top
    quartile.

    INPUTS:
        * realised_vol  : annualised realised-vol Series for one ticker

    OUTPUTS:
        * Boolean Series aligned to the input index; True on regime days.
    """
    threshold = realised_vol.quantile(HIGH_VOL_QUANTILE)
    return realised_vol >= threshold


def build_regime_masks(
    four_week: pd.Series,
    vix: pd.Series,
    realised_vol_series: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Build the three regime masks for one semi ticker by conjoining the
    left-quartile 4w return mask with each of the volatility filters.

    INPUTS:
        * four_week            : 4-week return Series for one semi
        * vix                  : VIX Series
        * realised_vol_series  : 21d realised vol Series for the same semi

    OUTPUTS:
        * Dict {regime_label: boolean Series} keyed by REGIME_LABELS, aligned
          on the intersection of the three input indices.
    """
    common_index = (
        four_week.dropna().index
        .intersection(vix.dropna().index)
        .intersection(realised_vol_series.dropna().index)
    )
    fw_aligned = four_week.reindex(common_index)
    vix_aligned = vix.reindex(common_index)
    rv_aligned = realised_vol_series.reindex(common_index)

    left_quartile = left_quartile_mask(fw_aligned)
    vix_top = high_vix_mask(vix_aligned)
    rv_top = high_realised_vol_mask(rv_aligned)

    return {
        REGIME_NO_FILTER: left_quartile,
        REGIME_HIGH_VIX:  left_quartile & vix_top,
        REGIME_HIGH_RV:   left_quartile & rv_top,
    }


# Conditional gold statistics.

def conditional_gold_stats(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    ticker: str,
    regime_label: str,
) -> ConditionalGoldStats:
    """
    Compute the headline conditional gold statistics restricted to regime
    days for one (ticker, regime) pair. Returns a fully-populated dataclass
    even when no regime days exist (the fields become NaN), so that the
    downstream summary table is always rectangular.

    INPUTS:
        * daily_returns  : DataFrame of daily log returns (cols prefixed 'r_')
        * mask           : boolean Series indexed by date
        * ticker         : semi ticker name (column is 'r_<ticker>')
        * regime_label   : one of REGIME_LABELS

    OUTPUTS:
        * Populated ConditionalGoldStats record.
    """
    aligned_mask = mask.reindex(daily_returns.index).fillna(False).astype(bool)
    regime_returns = daily_returns.loc[aligned_mask]
    n_regime_obs = int(len(regime_returns))

    if n_regime_obs == 0:
        nan = float("nan")
        return ConditionalGoldStats(
            ticker = ticker, regime = regime_label, n_obs = 0,
            gold_mean_bps = nan, gold_median_bps = nan, gold_std_bps = nan,
            gold_hit_rate_positive = nan,
            semi_mean_bps = nan,
            corr_pearson = nan, corr_spearman = nan,
        )

    gold_returns = regime_returns["r_gold"]
    semi_returns = regime_returns[f"r_{ticker}"]
    return ConditionalGoldStats(
        ticker = ticker,
        regime = regime_label,
        n_obs = n_regime_obs,
        gold_mean_bps = float(gold_returns.mean() * BPS_SCALAR),
        gold_median_bps = float(gold_returns.median() * BPS_SCALAR),
        gold_std_bps = float(gold_returns.std() * BPS_SCALAR),
        gold_hit_rate_positive = float((gold_returns > 0).mean()),
        semi_mean_bps = float(semi_returns.mean() * BPS_SCALAR),
        corr_pearson = float(gold_returns.corr(semi_returns, method = "pearson")),
        corr_spearman = float(gold_returns.corr(semi_returns, method = "spearman")),
    )


# Rolling conditional correlation.

def rolling_conditional_correlation(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    ticker: str,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.Series:
    """
    Within each rolling window, restrict to the days where the regime mask
    is True and compute Pearson correlation between r_gold and r_<ticker>.
    Windows with too few regime days are skipped, which produces a sparse
    time series rather than NaN-filled output.

    INPUTS:
        * daily_returns  : DataFrame with r_gold and r_<ticker>
        * mask           : boolean Series indexed by date
        * ticker         : semi ticker name
        * window         : rolling window length in trading days

    OUTPUTS:
        * Series of correlations indexed by window end-date.
    """
    aligned_mask = (
        mask.reindex(daily_returns.index).fillna(False).astype(bool).values
    )
    gold_arr = daily_returns["r_gold"].values
    semi_arr = daily_returns[f"r_{ticker}"].values
    date_index = daily_returns.index

    rolling_points: List[Tuple[pd.Timestamp, float]] = []
    n_total = len(daily_returns)
    for window_end in range(window, n_total):
        slice_mask = aligned_mask[window_end - window:window_end]
        if slice_mask.sum() < MIN_OBS_FOR_ROLLING_CORR:
            continue
        gold_slice = gold_arr[window_end - window:window_end][slice_mask]
        semi_slice = semi_arr[window_end - window:window_end][slice_mask]
        if gold_slice.std() == 0 or semi_slice.std() == 0:
            continue
        corr_val = float(np.corrcoef(gold_slice, semi_slice)[0, 1])
        rolling_points.append((date_index[window_end], corr_val))

    if not rolling_points:
        return pd.Series(dtype = float)
    return pd.Series(dict(rolling_points))


# Regime-filtered correlation matrices.

def regime_filtered_correlation_matrix(
    daily_returns: pd.DataFrame,
    regime_masks_by_ticker: Dict[str, pd.Series],
    method: str,
) -> pd.DataFrame:
    """
    Build a correlation matrix over [r_gold, r_<each semi>] restricted to
    the union of the per-ticker regime masks. The union is used so that the
    matrix has a single internally-coherent sample, rather than mixing per-
    cell samples.

    INPUTS:
        * daily_returns           : DataFrame of daily log returns
        * regime_masks_by_ticker  : one boolean mask per semi ticker
        * method                  : 'pearson' or 'spearman'

    OUTPUTS:
        * Correlation matrix DataFrame, or an empty DataFrame if no masks
          were supplied.
    """
    union_mask = None
    for mask_series in regime_masks_by_ticker.values():
        aligned = mask_series.reindex(daily_returns.index).fillna(False).astype(bool)
        union_mask = aligned if union_mask is None else (union_mask | aligned)
    if union_mask is None:
        return pd.DataFrame()
    regime_returns = daily_returns.loc[union_mask]
    cols = ["r_gold"] + [f"r_{ticker_name}" for ticker_name in SEMI_TICKERS]
    return regime_returns[cols].corr(method = method)


# Plotting helpers.

def _add_caption(fig: plt.Figure, caption_text: str) -> None:
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


def _save_figure(fig: plt.Figure, file_name: str) -> Path:
    """
    Save a matplotlib figure into VISUALS_DIR and close it.

    INPUTS:
        * fig        : matplotlib Figure
        * file_name  : file name (with extension)

    OUTPUTS:
        * Path to the written file.
    """
    output_path = VISUALS_DIR / file_name
    fig.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(fig)
    return output_path


def plot_price_panel(prices: pd.DataFrame, vix: pd.Series) -> Path:
    """
    Render normalised prices for all four assets on the left axis and the
    VIX on a secondary right axis. Provides a one-glance overview of the
    sample.

    INPUTS:
        * prices  : DataFrame of adjusted close prices
        * vix     : VIX Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax_left = plt.subplots(figsize = TALL_WIDE_FIGSIZE)
    normalised = prices / prices.iloc[0] * 100
    # Plot each column in our project palette explicitly so that the colours
    # match the rest of the figures.
    palette_for_prices = {
        "gold":    GOLD_COLOUR,
        "samsung": SEMI_COLOURS["samsung"],
        "skhynix": SEMI_COLOURS["skhynix"],
        "soxx":    SEMI_COLOURS["soxx"],
    }
    for asset_name in normalised.columns:
        ax_left.plot(
            normalised.index, normalised[asset_name],
            color = palette_for_prices.get(asset_name, NEUTRAL_DARK),
            label = asset_name,
        )
    ax_left.set_ylabel("Price index (start = 100)")
    ax_left.set_title("Normalised prices and VIX")
    ax_left.legend(loc = "upper left")
    ax_right = ax_left.twinx()
    ax_right.plot(
        vix.index, vix.values,
        color = NEUTRAL_GREY, alpha = VIX_OVERLAY_ALPHA,
        lw = VIX_LINE_LINEWIDTH, label = "VIX",
    )
    ax_right.set_ylabel("VIX")
    ax_right.legend(loc = "upper right")
    fig.tight_layout()
    _add_caption(
        fig,
        "Daily close prices for gold (GC=F), Samsung (005930.KS), "
        "SK Hynix (000660.KS) and SOXX, rebased so each series starts at 100. "
        "Grey line on the right axis is the CBOE VIX. "
        "VIX = CBOE Volatility Index, the 30-day implied vol of S&P 500 options.",
    )
    return _save_figure(fig, "prices_and_vix.png")


def plot_vix_regime(vix: pd.Series) -> Path:
    """
    Render the VIX series with the top-quartile high-vol regime shaded.

    INPUTS:
        * vix  : VIX Series

    OUTPUTS:
        * Path to the saved PNG.
    """
    threshold = vix.quantile(HIGH_VOL_QUANTILE)
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    ax.plot(vix.index, vix.values, color = PRIMARY_BLUE, lw = VIX_REGIME_LINEWIDTH)
    ax.fill_between(
        vix.index, vix.values, threshold,
        where = (vix.values >= threshold),
        color = PRIMARY_RED, alpha = REGIME_FILL_ALPHA,
        label = "VIX in top quartile",
    )
    ax.axhline(
        threshold, color = PRIMARY_RED, lw = THRESHOLD_LINEWIDTH,
        linestyle = "--", label = f"75th pctile = {threshold:.1f}",
    )
    ax.set_title("VIX with high-volatility regime highlighted")
    ax.set_ylabel("VIX")
    ax.legend()
    fig.tight_layout()
    _add_caption(
        fig,
        "Blue line: VIX daily close over the sample. "
        "Shaded red region: days where VIX is at or above its full-sample "
        "75th percentile (the 'high VIX' regime used as one of the IV-proxy "
        "filters downstream). "
        "VIX = CBOE Volatility Index.",
    )
    return _save_figure(fig, "vix_regime.png")


def plot_realised_volatility(realised_vol_df: pd.DataFrame) -> Path:
    """
    Render rolling annualised realised volatility, one line per semi ticker
    in the per-ticker red/green/blue palette.

    INPUTS:
        * realised_vol_df  : DataFrame of realised vols (cols prefixed 'rv_')

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    for ticker_name in SEMI_TICKERS:
        col_name = f"rv_{ticker_name}"
        ax.plot(
            realised_vol_df.index, realised_vol_df[col_name],
            color = SEMI_COLOURS[ticker_name], label = ticker_name,
        )
    ax.set_title(f"{REALISED_VOL_WINDOW}d annualised realised volatility")
    ax.set_ylabel("Annualised vol")
    ax.legend()
    fig.tight_layout()
    _add_caption(
        fig,
        "21-day rolling annualised realised volatility for each semi ticker. "
        "RV_t = std(r_{t-20}, ..., r_t) x sqrt(252), where r_t is the daily "
        "log return. This series is used as the stock-specific IV proxy in "
        "the 'high realised vol' regime filter.",
    )
    return _save_figure(fig, "realised_volatility.png")


def plot_four_week_return_distributions(four_week: pd.DataFrame) -> Path:
    """
    Render the 4-week return distribution per semi as a histogram with KDE
    overlay and the left-quartile threshold marked.

    INPUTS:
        * four_week  : DataFrame of 4-week returns (cols prefixed 'r4w_')

    OUTPUTS:
        * Path to the saved PNG.
    """
    semi_cols = [f"r4w_{ticker_name}" for ticker_name in SEMI_TICKERS]
    fig_width = HISTOGRAM_PANEL_WIDTH_PER_TICKER * len(semi_cols)
    fig, axes = plt.subplots(1, len(semi_cols), figsize = (fig_width, HISTOGRAM_PANEL_HEIGHT))
    for ax, ticker_name, col_name in zip(axes, SEMI_TICKERS, semi_cols):
        series = four_week[col_name].dropna()
        threshold = series.quantile(LEFT_QUARTILE_THRESHOLD)
        sns.histplot(
            series, bins = HISTOGRAM_BINS, kde = True,
            ax = ax, color = SEMI_COLOURS[ticker_name],
        )
        ax.axvline(
            threshold, color = PRIMARY_RED, linestyle = "--",
            label = f"Q1 = {threshold:+.3f}",
        )
        ax.set_title(col_name)
        ax.legend()
    fig.suptitle("4-week semi return distributions", fontsize = TITLE_FONT)
    fig.tight_layout()
    _add_caption(
        fig,
        "Histogram of the 4-week (20 trading day) rolling log return for "
        "each semi. r4w_X = log(P_X_t / P_X_{t-20}). The dashed red line "
        "marks Q1, the 25th percentile of that distribution, used as the "
        "left-quartile threshold that defines the negative-return regime.",
    )
    return _save_figure(fig, "four_week_return_distributions.png")


def plot_volume(volumes: pd.DataFrame) -> Path:
    """
    Render the 20-day rolling mean daily trading volume per semi.

    INPUTS:
        * volumes  : DataFrame of daily trading volumes

    OUTPUTS:
        * Path to the saved PNG.
    """
    smoothed = volumes.rolling(VOLUME_AVG_WINDOW).mean()
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    for ticker_name in SEMI_TICKERS:
        if ticker_name not in smoothed.columns:
            continue
        ax.plot(
            smoothed.index, smoothed[ticker_name],
            color = SEMI_COLOURS[ticker_name], label = ticker_name,
        )
    ax.set_title(f"{VOLUME_AVG_WINDOW}d rolling mean trading volume")
    ax.set_ylabel("Shares")
    ax.legend()
    fig.tight_layout()
    _add_caption(
        fig,
        "20-day rolling mean of daily trading volume (shares) per semi "
        "ticker. Provided as background context only; trading volume is "
        "not part of the regime-filter logic.",
    )
    return _save_figure(fig, "trading_volume.png")


def plot_conditional_gold_bars(
    stats_list: List[ConditionalGoldStats],
    regime_label: str,
) -> Path:
    """
    Render a bar chart of gold's mean daily return on regime days (left
    axis) plus gold's hit-rate-positive on regime days (right-axis line),
    one regime per figure.

    INPUTS:
        * stats_list    : list of all ConditionalGoldStats records
        * regime_label  : one of REGIME_LABELS; the regime to plot

    OUTPUTS:
        * Path to the saved PNG.
    """
    filtered = [stat for stat in stats_list if stat.regime == regime_label]
    fig, ax_left = plt.subplots(figsize = BAR_FIGSIZE)
    tickers = [stat.ticker for stat in filtered]
    means = [stat.gold_mean_bps for stat in filtered]
    hit_rates = [stat.gold_hit_rate_positive for stat in filtered]
    n_obs_list = [stat.n_obs for stat in filtered]

    bar_positions = np.arange(len(tickers))
    bar_colours = [SEMI_COLOURS[ticker_name] for ticker_name in tickers]
    ax_left.bar(
        bar_positions, means,
        color = bar_colours, label = "Gold mean (bps)",
    )
    ax_left.set_xticks(bar_positions)
    ax_left.set_xticklabels(
        [f"{ticker_name}\n(n = {n_regime})"
         for ticker_name, n_regime in zip(tickers, n_obs_list)]
    )
    ax_left.axhline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax_left.set_ylabel("Mean daily gold return (bps)")

    ax_right = ax_left.twinx()
    ax_right.plot(
        bar_positions, hit_rates,
        color = ACCENT_RED, marker = "o",
        label = "Gold hit-rate positive",
    )
    ax_right.set_ylabel("Fraction of regime days with r_gold > 0")
    ax_right.set_ylim(HIT_RATE_AXIS_MIN, HIT_RATE_AXIS_MAX)
    ax_right.axhline(
        HIT_RATE_REFERENCE_VALUE,
        color = ACCENT_RED, linestyle = ":",
        alpha = HIT_RATE_REFERENCE_ALPHA,
    )

    ax_left.set_title(
        f"Conditional gold performance\n[{REGIME_DISPLAY[regime_label]}]"
    )
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc = "upper left",
    )
    fig.tight_layout()
    _add_caption(
        fig,
        "For each semi ticker, gold's mean daily log return on regime days "
        "(bars, left axis, basis points) and the share of regime days on "
        "which gold closed up (red line, right axis). bps = basis points = "
        "1e-4 of log return. n = number of days that satisfy the regime "
        "mask. The dotted red line marks a 50% hit rate. "
        "Regime: " + REGIME_DISPLAY[regime_label] + ".",
    )
    return _save_figure(fig, f"conditional_gold_{regime_label}.png")


def plot_rolling_conditional_correlation(
    rolling_by_ticker: Dict[str, pd.Series],
    regime_label: str,
) -> Path:
    """
    Render the rolling 252-day conditional correlation between gold and each
    semi, one regime per figure.

    INPUTS:
        * rolling_by_ticker  : dict of {ticker_name: rolling correlation Series}
        * regime_label       : one of REGIME_LABELS

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    for ticker_name, corr_series in rolling_by_ticker.items():
        if corr_series.empty:
            continue
        ax.plot(
            corr_series.index, corr_series.values,
            color = SEMI_COLOURS[ticker_name],
            label = f"{ticker_name} vs gold",
        )
    ax.axhline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_title(
        f"Rolling {ROLLING_CORR_WINDOW}d corr (gold vs semi) | "
        f"{REGIME_DISPLAY[regime_label]}"
    )
    ax.set_ylabel("Conditional rho")
    ax.legend()
    fig.tight_layout()
    _add_caption(
        fig,
        "Rolling 252-day Pearson correlation between r_gold and r_<semi>, "
        "restricted within each window to days that satisfy the regime mask. "
        "rho = Pearson correlation coefficient. r_X = daily log return of "
        "asset X. Windows with fewer than 15 regime days are skipped. "
        "Regime: " + REGIME_DISPLAY[regime_label] + ".",
    )
    return _save_figure(fig, f"rolling_corr_{regime_label}.png")


def plot_correlation_matrices_grid(
    matrices_by_regime_and_method: Dict[Tuple[str, str], pd.DataFrame],
) -> Path:
    """
    Render a grid of regime-filtered correlation matrices. Rows: regime
    variants. Columns: Pearson and Spearman.

    INPUTS:
        * matrices_by_regime_and_method  : dict keyed by (regime_label, method)

    OUTPUTS:
        * Path to the saved PNG.
    """
    grid_height = MATRIX_GRID_ROW_HEIGHT * len(REGIME_LABELS)
    fig, axes = plt.subplots(
        len(REGIME_LABELS), 2,
        figsize = (MATRIX_GRID_WIDTH, grid_height),
    )
    for row_idx, regime_label in enumerate(REGIME_LABELS):
        for col_idx, method in enumerate(["pearson", "spearman"]):
            ax = axes[row_idx, col_idx]
            matrix = matrices_by_regime_and_method.get((regime_label, method))
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
                f"{method.capitalize()} | {REGIME_DISPLAY[regime_label]}",
                fontsize = TEXT_FONT,
            )
    fig.suptitle(
        "Regime-filtered correlation matrices (rows = regime, cols = method)",
        fontsize = TITLE_FONT,
    )
    fig.tight_layout()
    _add_caption(
        fig,
        "Correlation matrices over [r_gold, r_samsung, r_skhynix, r_soxx], "
        "computed only on days that satisfy the union of each regime's "
        "ticker-specific masks. Rows: regime variants. Columns: correlation "
        "method. Pearson = linear correlation; Spearman = rank-based "
        "correlation, robust to outliers. r_X = daily log return of asset X.",
    )
    return _save_figure(fig, "correlation_matrices_grid.png")


# JSON serialisation helper.

def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert dataclasses, dicts, lists and numpy scalars into
    JSON-friendly Python types.

    INPUTS:
        * obj  : any Python object that may contain dataclasses or numpy types

    OUTPUTS:
        * JSON-friendly equivalent (dict, list, float, int, str, ...).
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _to_jsonable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.round(4).to_dict()
    return obj


# Plot styling.

def configure_plot_style() -> None:
    """
    Apply the project-wide matplotlib style: seaborn whitegrid, global font
    sizes, and a red/green/blue colour cycler so that pandas .plot() calls
    pick up the project palette by default.

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
    Execute the end-to-end pipeline: fetch the data panel, build returns and
    rolling windows, construct the three regime masks per ticker, compute
    conditional statistics, the rolling conditional correlations and the
    regime-filtered correlation matrices, render every figure and persist
    every table.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Populated PipelineSummary, also persisted as JSON in
          DATA_PROCESSED_DIR. Visuals are written to VISUALS_DIR; processed
          tables to DATA_PROCESSED_DIR; raw downloads cached under
          DATA_RAW_DIR.
    """
    for required_dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALS_DIR):
        required_dir.mkdir(parents = True, exist_ok = True)
    configure_plot_style()

    panel = fetch_panel(start_date, end_date)
    prices = panel["prices"]
    volumes = panel["volumes"]
    vix = panel["vix"]

    save_table(prices, DATA_PROCESSED_DIR, "prices")
    save_table(volumes, DATA_PROCESSED_DIR, "volumes")
    save_table(vix, DATA_PROCESSED_DIR, "vix")

    daily_returns = to_log_returns(prices)
    four_week = rolling_four_week_returns(prices)
    realised_vol = rolling_realised_vol(daily_returns)
    save_table(daily_returns, DATA_PROCESSED_DIR, "log_returns")
    save_table(four_week, DATA_PROCESSED_DIR, "four_week_returns")
    save_table(realised_vol, DATA_PROCESSED_DIR, "realised_volatility")

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(prices.index.min().date()),
        end = str(prices.index.max().date()),
        n_obs = int(len(daily_returns)),
    )
    summary.notes.append(
        "Historical implied volatility per stock is not available on free "
        "data sources; used VIX (global) and 21d realised vol "
        "(stock-specific) as IV proxies."
    )

    log.info("Generating context plots")
    plot_price_panel(prices, vix)
    plot_vix_regime(vix)
    plot_realised_volatility(realised_vol)
    plot_four_week_return_distributions(four_week)
    if not volumes.empty:
        plot_volume(volumes)

    log.info("Building regime masks and conditional statistics")
    masks_by_ticker: Dict[str, Dict[str, pd.Series]] = {}
    for ticker_name in SEMI_TICKERS:
        fw_series = four_week[f"r4w_{ticker_name}"]
        rv_series = realised_vol[f"rv_{ticker_name}"]
        masks_by_ticker[ticker_name] = build_regime_masks(
            fw_series, vix, rv_series,
        )
        for regime_label, mask in masks_by_ticker[ticker_name].items():
            summary.conditional_stats.append(
                conditional_gold_stats(
                    daily_returns, mask, ticker_name, regime_label,
                )
            )

    log.info("Plotting conditional gold performance")
    for regime_label in REGIME_LABELS:
        plot_conditional_gold_bars(summary.conditional_stats, regime_label)

    log.info("Computing rolling 252d conditional correlations")
    for regime_label in REGIME_LABELS:
        rolling_by_ticker: Dict[str, pd.Series] = {}
        for ticker_name in SEMI_TICKERS:
            mask = masks_by_ticker[ticker_name][regime_label]
            rolling_series = rolling_conditional_correlation(
                daily_returns, mask, ticker_name, ROLLING_CORR_WINDOW,
            )
            rolling_by_ticker[ticker_name] = rolling_series
            save_table(
                rolling_series,
                DATA_PROCESSED_DIR,
                f"rolling_corr_{ticker_name}_{regime_label}",
            )
        plot_rolling_conditional_correlation(rolling_by_ticker, regime_label)

    log.info("Computing regime-filtered correlation matrices")
    matrices_by_regime_and_method: Dict[Tuple[str, str], pd.DataFrame] = {}
    for regime_label in REGIME_LABELS:
        per_ticker_masks = {
            ticker_name: masks_by_ticker[ticker_name][regime_label]
            for ticker_name in SEMI_TICKERS
        }
        for method in ["pearson", "spearman"]:
            matrix = regime_filtered_correlation_matrix(
                daily_returns, per_ticker_masks, method,
            )
            matrices_by_regime_and_method[(regime_label, method)] = matrix
            save_table(
                matrix,
                DATA_PROCESSED_DIR,
                f"corr_matrix_{method}_{regime_label}",
            )
    plot_correlation_matrices_grid(matrices_by_regime_and_method)

    matrices_json = {
        f"{regime_label}__{method}": matrix.round(4).to_dict()
        for (regime_label, method), matrix in matrices_by_regime_and_method.items()
    }
    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_payload = _to_jsonable(summary)
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


# Logging.

warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gold-vol-regime")


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
    print("\nConditional gold statistics:")
    header = (
        f"  {'ticker':8s}  {'regime':22s}  {'n':>5s}  "
        f"{'gold_mu_bps':>11s}  {'gold_med_bps':>12s}  "
        f"{'hit_rate':>8s}  {'rho_p':>6s}  {'rho_s':>6s}"
    )
    print(header)
    for stat in summary.conditional_stats:
        print(
            f"  {stat.ticker:8s}  {stat.regime:22s}  "
            f"{stat.n_obs:5d}  {stat.gold_mean_bps:+11.2f}  "
            f"{stat.gold_median_bps:+12.2f}  "
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
