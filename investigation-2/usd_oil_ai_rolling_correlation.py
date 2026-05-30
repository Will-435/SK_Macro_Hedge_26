"""
usd_oil_ai_rolling_correlation.py

Rolling-correlation comparison between the US Dollar Index, the price of oil,
and two complementary AI-equity proxies (an equal-weighted Mag 7 basket and
the Nasdaq-100 ETF QQQ).

The headline figure is a single line plot carrying three rolling-correlation
series, all measured against USD log returns:

    1. corr(USD, oil)            : the classic 'commodity-dollar' relationship.
    2. corr(USD, Mag 7 basket)   : a concentrated AI-equity proxy.
    3. corr(USD, QQQ)            : a broader large-cap tech proxy.

A second figure repeats the analysis on a shorter 63-day window to surface
faster dynamics that the 252-day chart smooths away. Two context plots are
also produced (normalised price panel and return-distribution KDEs).

Run:
    python usd_oil_ai_rolling_correlation.py --start 2015-01-01 --end 2026-05-01
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
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# Tickers (friendly name to Yahoo Finance symbol).
USD_NAME = "usd"
USD_SYMBOL = "DX-Y.NYB"           # ICE US Dollar Index
OIL_NAME = "oil"
OIL_SYMBOL = "CL=F"               # WTI Crude front-month future
QQQ_NAME = "qqq"
QQQ_SYMBOL = "QQQ"                # Invesco QQQ Trust (Nasdaq-100 ETF)
GOLD_NAME = "gold"
GOLD_SYMBOL = "GC=F"               # COMEX gold front-month future
MAG7_BASKET_NAME = "mag7"
MAG7_TICKERS: List[str] = [
    "NVDA", "MSFT", "GOOGL", "AAPL", "AMZN", "META", "TSLA",
]

# Asset display labels for legends and captions.
ASSET_DISPLAY: Dict[str, str] = {
    USD_NAME:          "US Dollar Index",
    OIL_NAME:          "WTI crude oil",
    QQQ_NAME:          "Nasdaq-100 (QQQ)",
    GOLD_NAME:         "Gold (GC=F)",
    MAG7_BASKET_NAME:  "Mag 7 equal-weighted basket",
}

# Trio used in the gold / USD / NASDAQ correlation matrices.
TRIO_ASSET_ORDER: List[str] = [GOLD_NAME, USD_NAME, QQQ_NAME]
TRIO_DISPLAY_LABELS: Dict[str, str] = {
    GOLD_NAME: "Gold",
    USD_NAME:  "USD",
    QQQ_NAME:  "NASDAQ",
}

# Calendar and window sizes.
TRADING_DAYS_PER_YEAR = 252
ROLLING_CORR_LONG_WINDOW = 252
ROLLING_CORR_SHORT_WINDOW = 63
MIN_PERIODS_LONG = 200
MIN_PERIODS_SHORT = 50

# Bearish regime parameters (re-used from the SK Hynix deep dive). Applied
# in this file to the Nasdaq-100 ETF (QQQ) as the equity stress proxy.
BEARISH_REGIME_TARGET = QQQ_NAME
FOUR_WEEK_TRADING_DAYS = 20
REALISED_VOL_WINDOW = 21
BEARISH_QUANTILE = 0.10
HIGH_VOL_QUANTILE = 0.75

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Synthetic basket base index value (so that the basket price series can be
# plotted on the same normalised axis as the other assets).
BASKET_BASE_INDEX_VALUE = 100.0

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: figure sizes.
WIDE_FIGSIZE = (12, 5)
TALL_WIDE_FIGSIZE = (12, 5.5)
DISTRIBUTION_PANEL_WIDTH_PER_SERIES = 4
DISTRIBUTION_PANEL_HEIGHT = 4
MATRIX_GRID_FIGSIZE = (11, 9)

# Heatmap configuration for the trio correlation matrices.
HEATMAP_CMAP = "RdBu_r"
HEATMAP_VMIN = -1.0
HEATMAP_VMAX = 1.0
HEATMAP_ANNOT_SIZE = 9

# Plotting: colours. Unconventional red, green and blue shades favoured by
# the project style guide. Matplotlib's `color` keyword stays in US spelling
# because that is the library API; our own identifiers use British spelling.
PRIMARY_RED = "#C1272D"
PRIMARY_GREEN = "#386641"
PRIMARY_BLUE = "#1F4E79"
ACCENT_RED = "#E63946"
ACCENT_GREEN = "#52B788"
ACCENT_BLUE = "#457B9D"
NEUTRAL_DARK = "#222222"
NEUTRAL_GREY = "#5C5C5C"
CAPTION_COLOUR = "dimgray"

# Per-pair colour assignments for the rolling-correlation chart.
PAIR_COLOURS: Dict[str, str] = {
    OIL_NAME:         PRIMARY_BLUE,
    MAG7_BASKET_NAME: PRIMARY_RED,
    QQQ_NAME:         PRIMARY_GREEN,
}

# Per-asset colours used in the price panel and distribution plots.
ASSET_COLOURS: Dict[str, str] = {
    USD_NAME:         NEUTRAL_DARK,
    OIL_NAME:         PRIMARY_BLUE,
    MAG7_BASKET_NAME: PRIMARY_RED,
    QQQ_NAME:         PRIMARY_GREEN,
    GOLD_NAME:        ACCENT_RED,
}

# Assets included in the context plots (price panel and return distributions).
# Gold is downloaded for the trio correlation matrices but kept out of these
# context charts so the rolling-correlation universe stays visually clean.
CONTEXT_ASSETS: List[str] = [USD_NAME, OIL_NAME, MAG7_BASKET_NAME, QQQ_NAME]

# Plotting: line widths, alphas, dpi.
REFERENCE_AXIS_LINEWIDTH = 0.5
ROLLING_CORR_LINEWIDTH = 1.4
PRICE_LINEWIDTH = 1.2
HISTOGRAM_BINS = 60
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Plotting: caption layout.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18

# Filesystem layout. One sub-directory per script keeps each pipeline's
# outputs cleanly separated from any other pipeline in this repository.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "usd_oil_ai_rolling_correlation"
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
log = logging.getLogger("usd-oil-ai")


# Result containers (dataclasses).

@dataclass
class PairCorrelationSummary:
    """
    Summary of the full-sample correlation and rolling-correlation statistics
    for a single (USD, other-asset) pair.

    INPUTS:
        * other_asset
        * n_obs
        * full_sample_corr_pearson
        * full_sample_corr_spearman
        * rolling_long_mean
        * rolling_long_min
        * rolling_long_max
        * rolling_short_mean
        * rolling_short_min
        * rolling_short_max

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    other_asset: str
    n_obs: int
    full_sample_corr_pearson: float
    full_sample_corr_spearman: float
    rolling_long_mean: float
    rolling_long_min: float
    rolling_long_max: float
    rolling_short_mean: float
    rolling_short_min: float
    rolling_short_max: float


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
        * pair_summaries

    OUTPUTS:
        * Dataclass aggregating the metadata and per-pair correlation records.
    """
    run_at: str
    start: str
    end: str
    n_obs: int
    notes: List[str] = field(default_factory = list)
    pair_summaries: List[PairCorrelationSummary] = field(default_factory = list)


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


def extract_close_series(raw_frame: pd.DataFrame, friendly_name: str) -> pd.Series:
    """
    Pull the close-price column out of a raw yfinance frame, defensively
    handling the case where the column comes back as a one-column DataFrame
    rather than a Series.

    INPUTS:
        * raw_frame      : DataFrame returned by yfinance.download
        * friendly_name  : name to assign to the output Series

    OUTPUTS:
        * Close-price Series renamed to friendly_name.
    """
    close_col = "Close" if "Close" in raw_frame.columns else raw_frame.columns[0]
    close_series = raw_frame[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series.rename(friendly_name)


def build_mag7_basket(
    mag7_close_frame: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Construct an equal-weighted Mag 7 basket from the per-constituent close
    prices. Daily log returns are averaged across the seven names and then
    compounded into a synthetic price index starting at
    BASKET_BASE_INDEX_VALUE.

    INPUTS:
        * mag7_close_frame  : DataFrame of close prices, one column per
                              MAG7_TICKERS member

    OUTPUTS:
        * Dict with two keys:
            'returns' : Series of equal-weighted daily log returns.
            'price'   : Series of the synthetic basket price index.
    """
    aligned_frame = mag7_close_frame.dropna()
    constituent_returns = np.log(aligned_frame).diff().dropna()
    basket_returns = constituent_returns.mean(axis = 1)
    basket_returns = basket_returns.rename(MAG7_BASKET_NAME)

    basket_price_values = BASKET_BASE_INDEX_VALUE * np.exp(basket_returns.cumsum())
    basket_price_series = basket_price_values.rename(MAG7_BASKET_NAME)
    return {"returns": basket_returns, "price": basket_price_series}


def fetch_panel(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Build the aligned panel of close prices, the Mag 7 basket, and the daily
    log returns used downstream. Raw yfinance frames are cached as Parquet
    under DATA_RAW_DIR so that subsequent runs can inspect the un-processed
    inputs without refetching.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Dict with keys:
            'prices'  : DataFrame of close prices for USD, oil, QQQ and
                        the synthetic Mag 7 basket.
            'returns' : DataFrame of daily log returns for the same four
                        series (columns prefixed 'r_').
    """
    log.info("Downloading USD, oil, QQQ, gold and the Mag 7 constituents")

    usd_raw = download_single_ticker(USD_SYMBOL, start_date, end_date)
    save_table(usd_raw, DATA_RAW_DIR, f"raw_{USD_NAME}")
    usd_close = extract_close_series(usd_raw, USD_NAME)

    oil_raw = download_single_ticker(OIL_SYMBOL, start_date, end_date)
    save_table(oil_raw, DATA_RAW_DIR, f"raw_{OIL_NAME}")
    oil_close = extract_close_series(oil_raw, OIL_NAME)

    qqq_raw = download_single_ticker(QQQ_SYMBOL, start_date, end_date)
    save_table(qqq_raw, DATA_RAW_DIR, f"raw_{QQQ_NAME}")
    qqq_close = extract_close_series(qqq_raw, QQQ_NAME)

    gold_raw = download_single_ticker(GOLD_SYMBOL, start_date, end_date)
    save_table(gold_raw, DATA_RAW_DIR, f"raw_{GOLD_NAME}")
    gold_close = extract_close_series(gold_raw, GOLD_NAME)

    mag7_close_columns: List[pd.Series] = []
    for ticker_symbol in MAG7_TICKERS:
        constituent_raw = download_single_ticker(ticker_symbol, start_date, end_date)
        save_table(constituent_raw, DATA_RAW_DIR, f"raw_mag7_{ticker_symbol.lower()}")
        constituent_close = extract_close_series(constituent_raw, ticker_symbol)
        mag7_close_columns.append(constituent_close)
    mag7_close_frame = pd.concat(mag7_close_columns, axis = 1)

    basket_pieces = build_mag7_basket(mag7_close_frame)
    basket_price_series = basket_pieces["price"]

    prices = pd.concat(
        [usd_close, oil_close, qqq_close, gold_close, basket_price_series],
        axis = 1,
    )
    prices = prices.dropna(how = "all")
    prices = prices.ffill(limit = PANEL_FFILL_LIMIT).dropna()

    log.info(
        "Aligned panel: %d rows (%s to %s)",
        prices.shape[0],
        prices.index.min().date(),
        prices.index.max().date(),
    )

    returns = np.log(prices).diff().dropna()
    new_columns: List[str] = []
    for col_name in returns.columns:
        new_columns.append(f"r_{col_name}")
    returns.columns = new_columns

    save_table(mag7_close_frame, DATA_PROCESSED_DIR, "mag7_constituent_close")
    return {"prices": prices, "returns": returns}


# Rolling correlations.

def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """
    Compute the rolling Pearson correlation between two return series.

    INPUTS:
        * series_a    : first return series
        * series_b    : second return series
        * window      : rolling window length in trading days
        * min_periods : minimum window observations required for a point

    OUTPUTS:
        * Series of rolling Pearson correlations indexed by window end-date.
    """
    return series_a.rolling(window = window, min_periods = min_periods).corr(series_b)


def summarise_pair(
    returns: pd.DataFrame,
    other_asset_name: str,
    rolling_long: pd.Series,
    rolling_short: pd.Series,
) -> PairCorrelationSummary:
    """
    Build a PairCorrelationSummary for one (USD, other-asset) pair: the
    full-sample Pearson and Spearman correlations, plus the mean / min /
    max of each of the two rolling correlation series.

    INPUTS:
        * returns           : DataFrame of daily log returns (cols 'r_...')
        * other_asset_name  : friendly name of the second asset
        * rolling_long      : rolling 252-day correlation Series
        * rolling_short     : rolling 63-day correlation Series

    OUTPUTS:
        * Populated PairCorrelationSummary record.
    """
    usd_returns = returns[f"r_{USD_NAME}"]
    other_returns = returns[f"r_{other_asset_name}"]
    long_clean = rolling_long.dropna()
    short_clean = rolling_short.dropna()
    return PairCorrelationSummary(
        other_asset = other_asset_name,
        n_obs = int(len(returns)),
        full_sample_corr_pearson = float(usd_returns.corr(other_returns, method = "pearson")),
        full_sample_corr_spearman = float(usd_returns.corr(other_returns, method = "spearman")),
        rolling_long_mean = float(long_clean.mean()) if not long_clean.empty else float("nan"),
        rolling_long_min = float(long_clean.min()) if not long_clean.empty else float("nan"),
        rolling_long_max = float(long_clean.max()) if not long_clean.empty else float("nan"),
        rolling_short_mean = float(short_clean.mean()) if not short_clean.empty else float("nan"),
        rolling_short_min = float(short_clean.min()) if not short_clean.empty else float("nan"),
        rolling_short_max = float(short_clean.max()) if not short_clean.empty else float("nan"),
    )


# Bearish regime and trio correlation matrices.

def rolling_four_week_return(prices: pd.DataFrame, ticker_name: str) -> pd.Series:
    """
    Compute the rolling 4-week (20 trading day) log return for one ticker.

    INPUTS:
        * prices       : DataFrame of close prices
        * ticker_name  : column name to operate on

    OUTPUTS:
        * Series of rolling 4-week log returns indexed by date.
    """
    log_prices = np.log(prices[ticker_name])
    four_week = log_prices - log_prices.shift(FOUR_WEEK_TRADING_DAYS)
    return four_week.dropna().rename(f"r4w_{ticker_name}")


def rolling_realised_volatility(returns: pd.DataFrame, ticker_name: str) -> pd.Series:
    """
    Compute the annualised rolling realised volatility for one ticker.

    INPUTS:
        * returns      : DataFrame of daily log returns (cols prefixed 'r_')
        * ticker_name  : ticker friendly name (input column is 'r_<ticker>')

    OUTPUTS:
        * Annualised realised-vol Series indexed by date.
    """
    target_returns = returns[f"r_{ticker_name}"]
    rolling_std = target_returns.rolling(REALISED_VOL_WINDOW).std()
    annualised = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annualised.rename(f"rv_{ticker_name}")


def bearish_regime_mask(
    four_week: pd.Series, realised_vol: pd.Series,
) -> pd.Series:
    """
    Build the bearish regime mask used for the regime-filtered correlation
    matrices: 4-week return at or below BEARISH_QUANTILE AND realised vol at
    or above HIGH_VOL_QUANTILE, both thresholds computed full-sample.

    INPUTS:
        * four_week     : 4-week return Series for the regime target ticker
        * realised_vol  : annualised realised vol Series for the same ticker

    OUTPUTS:
        * Boolean Series aligned to the intersection of the two input indices;
          True on bearish-and-high-vol days.
    """
    common_index = four_week.dropna().index.intersection(realised_vol.dropna().index)
    fw_aligned = four_week.reindex(common_index)
    rv_aligned = realised_vol.reindex(common_index)
    bearish = fw_aligned <= fw_aligned.quantile(BEARISH_QUANTILE)
    high_vol = rv_aligned >= rv_aligned.quantile(HIGH_VOL_QUANTILE)
    return bearish & high_vol


def trio_correlation_matrix(
    returns: pd.DataFrame, mask: pd.Series, method: str,
) -> pd.DataFrame:
    """
    Build a 3x3 correlation matrix over [r_gold, r_usd, r_qqq] restricted to
    the days where the supplied mask is True. The column order follows
    TRIO_ASSET_ORDER so the output is comparable across sub-samples.

    INPUTS:
        * returns  : DataFrame of daily log returns
        * mask     : boolean Series indexed by date
        * method   : 'pearson' or 'spearman'

    OUTPUTS:
        * 3x3 correlation matrix DataFrame, or an empty DataFrame when there
          are no observations in the masked sample.
    """
    aligned_mask = mask.reindex(returns.index).fillna(False).astype(bool)
    subset_frame = returns.loc[aligned_mask]
    if len(subset_frame) == 0:
        return pd.DataFrame()
    return_cols = [f"r_{asset_name}" for asset_name in TRIO_ASSET_ORDER]
    matrix = subset_frame[return_cols].corr(method = method)
    display_labels = [TRIO_DISPLAY_LABELS[asset_name]
                      for asset_name in TRIO_ASSET_ORDER]
    matrix.index = display_labels
    matrix.columns = display_labels
    return matrix


# Plotting helpers.

def add_caption(fig: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption beneath a figure. Reserves vertical margin
    so that the caption is not clipped when the figure is saved.

    INPUTS:
        * fig            : matplotlib Figure
        * caption_text   : caption string defining every symbol in the figure

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
        * Path to the written PNG.
    """
    output_path = VISUALS_DIR / file_name
    fig.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(fig)
    return output_path


def plot_price_panel(prices: pd.DataFrame) -> Path:
    """
    Render the four price series rebased so that each starts at 100 on the
    first trading day of the sample.

    INPUTS:
        * prices  : DataFrame of close prices for USD, oil, QQQ, Mag 7

    OUTPUTS:
        * Path to the saved PNG.
    """
    normalised = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize = TALL_WIDE_FIGSIZE)
    for asset_name in CONTEXT_ASSETS:
        if asset_name not in normalised.columns:
            continue
        ax.plot(
            normalised.index, normalised[asset_name],
            color = ASSET_COLOURS.get(asset_name, NEUTRAL_DARK),
            lw = PRICE_LINEWIDTH,
            label = ASSET_DISPLAY.get(asset_name, asset_name),
        )
    ax.set_title("Normalised price series (start = 100)")
    ax.set_ylabel("Price index")
    ax.legend(loc = "upper left")
    fig.tight_layout()
    add_caption(
        fig,
        "Daily close prices for US Dollar Index (DX-Y.NYB), WTI crude oil "
        "(CL=F), Nasdaq-100 ETF (QQQ) and the Mag 7 equal-weighted basket "
        "(NVDA, MSFT, GOOGL, AAPL, AMZN, META, TSLA), each rebased to 100 "
        "on the first trading day in the sample.",
    )
    return save_figure(fig, "price_panel.png")


def plot_return_distributions(returns: pd.DataFrame) -> Path:
    """
    Render KDE histograms of the daily log returns of each series so the
    relative volatility and tail shape are visible.

    INPUTS:
        * returns  : DataFrame of daily log returns (cols prefixed 'r_')

    OUTPUTS:
        * Path to the saved PNG.
    """
    asset_order = CONTEXT_ASSETS
    fig_width = DISTRIBUTION_PANEL_WIDTH_PER_SERIES * len(asset_order)
    fig, axes = plt.subplots(
        1, len(asset_order),
        figsize = (fig_width, DISTRIBUTION_PANEL_HEIGHT),
        sharey = True,
    )
    for ax, asset_name in zip(axes, asset_order):
        col_name = f"r_{asset_name}"
        series = returns[col_name].dropna()
        sns.histplot(
            series, bins = HISTOGRAM_BINS, kde = True,
            ax = ax, color = ASSET_COLOURS[asset_name],
        )
        ax.set_title(ASSET_DISPLAY[asset_name], fontsize = TEXT_FONT)
        ax.set_xlabel("Daily log return")
    fig.suptitle("Return distributions", fontsize = TITLE_FONT)
    fig.tight_layout()
    add_caption(
        fig,
        "Histogram and KDE of daily log returns for each series. The wider "
        "the spread, the higher the realised daily volatility. r_X = daily "
        "log return of asset X.",
    )
    return save_figure(fig, "return_distributions.png")


def plot_trio_correlation_matrices(
    matrices_by_sample_and_method: Dict[tuple[str, str], pd.DataFrame],
    n_obs_by_sample: Dict[str, int],
) -> Path:
    """
    Render a 2x2 grid of correlation matrices over [gold, USD, NASDAQ].
    Rows compare the full sample with the bearish regime; columns compare
    the Pearson and Spearman methods.

    INPUTS:
        * matrices_by_sample_and_method  : dict keyed by
          (sample_label, method) where sample_label is 'full' or 'regime'
          and method is 'pearson' or 'spearman'
        * n_obs_by_sample                : dict {sample_label: int} giving
          the number of days in each sub-sample

    OUTPUTS:
        * Path to the saved PNG.
    """
    sample_labels = ["full", "regime"]
    sample_display = {
        "full":   "Full sample",
        "regime": "Bearish regime (NASDAQ)",
    }
    methods = ["pearson", "spearman"]

    fig, axes = plt.subplots(
        len(sample_labels), len(methods),
        figsize = MATRIX_GRID_FIGSIZE,
    )
    for row_idx, sample_label in enumerate(sample_labels):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            matrix = matrices_by_sample_and_method.get((sample_label, method))
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
            n_in_sample = n_obs_by_sample.get(sample_label, 0)
            ax.set_title(
                f"{method.capitalize()} | {sample_display[sample_label]} "
                f"(n = {n_in_sample})",
                fontsize = TEXT_FONT,
            )
    fig.suptitle(
        "Gold, USD and NASDAQ correlations: full sample vs bearish regime",
        fontsize = TITLE_FONT,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Correlation matrices over [gold (GC=F), US Dollar Index (DX-Y.NYB), "
        "Nasdaq-100 (QQQ)]. Top row: full sample. Bottom row: days where the "
        "NASDAQ 4-week return is at or below its 10th percentile AND its "
        "21-day annualised realised volatility is at or above its 75th "
        "percentile. Left: Pearson (linear). Right: Spearman (rank-based, "
        "robust to outliers). n = number of trading days in each sub-sample.",
    )
    return save_figure(fig, "trio_correlation_matrices.png")


def plot_rolling_correlation_comparison(
    rolling_series_by_pair: Dict[str, pd.Series],
    window: int,
    file_name: str,
) -> Path:
    """
    Render the headline figure: one line per (USD, other-asset) pair on the
    same axes, with a zero reference and a clear legend.

    INPUTS:
        * rolling_series_by_pair  : dict {other_asset_name: rolling corr Series}
        * window                  : rolling window length (used in title only)
        * file_name               : output PNG file name

    OUTPUTS:
        * Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize = WIDE_FIGSIZE)
    for other_asset_name, rolling_series in rolling_series_by_pair.items():
        clean_series = rolling_series.dropna()
        if clean_series.empty:
            continue
        ax.plot(
            clean_series.index, clean_series.values,
            color = PAIR_COLOURS.get(other_asset_name, NEUTRAL_DARK),
            lw = ROLLING_CORR_LINEWIDTH,
            label = f"USD vs {ASSET_DISPLAY.get(other_asset_name, other_asset_name)}",
        )
    ax.axhline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_title(
        f"Rolling {window}-day correlation with the US Dollar Index"
    )
    ax.set_ylabel("Rolling Pearson rho")
    ax.legend(loc = "upper left")
    fig.tight_layout()
    add_caption(
        fig,
        f"Rolling {window}-day Pearson correlation between the US Dollar "
        "Index daily log return and each of WTI crude oil, the Mag 7 equal-"
        "weighted basket, and the Nasdaq-100 ETF. rho = Pearson correlation. "
        "Each line is stamped at the window end-date.",
    )
    return save_figure(fig, file_name)


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
    Execute the end-to-end pipeline: fetch the data panel, build the Mag 7
    basket, compute daily log returns, compute long- and short-window
    rolling correlations against the US Dollar Index for each of oil, the
    Mag 7 basket and QQQ, render every figure and persist every table.

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
    returns = panel["returns"]
    save_table(prices, DATA_PROCESSED_DIR, "prices")
    save_table(returns, DATA_PROCESSED_DIR, "log_returns")

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(prices.index.min().date()),
        end = str(prices.index.max().date()),
        n_obs = int(len(returns)),
    )
    summary.notes.append(
        "USD = ICE US Dollar Index (DX-Y.NYB). Oil = WTI front-month "
        "(CL=F). Mag 7 = equal-weighted log-return basket of NVDA, MSFT, "
        "GOOGL, AAPL, AMZN, META, TSLA, compounded into a synthetic price "
        "index. QQQ = Invesco QQQ Trust (Nasdaq-100)."
    )

    log.info("Generating context plots")
    plot_price_panel(prices)
    plot_return_distributions(returns)

    log.info("Computing rolling correlations against USD")
    usd_returns = returns[f"r_{USD_NAME}"]
    other_asset_names = [OIL_NAME, MAG7_BASKET_NAME, QQQ_NAME]

    rolling_long_by_pair: Dict[str, pd.Series] = {}
    rolling_short_by_pair: Dict[str, pd.Series] = {}
    for other_asset_name in other_asset_names:
        other_returns = returns[f"r_{other_asset_name}"]
        rolling_long = rolling_correlation(
            usd_returns, other_returns,
            ROLLING_CORR_LONG_WINDOW, MIN_PERIODS_LONG,
        )
        rolling_short = rolling_correlation(
            usd_returns, other_returns,
            ROLLING_CORR_SHORT_WINDOW, MIN_PERIODS_SHORT,
        )
        rolling_long_by_pair[other_asset_name] = rolling_long
        rolling_short_by_pair[other_asset_name] = rolling_short
        save_table(
            rolling_long.to_frame(name = "corr"),
            DATA_PROCESSED_DIR,
            f"rolling_corr_long_usd_{other_asset_name}",
        )
        save_table(
            rolling_short.to_frame(name = "corr"),
            DATA_PROCESSED_DIR,
            f"rolling_corr_short_usd_{other_asset_name}",
        )
        summary.pair_summaries.append(
            summarise_pair(returns, other_asset_name, rolling_long, rolling_short)
        )

    log.info("Plotting rolling-correlation comparison")
    plot_rolling_correlation_comparison(
        rolling_long_by_pair,
        ROLLING_CORR_LONG_WINDOW,
        "rolling_correlation_long.png",
    )
    plot_rolling_correlation_comparison(
        rolling_short_by_pair,
        ROLLING_CORR_SHORT_WINDOW,
        "rolling_correlation_short.png",
    )

    log.info("Building gold / USD / NASDAQ correlation matrices")
    target_name = BEARISH_REGIME_TARGET
    target_four_week = rolling_four_week_return(prices, target_name)
    target_realised_vol = rolling_realised_volatility(returns, target_name)
    save_table(
        target_four_week.to_frame(),
        DATA_PROCESSED_DIR,
        f"four_week_returns_{target_name}",
    )
    save_table(
        target_realised_vol.to_frame(),
        DATA_PROCESSED_DIR,
        f"realised_volatility_{target_name}",
    )

    regime_mask = bearish_regime_mask(target_four_week, target_realised_vol)
    full_sample_mask = pd.Series(True, index = returns.index)

    matrices_by_sample_and_method: Dict[tuple[str, str], pd.DataFrame]={}
    n_obs_by_sample: Dict[str, int] = {}
    for sample_label, mask in [
        ("full", full_sample_mask),
        ("regime", regime_mask),
    ]:
        n_obs_by_sample[sample_label] = int(
            mask.reindex(returns.index).fillna(False).astype(bool).sum()
        )
        for method in ["pearson", "spearman"]:
            matrix = trio_correlation_matrix(returns, mask, method)
            matrices_by_sample_and_method[(sample_label, method)] = matrix
            save_table(
                matrix,
                DATA_PROCESSED_DIR,
                f"trio_corr_{method}_{sample_label}",
            )
    plot_trio_correlation_matrices(matrices_by_sample_and_method, n_obs_by_sample)

    matrices_json: Dict[str, Any] = {}
    for (sample_label, method), matrix in matrices_by_sample_and_method.items():
        matrices_json[f"trio_{sample_label}_{method}"] = matrix.round(4).to_dict()

    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_payload = to_jsonable(summary)
    summary_payload["trio_correlation_matrices"] = matrices_json
    summary_payload["trio_n_obs"] = n_obs_by_sample
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
    print("\nPer-pair correlation summary (vs USD):")
    header = (
        f"  {'other_asset':10s}  {'rho_p':>6s}  {'rho_s':>6s}  "
        f"{'L mean':>7s}  {'L min':>7s}  {'L max':>7s}  "
        f"{'S mean':>7s}  {'S min':>7s}  {'S max':>7s}"
    )
    print(header)
    for pair_record in summary.pair_summaries:
        print(
            f"  {pair_record.other_asset:10s}  "
            f"{pair_record.full_sample_corr_pearson:+6.3f}  "
            f"{pair_record.full_sample_corr_spearman:+6.3f}  "
            f"{pair_record.rolling_long_mean:+7.3f}  "
            f"{pair_record.rolling_long_min:+7.3f}  "
            f"{pair_record.rolling_long_max:+7.3f}  "
            f"{pair_record.rolling_short_mean:+7.3f}  "
            f"{pair_record.rolling_short_min:+7.3f}  "
            f"{pair_record.rolling_short_max:+7.3f}"
        )
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")
    print(f"Raw in     : {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
