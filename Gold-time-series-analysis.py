"""
Gold-time-series-analysis.py

Conditional gold-performance analysis against South Korean semiconductor
producers (Samsung Electronics, SK Hynix) and the global semiconductor ETF
(SOXX), focusing on negative-volatile regimes.

Research question
-----------------
When semiconductor equities suffer left-tail 4-week price moves, AND when
volatility regimes are elevated, how does gold behave? Is it a diversifier,
a hedge, or does it co-move with the stress?

Volatility proxies
------------------
Historical per-stock implied volatility is not available on free data sources
(yfinance returns only current option snapshots). This pipeline therefore uses
two complementary proxies, run in parallel for every IV-dependent output:

    1. CBOE VIX (ticker ^VIX) - global equity-vol regime indicator.
    2. Stock-specific 21-day rolling realised volatility - a local IV proxy.

For every IV-dependent output the pipeline produces three variants:

    a) NO_VOL_FILTER  - left-quartile 4w semi return only, no vol condition.
    b) HIGH_VIX       - left-quartile 4w semi return AND VIX in its top quartile.
    c) HIGH_RV        - left-quartile 4w semi return AND the stock's own 21d
                        realised volatility in its top quartile.

Pipeline
--------
1. Data acquisition (yfinance) for prices, trading volume, and VIX.
2. Daily log returns and 4-week (20 trading day) rolling returns.
3. Stock-specific 21d annualised realised volatility.
4. Regime masks: left-quartile 4w semi returns, high VIX, high realised vol.
5. Conditional gold-performance summary (mean, median, hit rate, etc.) under
   each of the three regime variants for each semi ticker.
6. Rolling 252-day conditional correlation: corr(gold, semi) restricted to
   days satisfying the regime mask. Plotted for all three variants.
7. Full-period correlation matrices (Pearson and Spearman) on regime-filtered
   data, one matrix per regime variant.
8. Supplementary context plots: price panel, VIX with high-regime highlight,
   4w return distributions, trading volume.
9. Persistence of artefacts (figures, tables, JSON summary) to ./outputs/.

Run
---
    python Gold-time-series-analysis.py --start 2015-01-01 --end 2026-05-01
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Price tickers (friendly name -> Yahoo Finance symbol).
PRICE_TICKERS: Dict[str, str] = {
    "gold":    "GC=F",
    "samsung": "005930.KS",
    "skhynix": "000660.KS",
    "soxx":    "SOXX",
}

# Tickers for which we want volume (not gold, since gold here is a future).
VOLUME_TICKERS: List[str] = ["samsung", "skhynix", "soxx"]

# Semi tickers used in the conditional analysis.
SEMI_TICKERS: List[str] = ["samsung", "skhynix", "soxx"]

# VIX as the global vol-regime proxy.
VIX_TICKER = "^VIX"

# Calendar / windows.
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

# Plotting.
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

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

# Filesystem.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# CLI defaults.
DEFAULT_START_DATE = "2015-01-01"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gold-vol-regime")


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class ConditionalGoldStats:
    """Summary of gold behaviour under a single regime for a single semi."""
    ticker: str
    regime: str
    n_obs: int
    gold_mean_bps: float
    gold_median_bps: float
    gold_std_bps: float
    gold_hit_rate_positive: float    # fraction of regime days with r_gold > 0
    semi_mean_bps: float
    corr_pearson: float
    corr_spearman: float


@dataclass
class PipelineSummary:
    run_at: str
    start: str
    end: str
    n_obs: int
    notes: List[str] = field(default_factory=list)
    conditional_stats: List[ConditionalGoldStats] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Data acquisition
# -----------------------------------------------------------------------------

def _download_one(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download one ticker, flatten MultiIndex columns if necessary."""
    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def fetch_panel(start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Build a dictionary of price, volume, and VIX series.

    Inputs:
        start_date - ISO date string, inclusive
        end_date   - ISO date string, exclusive
    Output:
        Dict with keys:
            'prices'  - DataFrame of adjusted close prices (one column per
                        friendly ticker name)
            'volumes' - DataFrame of daily traded volume for VOLUME_TICKERS
            'vix'     - Series of VIX close values
    Method:
        Each ticker is downloaded individually, columns are flattened, and
        the three resulting frames are aligned on a forward-filled common
        trading calendar so that downstream analysis sees a clean rectangular
        panel.
    """
    log.info("Downloading %d price tickers + VIX...", len(PRICE_TICKERS))
    price_frames: List[pd.Series] = []
    volume_frames: List[pd.Series] = []

    for friendly_name, symbol in PRICE_TICKERS.items():
        raw_df = _download_one(symbol, start_date, end_date)
        close_col = "Close" if "Close" in raw_df.columns else raw_df.columns[0]
        price_series = raw_df[close_col]
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        price_frames.append(price_series.rename(friendly_name))
        if friendly_name in VOLUME_TICKERS and "Volume" in raw_df.columns:
            vol_series = raw_df["Volume"]
            if isinstance(vol_series, pd.DataFrame):
                vol_series = vol_series.iloc[:, 0]
            volume_frames.append(vol_series.rename(friendly_name))

    vix_raw = _download_one(VIX_TICKER, start_date, end_date)
    vix_close_col = "Close" if "Close" in vix_raw.columns else vix_raw.columns[0]
    vix_series = vix_raw[vix_close_col]
    if isinstance(vix_series, pd.DataFrame):
        vix_series = vix_series.iloc[:, 0]
    vix_series = vix_series.rename("vix")

    prices = pd.concat(price_frames, axis=1).dropna(how="all")
    prices = prices.ffill(limit=PANEL_FFILL_LIMIT).dropna()
    volumes = pd.concat(volume_frames, axis=1).reindex(prices.index).ffill()
    vix_aligned = vix_series.reindex(prices.index).ffill()

    log.info(
        "Aligned panel: %d rows (%s -> %s)",
        prices.shape[0],
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return {"prices": prices, "volumes": volumes, "vix": vix_aligned}


# -----------------------------------------------------------------------------
# Returns, rolling windows, realised volatility
# -----------------------------------------------------------------------------

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, with columns prefixed by 'r_'."""
    returns = np.log(prices).diff().dropna()
    returns.columns = [f"r_{col}" for col in returns.columns]
    return returns


def rolling_four_week_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rolling FOUR_WEEK_TRADING_DAYS log return per series."""
    log_prices = np.log(prices)
    four_w = log_prices - log_prices.shift(FOUR_WEEK_TRADING_DAYS)
    four_w.columns = [f"r4w_{col}" for col in four_w.columns]
    return four_w.dropna(how="all")


def rolling_realised_vol(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling realised volatility (window = REALISED_VOL_WINDOW)."""
    rv = returns.rolling(REALISED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rv.columns = [f"rv_{col[2:]}" for col in returns.columns]  # strip 'r_'
    return rv


# -----------------------------------------------------------------------------
# Regime masks
# -----------------------------------------------------------------------------

def left_quartile_mask(four_week: pd.Series) -> pd.Series:
    """Boolean mask: True where the 4-week return is in its bottom quartile."""
    threshold = four_week.quantile(LEFT_QUARTILE_THRESHOLD)
    return four_week <= threshold


def high_vix_mask(vix: pd.Series) -> pd.Series:
    """Boolean mask: True where VIX is in its top quartile of the full sample."""
    threshold = vix.quantile(HIGH_VOL_QUANTILE)
    return vix >= threshold


def high_realised_vol_mask(realised_vol: pd.Series) -> pd.Series:
    """Boolean mask: True where the ticker's own realised vol is in its top quartile."""
    threshold = realised_vol.quantile(HIGH_VOL_QUANTILE)
    return realised_vol >= threshold


def build_regime_masks(
    four_week: pd.Series,
    vix: pd.Series,
    realised_vol_series: pd.Series,
) -> Dict[str, pd.Series]:
    """Return dict of regime masks keyed by REGIME_LABELS.

    Inputs:
        four_week           - 4-week rolling return for one semi ticker
        vix                 - VIX series, aligned to four_week's index
        realised_vol_series - 21d realised vol for the same semi ticker
    Output:
        Dict {regime_label: boolean Series aligned to four_week.index}.
    Method:
        Each regime mask conjuncts the left-quartile 4w return condition with
        the relevant volatility condition (none, high VIX, or high realised
        vol). All series are aligned on the intersection of their indices.
    """
    common_index = (
        four_week.dropna().index
        .intersection(vix.dropna().index)
        .intersection(realised_vol_series.dropna().index)
    )
    fw_aligned = four_week.reindex(common_index)
    vix_aligned = vix.reindex(common_index)
    rv_aligned = realised_vol_series.reindex(common_index)

    left_q = left_quartile_mask(fw_aligned)
    high_vix = high_vix_mask(vix_aligned)
    high_rv = high_realised_vol_mask(rv_aligned)

    return {
        REGIME_NO_FILTER: left_q,
        REGIME_HIGH_VIX:  left_q & high_vix,
        REGIME_HIGH_RV:   left_q & high_rv,
    }


# -----------------------------------------------------------------------------
# Conditional gold statistics
# -----------------------------------------------------------------------------

def conditional_gold_stats(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    ticker: str,
    regime_label: str,
) -> ConditionalGoldStats:
    """Compute gold performance statistics restricted to regime days.

    Inputs:
        daily_returns - DataFrame of daily log returns (columns prefixed 'r_')
        mask          - boolean Series indexed by date; True for regime days
        ticker        - semi ticker name (used for the semi return column)
        regime_label  - one of REGIME_LABELS
    Output:
        Populated ConditionalGoldStats dataclass.
    Method:
        Aligns mask to the daily returns index, then computes summary stats
        for r_gold on the regime days. Correlation is computed between r_gold
        and r_<ticker> on the same subset. All bps figures are 1e4 * raw log
        return.
    """
    aligned_mask = mask.reindex(daily_returns.index).fillna(False)
    sub = daily_returns.loc[aligned_mask]
    n_obs = int(len(sub))

    if n_obs == 0:
        nan = float("nan")
        return ConditionalGoldStats(
            ticker=ticker, regime=regime_label, n_obs=0,
            gold_mean_bps=nan, gold_median_bps=nan, gold_std_bps=nan,
            gold_hit_rate_positive=nan,
            semi_mean_bps=nan,
            corr_pearson=nan, corr_spearman=nan,
        )

    gold = sub["r_gold"]
    semi = sub[f"r_{ticker}"]
    return ConditionalGoldStats(
        ticker=ticker,
        regime=regime_label,
        n_obs=n_obs,
        gold_mean_bps=float(gold.mean() * 1e4),
        gold_median_bps=float(gold.median() * 1e4),
        gold_std_bps=float(gold.std() * 1e4),
        gold_hit_rate_positive=float((gold > 0).mean()),
        semi_mean_bps=float(semi.mean() * 1e4),
        corr_pearson=float(gold.corr(semi, method="pearson")),
        corr_spearman=float(gold.corr(semi, method="spearman")),
    )


# -----------------------------------------------------------------------------
# Rolling conditional correlation
# -----------------------------------------------------------------------------

def rolling_conditional_correlation(
    daily_returns: pd.DataFrame,
    mask: pd.Series,
    ticker: str,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.Series:
    """Rolling 252-day correlation between r_gold and r_<ticker>, restricted
    inside each window to days where `mask` is True.

    Inputs:
        daily_returns - DataFrame with r_gold and r_<ticker>
        mask          - boolean Series indexed by date
        ticker        - semi ticker name
        window        - rolling window length in trading days
    Output:
        Series of correlations indexed by window end-date. Windows with fewer
        than MIN_OBS_FOR_ROLLING_CORR regime days are skipped.
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
        return pd.Series(dtype=float)
    return pd.Series(dict(rolling_points))


# -----------------------------------------------------------------------------
# Regime-filtered correlation matrices
# -----------------------------------------------------------------------------

def regime_filtered_correlation_matrix(
    daily_returns: pd.DataFrame,
    regime_masks_by_ticker: Dict[str, pd.Series],
    method: str,
) -> pd.DataFrame:
    """Build a correlation matrix on regime-filtered data.

    Inputs:
        daily_returns          - DataFrame of daily log returns
        regime_masks_by_ticker - one boolean mask per semi ticker
        method                 - 'pearson' or 'spearman'
    Output:
        DataFrame correlation matrix over [r_gold, r_<each semi>], where each
        row/column pair (gold, semi_i) is computed on the union of regime days
        across all semis. The semi-semi cells are likewise computed on the
        union of masks so that the matrix is internally coherent.
    """
    union_mask = None
    for mask_series in regime_masks_by_ticker.values():
        aligned = mask_series.reindex(daily_returns.index).fillna(False)
        union_mask = aligned if union_mask is None else (union_mask | aligned)
    if union_mask is None:
        return pd.DataFrame()
    sub = daily_returns.loc[union_mask]
    cols = ["r_gold"] + [f"r_{t}" for t in SEMI_TICKERS]
    return sub[cols].corr(method=method)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, file_name: str) -> Path:
    """Save fig at DEFAULT_DPI and close it."""
    output_path = OUTPUT_DIR / file_name
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return output_path


def plot_price_panel(prices: pd.DataFrame, vix: pd.Series) -> Path:
    """Normalised price panel with VIX on a secondary axis."""
    fig, ax_left = plt.subplots(figsize=(12, 5.5))
    normalised = prices / prices.iloc[0] * 100
    normalised.plot(ax=ax_left)
    ax_left.set_ylabel("Price index (start = 100)")
    ax_left.set_title("Normalised prices + VIX")
    ax_right = ax_left.twinx()
    ax_right.plot(vix.index, vix.values, color="black", alpha=0.35, lw=0.7,
                  label="VIX")
    ax_right.set_ylabel("VIX")
    ax_right.legend(loc="upper right")
    fig.tight_layout()
    return _save_figure(fig, "prices_and_vix.png")


def plot_vix_regime(vix: pd.Series) -> Path:
    """VIX over time with the high-VIX regime highlighted."""
    threshold = vix.quantile(HIGH_VOL_QUANTILE)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vix.index, vix.values, color="steelblue", lw=0.8)
    ax.fill_between(
        vix.index, vix.values, threshold,
        where=(vix.values >= threshold),
        color="crimson", alpha=0.3, label="VIX >= 75th percentile",
    )
    ax.axhline(threshold, color="crimson", lw=0.8, linestyle="--",
               label=f"75th pctile = {threshold:.1f}")
    ax.set_title("VIX with high-volatility regime highlighted")
    ax.set_ylabel("VIX")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, "vix_regime.png")


def plot_realised_vol(realised_vol_df: pd.DataFrame) -> Path:
    """21-day annualised realised volatility, one line per semi ticker."""
    fig, ax = plt.subplots(figsize=(12, 5))
    semi_cols = [f"rv_{t}" for t in SEMI_TICKERS]
    realised_vol_df[semi_cols].plot(ax=ax)
    ax.set_title(f"{REALISED_VOL_WINDOW}d annualised realised volatility")
    ax.set_ylabel("Annualised vol")
    fig.tight_layout()
    return _save_figure(fig, "realised_volatility.png")


def plot_four_week_returns(four_week: pd.DataFrame) -> Path:
    """Distribution of 4-week semi returns with the left-quartile threshold."""
    semi_cols = [f"r4w_{t}" for t in SEMI_TICKERS]
    fig, axes = plt.subplots(1, len(semi_cols), figsize=(5 * len(semi_cols), 4))
    for ax, col_name in zip(axes, semi_cols):
        series = four_week[col_name].dropna()
        threshold = series.quantile(LEFT_QUARTILE_THRESHOLD)
        sns.histplot(series, bins=60, kde=True, ax=ax, color="steelblue")
        ax.axvline(threshold, color="crimson", linestyle="--",
                   label=f"Q1 = {threshold:+.3f}")
        ax.set_title(col_name)
        ax.legend()
    fig.suptitle("4-week semi return distributions", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, "four_week_return_distributions.png")


def plot_volume(volumes: pd.DataFrame) -> Path:
    """20-day rolling average daily trading volume for each semi."""
    smoothed = volumes.rolling(VOLUME_AVG_WINDOW).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    smoothed.plot(ax=ax)
    ax.set_title(f"{VOLUME_AVG_WINDOW}d rolling mean trading volume")
    ax.set_ylabel("Shares")
    fig.tight_layout()
    return _save_figure(fig, "trading_volume.png")


def plot_conditional_gold_bars(
    stats_list: List[ConditionalGoldStats],
    regime_label: str,
) -> Path:
    """Bar chart of gold mean return and hit-rate-positive per ticker.

    For one regime, shows mean gold return (bps, left axis) and the fraction
    of regime days on which gold closed up (right axis), per semi ticker.
    """
    filtered = [stat for stat in stats_list if stat.regime == regime_label]
    fig, ax_left = plt.subplots(figsize=(8, 5))
    tickers = [stat.ticker for stat in filtered]
    means = [stat.gold_mean_bps for stat in filtered]
    hit_rates = [stat.gold_hit_rate_positive for stat in filtered]
    n_obs = [stat.n_obs for stat in filtered]

    bar_pos = np.arange(len(tickers))
    ax_left.bar(bar_pos, means, color="goldenrod", label="Gold mean (bps)")
    ax_left.set_xticks(bar_pos)
    ax_left.set_xticklabels(
        [f"{t}\n(n={n})" for t, n in zip(tickers, n_obs)]
    )
    ax_left.axhline(0, color="k", lw=0.5)
    ax_left.set_ylabel("Mean daily gold return (bps)")

    ax_right = ax_left.twinx()
    ax_right.plot(bar_pos, hit_rates, color="navy", marker="o",
                  label="Gold hit-rate positive")
    ax_right.set_ylabel("Fraction of regime days with r_gold > 0")
    ax_right.set_ylim(0, 1)
    ax_right.axhline(0.5, color="navy", linestyle=":", alpha=0.4)

    ax_left.set_title(
        f"Conditional gold performance\n[{REGIME_DISPLAY[regime_label]}]"
    )
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                   loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, f"conditional_gold_{regime_label}.png")


def plot_rolling_conditional_correlation(
    rolling_by_ticker: Dict[str, pd.Series],
    regime_label: str,
) -> Path:
    """Rolling 252d conditional correlation between gold and each semi."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker_name, series in rolling_by_ticker.items():
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=f"{ticker_name} vs gold")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(
        f"Rolling {ROLLING_CORR_WINDOW}d corr (gold vs semi) | "
        f"{REGIME_DISPLAY[regime_label]}"
    )
    ax.set_ylabel("Conditional rho")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, f"rolling_corr_{regime_label}.png")


def plot_correlation_matrices_grid(
    matrices_by_regime_and_method: Dict[Tuple[str, str], pd.DataFrame],
) -> Path:
    """3x2 grid of regime-filtered correlation matrices.

    Rows = regimes (REGIME_LABELS), columns = (Pearson, Spearman).
    """
    fig, axes = plt.subplots(
        len(REGIME_LABELS), 2,
        figsize=(11, 4.2 * len(REGIME_LABELS)),
    )
    for row_idx, regime_label in enumerate(REGIME_LABELS):
        for col_idx, method in enumerate(["pearson", "spearman"]):
            ax = axes[row_idx, col_idx]
            matrix = matrices_by_regime_and_method.get((regime_label, method))
            if matrix is None or matrix.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue
            sns.heatmap(
                matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, ax=ax, cbar=True,
                annot_kws={"size": 9},
            )
            ax.set_title(
                f"{method.capitalize()} | {REGIME_DISPLAY[regime_label]}",
                fontsize=10,
            )
    fig.suptitle(
        "Regime-filtered correlation matrices (rows = regime, cols = method)",
        fontsize=13,
    )
    fig.tight_layout()
    return _save_figure(fig, "correlation_matrices_grid.png")


# -----------------------------------------------------------------------------
# JSON serialisation helper
# -----------------------------------------------------------------------------

def _to_jsonable(obj):
    """Recursively convert dataclasses / numpy scalars to JSON-friendly types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _to_jsonable(val) for key, val in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(val) for val in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.round(4).to_dict()
    return obj


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run_pipeline(start_date: str, end_date: str) -> PipelineSummary:
    """End-to-end pipeline.

    Inputs:
        start_date - ISO date string, inclusive
        end_date   - ISO date string, exclusive
    Output:
        Populated PipelineSummary, also persisted to pipeline_summary.json.
    Method:
        Acquires data, constructs returns and rolling windows, builds the
        three regime masks per ticker, computes conditional summary stats,
        rolling 252d conditional correlations, and regime-filtered correlation
        matrices (Pearson and Spearman). All artefacts are written to
        OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_style(SEABORN_STYLE)

    panel = fetch_panel(start_date, end_date)
    prices = panel["prices"]
    volumes = panel["volumes"]
    vix = panel["vix"]

    prices.to_csv(OUTPUT_DIR / "prices.csv")
    volumes.to_csv(OUTPUT_DIR / "volumes.csv")
    vix.to_csv(OUTPUT_DIR / "vix.csv")

    daily_returns = to_log_returns(prices)
    four_week = rolling_four_week_returns(prices)
    realised_vol = rolling_realised_vol(daily_returns)
    daily_returns.to_csv(OUTPUT_DIR / "log_returns.csv")
    four_week.to_csv(OUTPUT_DIR / "four_week_returns.csv")
    realised_vol.to_csv(OUTPUT_DIR / "realised_volatility.csv")

    summary = PipelineSummary(
        run_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        start=str(prices.index.min().date()),
        end=str(prices.index.max().date()),
        n_obs=int(len(daily_returns)),
    )
    summary.notes.append(
        "Historical implied volatility per stock is not available on free "
        "data sources; used VIX (global) and 21d realised vol (stock-specific) "
        "as IV proxies."
    )

    log.info("Generating context plots...")
    plot_price_panel(prices, vix)
    plot_vix_regime(vix)
    plot_realised_vol(realised_vol)
    plot_four_week_returns(four_week)
    if not volumes.empty:
        plot_volume(volumes)

    # Build masks and conditional statistics for every (ticker, regime) pair.
    log.info("Building regime masks and conditional statistics...")
    masks_by_ticker: Dict[str, Dict[str, pd.Series]] = {}
    for ticker_name in SEMI_TICKERS:
        fw_series = four_week[f"r4w_{ticker_name}"]
        rv_series = realised_vol[f"rv_{ticker_name}"]
        masks_by_ticker[ticker_name] = build_regime_masks(
            fw_series, vix, rv_series
        )
        for regime_label, mask in masks_by_ticker[ticker_name].items():
            summary.conditional_stats.append(
                conditional_gold_stats(
                    daily_returns, mask, ticker_name, regime_label
                )
            )

    log.info("Plotting conditional gold performance...")
    for regime_label in REGIME_LABELS:
        plot_conditional_gold_bars(summary.conditional_stats, regime_label)

    log.info("Computing rolling 252d conditional correlations...")
    for regime_label in REGIME_LABELS:
        rolling_by_ticker: Dict[str, pd.Series] = {}
        for ticker_name in SEMI_TICKERS:
            mask = masks_by_ticker[ticker_name][regime_label]
            rolling_by_ticker[ticker_name] = rolling_conditional_correlation(
                daily_returns, mask, ticker_name, ROLLING_CORR_WINDOW
            )
            rolling_by_ticker[ticker_name].to_csv(
                OUTPUT_DIR / f"rolling_corr_{ticker_name}_{regime_label}.csv",
                header=["corr"],
            )
        plot_rolling_conditional_correlation(rolling_by_ticker, regime_label)

    log.info("Computing regime-filtered correlation matrices...")
    matrices_by_regime_and_method: Dict[Tuple[str, str], pd.DataFrame] = {}
    for regime_label in REGIME_LABELS:
        per_ticker_masks = {
            ticker_name: masks_by_ticker[ticker_name][regime_label]
            for ticker_name in SEMI_TICKERS
        }
        for method in ["pearson", "spearman"]:
            matrix = regime_filtered_correlation_matrix(
                daily_returns, per_ticker_masks, method
            )
            matrices_by_regime_and_method[(regime_label, method)] = matrix
            matrix.to_csv(
                OUTPUT_DIR / f"corr_matrix_{method}_{regime_label}.csv"
            )
    plot_correlation_matrices_grid(matrices_by_regime_and_method)

    matrices_json = {
        f"{regime_label}__{method}": matrix.round(4).to_dict()
        for (regime_label, method), matrix in matrices_by_regime_and_method.items()
    }

    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    summary_payload = _to_jsonable(summary)
    summary_payload["correlation_matrices"] = matrices_json
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    log.info("Wrote summary -> %s", summary_path)
    return summary


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse start / end date command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end", default=datetime.utcnow().strftime("%Y-%m-%d"),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Entry point. Runs the full pipeline and prints a concise summary."""
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\n=== Pipeline summary ===")
    print(f"Window  : {summary.start} -> {summary.end}  (n={summary.n_obs})")
    print("\nConditional gold statistics:")
    header = (
        f"  {'ticker':8s}  {'regime':22s}  {'n':>5s}  "
        f"{'gold_mu_bps':>11s}  {'gold_med_bps':>12s}  "
        f"{'hit_rate':>8s}  {'rho_p':>6s}  {'rho_s':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stat in summary.conditional_stats:
        regime_short = stat.regime
        print(
            f"  {stat.ticker:8s}  {regime_short:22s}  "
            f"{stat.n_obs:5d}  {stat.gold_mean_bps:+11.2f}  "
            f"{stat.gold_median_bps:+12.2f}  "
            f"{stat.gold_hit_rate_positive:8.3f}  "
            f"{stat.corr_pearson:+6.3f}  {stat.corr_spearman:+6.3f}"
        )
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nArtefacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
