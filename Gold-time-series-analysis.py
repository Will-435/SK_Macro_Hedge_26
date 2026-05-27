"""
Gold-time-series-analysis.py

Comprehensive time series analysis using a Generalised Additive Model (GAM)
to investigate the joint dynamics of Gold prices, South Korean semiconductor
producers (Samsung Electronics, SK Hynix), and the KRW/USD exchange rate.

Purpose
-------
Part of the SK_Macro_Hedge_26 research pipeline. This module quantifies the
diversification value and hedging value of Gold against South Korean
semiconductor equity exposure, conditioning on FX (KRW) regime. A GAM is used
because the relationship between gold, FX, and tech equities is well known to
be non-linear and regime-dependant (e.g. gold's hedging power strengthens in
left-tail FX-stress episodes).

Pipeline
--------
1. Data aquisition (Yahoo Finance) and alignment to a common trading calendar.
2. Returns construction (log returns) plus stationarity diagnostics.
3. Univariate descriptive statistics and rolling-volatility regime detection.
4. Static and rolling Pearson / Spearman correlations.
5. GAM model:
        r_semi_t = f1(r_gold_t) + f2(r_krw_t) + f3(vol_krw_t)
                 + te(r_gold_t, r_krw_t) + AR(1) + eps_t
   estimated for each semiconductor ticker.
6. Hedge-ratio / beta estimation in the local neighbourhood of large negative
   semiconductor return days (conditional hedging value).
7. Diversification metrics: average correlation, tail dependence coefficient,
   and the diversification ratio of an equal-weighted vs gold-augmented
   portfolio.
8. Diagnostics: residual ACF, Ljung-Box, partial dependence plots, and
   out-of-sample pseudo R^2 via an expanding-window backtest.
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from pygam import LinearGAM, s, te

import yfinance as yf


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Tickers used in the analysis. Map of friendly name -> Yahoo Finance symbol.
TICKERS: Dict[str, str] = {
    "gold":      "GC=F",        # COMEX gold front-month future (USD/oz)
    "krw":       "KRW=X",       # USD/KRW spot (KRW per USD)
    "samsung":   "005930.KS",   # Samsung Electronics
    "skhynix":   "000660.KS",   # SK Hynix
    "soxx":      "SOXX",        # iShares Semiconductor ETF (global semi control)
}

SEMI_TICKERS: List[str] = ["samsung", "skhynix"]

# Calendar / window constants.
TRADING_DAYS_PER_YEAR = 252
KRW_VOL_WINDOW = 21
ROLLING_CORR_WINDOW = 90
ROLLING_VOL_WINDOW = 30
ROLLING_BETA_WINDOW = 90
ROLLING_TAIL_CORR_WINDOW = 252

# Tail / quantile thresholds.
WORST_DECILE_QUANTILE = 0.10
TAIL_DEPENDENCE_QUANTILE = 0.05
LEFT_QUARTILE_QUANTILE = 0.25
MIN_TAIL_OBS_FOR_ROLLING_CORR = 10
PLOT_COLOUR_QUANTILE_LOW = 0.02
PLOT_COLOUR_QUANTILE_HIGH = 0.98

# Backtest configuration.
BACKTEST_MIN_TRAIN_OBS = 500
BACKTEST_STEP = 21

# Forward-fill tolerance when aligning the panel.
PANEL_FFILL_LIMIT = 2

# GAM hyperparameters.
GAM_N_SPLINES_MAIN = 12
GAM_N_SPLINES_AUX = 8
GAM_N_SPLINES_AR1 = 6
GAM_N_SPLINES_TENSOR = 6
GAM_N_SPLINES_BACKTEST = 10
GAM_N_SPLINES_BACKTEST_AUX = 6
GAM_N_SPLINES_BACKTEST_TENSOR = 5
GAM_LAMBDA_GRID = np.logspace(-3, 3, 7)
GAM_CI_WIDTH = 0.95

# Diagnostics.
LJUNG_BOX_LAG = 10
RESIDUAL_ACF_LAGS = 30
DEFAULT_CORR_METHODS = ("pearson", "spearman")

# Portfolio frontier resolution.
FRONTIER_N_WEIGHTS = 51
FRONTIER_LABELLED_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Plotting style.
DEFAULT_DPI = 140
BPS_SCALE = 1.0e4
SEABORN_STYLE = "whitegrid"

# Filesystem.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# CLI defaults.
DEFAULT_START_DATE = "2015-01-01"


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gold-gam")


# -----------------------------------------------------------------------------
# Data classes (lightweight result containers)
# -----------------------------------------------------------------------------

@dataclass
class GAMResult:
    """Summary statistics for one fitted GAM."""
    ticker: str
    pseudo_r2: float
    edof: float
    aic: float
    gold_p: float
    krw_p: float
    interaction_p: float
    residual_ljungbox_p: float
    n_obs: int


@dataclass
class HedgeMetrics:
    """Hedging and diversification metrics for one semi ticker against gold."""
    ticker: str
    full_sample_beta_gold: float
    full_sample_corr_gold: float
    tail_corr_gold: float            # corr conditional on bottom-decile semi returns
    tail_corr_krw: float
    tail_dependence_lower: float     # empirical lower-tail dependance coefficient
    avg_rolling_corr_gold: float
    diversification_ratio: float     # DR of 50/50 semi/gold vs 100% semi


@dataclass
class PipelineSummary:
    """Top-level container for one full pipeline run."""
    run_at: str
    start: str
    end: str
    n_obs: int
    stationarity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gam_results: List[GAMResult] = field(default_factory=list)
    hedge_metrics: List[HedgeMetrics] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Data acquisition
# -----------------------------------------------------------------------------

def fetch_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers and align on date index.

    Inputs:
        start_date  - ISO date string (inclusive)
        end_date    - ISO date string (exclusive)
    Output:
        A DataFrame with one column per friendly ticker name, indexed by date.
    Method:
        Pulls each ticker individually from Yahoo Finance, flattens any
        MultiIndex columns that newer yfinance versions return, concatenates
        into a single panel, then forward-fills small gaps (up to
        PANEL_FFILL_LIMIT days) and drops any remaining incomplete rows so that
        downstream models recieve a clean rectangular panel.
    """
    log.info("Downloading price data for %d tickers...", len(TICKERS))
    price_frames: List[pd.Series] = []
    for friendly_name, yahoo_symbol in TICKERS.items():
        raw_df = yf.download(
            yahoo_symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if raw_df is None or raw_df.empty:
            raise RuntimeError(f"No data returned for {yahoo_symbol}")
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        target_col = "Close" if "Close" in raw_df.columns else raw_df.columns[0]
        series = raw_df[target_col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        price_frames.append(series.rename(friendly_name))

    prices = pd.concat(price_frames, axis=1)
    prices = prices.dropna(how="all").ffill(limit=PANEL_FFILL_LIMIT).dropna()
    log.info(
        "Aligned panel: %d rows x %d cols (%s -> %s)",
        prices.shape[0],
        prices.shape[1],
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price panel to log returns and prefix column names with 'r_'."""
    returns = np.log(prices).diff().dropna()
    returns.columns = [f"r_{col}" for col in returns.columns]
    return returns


# -----------------------------------------------------------------------------
# Stationarity diagnostics
# -----------------------------------------------------------------------------

def stationarity_report(returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Run ADF and KPSS tests on each return column.

    Returns a nested dictionary keyed by series name with ADF / KPSS test
    statistics and p-values, suitable for serialisation to JSON.
    """
    report: Dict[str, Dict[str, float]] = {}
    for col_name in returns.columns:
        series_values = returns[col_name].dropna().values
        adf_stat, adf_p, *_ = adfuller(series_values, autolag="AIC")
        try:
            kpss_stat, kpss_p, *_ = kpss(
                series_values, regression="c", nlags="auto"
            )
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
        report[col_name] = {
            "adf_stat": float(adf_stat),
            "adf_p": float(adf_p),
            "kpss_stat": float(kpss_stat),
            "kpss_p": float(kpss_p),
        }
    return report


# -----------------------------------------------------------------------------
# Descriptive statistics and correlations
# -----------------------------------------------------------------------------

def descriptive_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Build a per-series descriptive statistics table with annualised vol."""
    desc = returns.describe().T[["mean", "std", "min", "max"]]
    desc["ann_vol"] = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    desc["skew"] = returns.skew()
    desc["kurt"] = returns.kurt()
    return desc


def rolling_corr(
    returns: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int = ROLLING_CORR_WINDOW,
) -> pd.Series:
    """Rolling Pearson correlation between two return columns."""
    return returns[col_a].rolling(window).corr(returns[col_b])


# -----------------------------------------------------------------------------
# GAM modelling
# -----------------------------------------------------------------------------

def build_gam_design(
    returns: pd.DataFrame, target_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Construct the GAM design matrix.

    Inputs:
        returns     - DataFrame of log returns with at least r_gold and r_krw
        target_col  - the dependant variable column name (e.g. 'r_samsung')
    Outputs:
        design_matrix  - 2D array of shape (n, 4)
        target_vector  - 1D array of shape (n,)
        feature_names  - list of column names matching the design columns
    Method:
        Features are r_gold, r_krw, a rolling KRW volatility column (window =
        KRW_VOL_WINDOW), and a one-day lag of the target to absorb short-run
        autocorrelation. Rows containing NaNs introduced by the rolling and
        lag operations are dropped.
    """
    working_df = returns.copy()
    working_df["krw_vol"] = working_df["r_krw"].rolling(KRW_VOL_WINDOW).std()
    lag_col = f"{target_col}_lag1"
    working_df[lag_col] = working_df[target_col].shift(1)
    working_df = working_df.dropna()
    feature_names = ["r_gold", "r_krw", "krw_vol", lag_col]
    design_matrix = working_df[feature_names].values
    target_vector = working_df[target_col].values
    return design_matrix, target_vector, feature_names


def fit_gam(design_matrix: np.ndarray, target_vector: np.ndarray) -> LinearGAM:
    """Fit a GAM with smooth terms for gold, KRW, KRW-vol, the AR(1) term, and
    a tensor-product interaction between gold and KRW. Smoothing penalties are
    selected by grid search over GAM_LAMBDA_GRID."""
    # Term indices: 0=gold, 1=krw, 2=krw_vol, 3=ar1
    gam_terms = (
        s(0, n_splines=GAM_N_SPLINES_MAIN)
        + s(1, n_splines=GAM_N_SPLINES_MAIN)
        + s(2, n_splines=GAM_N_SPLINES_AUX)
        + s(3, n_splines=GAM_N_SPLINES_AR1)
        + te(0, 1, n_splines=GAM_N_SPLINES_TENSOR)
    )
    gam_model = LinearGAM(gam_terms).gridsearch(
        design_matrix,
        target_vector,
        progress=False,
        lam=GAM_LAMBDA_GRID,
    )
    return gam_model


def gam_significance(gam_model: LinearGAM) -> Dict[int, float]:
    """Wald-style p-values for each smooth term, keyed by term index."""
    raw_pvals = gam_model.statistics_.get("p_values", [])
    return {term_idx: float(pval) for term_idx, pval in enumerate(raw_pvals)}


# -----------------------------------------------------------------------------
# Hedge / diversification metrics
# -----------------------------------------------------------------------------

def ols_beta(dependant: pd.Series, regressor: pd.Series) -> float:
    """OLS slope coefficient of `dependant` regressed on `regressor`."""
    regressor_with_const = sm.add_constant(regressor.values)
    fitted = sm.OLS(dependant.values, regressor_with_const).fit()
    return float(fitted.params[1])


def lower_tail_dependence(
    series_a: pd.Series,
    series_b: pd.Series,
    quantile: float = TAIL_DEPENDENCE_QUANTILE,
) -> float:
    """Empirical lower-tail dependance coefficient.

    Inputs:
        series_a, series_b - aligned return series
        quantile           - tail threshold (default TAIL_DEPENDENCE_QUANTILE)
    Output:
        Symmetric estimate of
            lambda_L = P(F_b(b) <= q | F_a(a) <= q)
        averaged with the analogous conditional on series_b.
    """
    rank_a = series_a.rank(pct=True)
    rank_b = series_b.rank(pct=True)
    cond_a_given_b = (
        ((rank_b <= quantile) & (rank_a <= quantile)).sum()
        / max((rank_a <= quantile).sum(), 1)
    )
    cond_b_given_a = (
        ((rank_a <= quantile) & (rank_b <= quantile)).sum()
        / max((rank_b <= quantile).sum(), 1)
    )
    return float((cond_a_given_b + cond_b_given_a) / 2.0)


def diversification_ratio(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Diversification ratio DR = (w' sigma) / sqrt(w' Sigma w)."""
    individual_vols = np.sqrt(np.diag(covariance))
    portfolio_vol = float(np.sqrt(weights @ covariance @ weights))
    return float((weights @ individual_vols) / portfolio_vol)


def compute_hedge_metrics(returns: pd.DataFrame, ticker: str) -> HedgeMetrics:
    """Compute the full set of hedge / diversification metrics for one ticker.

    Inputs:
        returns - DataFrame of log returns
        ticker  - friendly ticker name (used as r_<ticker> internally)
    Output:
        HedgeMetrics dataclass populated with all summary statistics.
    Method:
        Combines full-sample OLS beta, full-sample correlation, tail-conditional
        correlations (worst WORST_DECILE_QUANTILE of semi returns), empirical
        lower-tail dependance, average rolling correlation, and the
        diversification ratio of a 50/50 semi+gold sleeve.
    """
    semi_returns = returns[f"r_{ticker}"]
    gold_returns = returns["r_gold"]
    krw_returns = returns["r_krw"]

    full_beta_gold = ols_beta(semi_returns, gold_returns)
    full_corr_gold = float(semi_returns.corr(gold_returns))

    bottom_mask = semi_returns <= semi_returns.quantile(WORST_DECILE_QUANTILE)
    tail_corr_gold_val = float(
        semi_returns[bottom_mask].corr(gold_returns[bottom_mask])
    )
    tail_corr_krw_val = float(
        semi_returns[bottom_mask].corr(krw_returns[bottom_mask])
    )

    tail_dep = lower_tail_dependence(
        semi_returns, gold_returns, quantile=TAIL_DEPENDENCE_QUANTILE
    )
    avg_rolling = float(
        rolling_corr(returns, f"r_{ticker}", "r_gold", ROLLING_CORR_WINDOW).mean()
    )

    pair_cov = returns[[f"r_{ticker}", "r_gold"]].cov().values
    dr_5050 = diversification_ratio(np.array([0.5, 0.5]), pair_cov)

    return HedgeMetrics(
        ticker=ticker,
        full_sample_beta_gold=full_beta_gold,
        full_sample_corr_gold=full_corr_gold,
        tail_corr_gold=tail_corr_gold_val,
        tail_corr_krw=tail_corr_krw_val,
        tail_dependence_lower=tail_dep,
        avg_rolling_corr_gold=avg_rolling,
        diversification_ratio=dr_5050,
    )


# -----------------------------------------------------------------------------
# Model diagnostics
# -----------------------------------------------------------------------------

def residual_ljungbox_pvalue(
    gam_model: LinearGAM, design_matrix: np.ndarray, target_vector: np.ndarray,
) -> float:
    """Ljung-Box p-value (lag LJUNG_BOX_LAG) on GAM residuals."""
    residuals = target_vector - gam_model.predict(design_matrix)
    lb_table = acorr_ljungbox(residuals, lags=[LJUNG_BOX_LAG], return_df=True)
    return float(lb_table["lb_pvalue"].iloc[0])


def expanding_window_backtest(
    returns: pd.DataFrame,
    target_col: str,
    min_train_obs: int = BACKTEST_MIN_TRAIN_OBS,
    step: int = BACKTEST_STEP,
) -> float:
    """Out-of-sample pseudo R^2 from an expanding-window backtest.

    Inputs:
        returns       - DataFrame of log returns
        target_col    - dependant variable column name
        min_train_obs - minimum size of the initial training window
        step          - number of test observations consumed per refit
    Output:
        OOS pseudo R^2 = 1 - SSR / SST, or NaN if the sample is too short.
    Method:
        Repeatedly refits the GAM on data [:cursor], predicts the next `step`
        observations, slides the cursor forward, and accumulates predictions
        until the sample is exhausted.
    """
    design_matrix, target_vector, _ = build_gam_design(returns, target_col)
    n_total = len(target_vector)
    if n_total <= min_train_obs + step:
        return float("nan")
    predictions: List[float] = []
    actuals: List[float] = []
    cursor = min_train_obs
    while cursor + step <= n_total:
        backtest_model = LinearGAM(
            s(0, n_splines=GAM_N_SPLINES_BACKTEST)
            + s(1, n_splines=GAM_N_SPLINES_BACKTEST)
            + s(2, n_splines=GAM_N_SPLINES_BACKTEST_AUX)
            + s(3, n_splines=GAM_N_SPLINES_BACKTEST_AUX)
            + te(0, 1, n_splines=GAM_N_SPLINES_BACKTEST_TENSOR)
        ).fit(design_matrix[:cursor], target_vector[:cursor])
        predictions.extend(
            backtest_model.predict(design_matrix[cursor:cursor + step]).tolist()
        )
        actuals.extend(target_vector[cursor:cursor + step].tolist())
        cursor += step
    pred_arr = np.array(predictions)
    actual_arr = np.array(actuals)
    ss_residual = float(np.sum((actual_arr - pred_arr) ** 2))
    ss_total = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    if ss_total <= 0:
        return float("nan")
    return float(1 - ss_residual / ss_total)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, file_name: str) -> Path:
    """Save a matplotlib figure to OUTPUT_DIR at DEFAULT_DPI and close it."""
    output_path = OUTPUT_DIR / file_name
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return output_path


def plot_partial_dependence(
    gam_model: LinearGAM, feature_names: List[str], ticker: str,
) -> Path:
    """Partial-dependence panel for a fitted GAM.

    Plots each smooth term with a 95% confidence band and, in the final cell,
    a 2D contour of the gold x KRW tensor-product interaction. Saved as
    pdp_<ticker>.png.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    panel_titles = [
        f"f(r_gold)  -> r_{ticker}",
        f"f(r_krw)   -> r_{ticker}",
        f"f(krw_vol) -> r_{ticker}",
        f"f(ar1)     -> r_{ticker}",
        "te(r_gold, r_krw)",
    ]
    for term_idx, ax in enumerate(axes[:5]):
        term_grid = gam_model.generate_X_grid(term=term_idx)
        if term_idx == 4:
            try:
                mesh_grid = gam_model.generate_X_grid(term=term_idx, meshgrid=True)
                surface_vals = gam_model.partial_dependence(
                    term=term_idx, X=mesh_grid, meshgrid=True
                )
                contour_set = ax.contourf(
                    mesh_grid[0], mesh_grid[1], surface_vals,
                    levels=20, cmap="RdBu_r",
                )
                plt.colorbar(contour_set, ax=ax)
                ax.set_xlabel("r_gold")
                ax.set_ylabel("r_krw")
            except Exception as exc:
                ax.text(
                    0.5, 0.5, f"interaction plot failed:\n{exc}", ha="center"
                )
        else:
            pdep_vals, conf_band = gam_model.partial_dependence(
                term=term_idx, X=term_grid, width=GAM_CI_WIDTH,
            )
            ax.plot(term_grid[:, term_idx], pdep_vals)
            ax.fill_between(
                term_grid[:, term_idx],
                conf_band[:, 0], conf_band[:, 1], alpha=0.25,
            )
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel(feature_names[term_idx])
        ax.set_title(panel_titles[term_idx])
    axes[5].axis("off")
    fig.suptitle(f"GAM partial-dependance | r_{ticker}", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, f"pdp_{ticker}.png")


def plot_rolling_correlations(returns: pd.DataFrame) -> Path:
    """90-day rolling correlations of each semi (and KRW) versus gold."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker_name in SEMI_TICKERS:
        corr_series = rolling_corr(
            returns, f"r_{ticker_name}", "r_gold", ROLLING_CORR_WINDOW
        )
        ax.plot(corr_series.index, corr_series.values, label=f"{ticker_name} vs gold")
    krw_corr_series = rolling_corr(
        returns, "r_krw", "r_gold", ROLLING_CORR_WINDOW
    )
    ax.plot(
        krw_corr_series.index, krw_corr_series.values,
        label="KRW vs gold", linestyle="--", alpha=0.7,
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{ROLLING_CORR_WINDOW}-day rolling correlation with Gold")
    ax.set_ylabel("Pearson rho")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, "rolling_correlations.png")


def plot_price_panel(prices: pd.DataFrame) -> Path:
    """Plot all price series rebased to 100 at the start of the sample."""
    normalised = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    normalised.plot(ax=ax)
    ax.set_title("Normalised price series (start = 100)")
    ax.set_ylabel("Index")
    fig.tight_layout()
    return _save_figure(fig, "normalised_prices.png")


def plot_correlation_heatmap(returns: pd.DataFrame) -> Path:
    """Static full-sample Pearson and Spearman correlation matrices, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panel_titles = ["Pearson correlation", "Spearman rank correlation"]
    for ax, method_name, title in zip(axes, DEFAULT_CORR_METHODS, panel_titles):
        corr_matrix = returns.corr(method=method_name)
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, square=True, ax=ax, cbar=True,
        )
        ax.set_title(title)
    fig.suptitle("Return correlations (full sample)", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, "correlation_heatmap.png")


def plot_return_distributions(returns: pd.DataFrame) -> Path:
    """Two-row panel: KDE of returns above QQ-plots versus the normal."""
    series_names = list(returns.columns)
    n_series = len(series_names)
    fig, axes = plt.subplots(2, n_series, figsize=(3.2 * n_series, 6.5))
    for col_idx, col_name in enumerate(series_names):
        series_values = returns[col_name].dropna().values
        sns.kdeplot(
            series_values, ax=axes[0, col_idx], fill=True, color="steelblue"
        )
        axes[0, col_idx].axvline(0, color="k", lw=0.5)
        axes[0, col_idx].set_title(col_name)
        axes[0, col_idx].set_xlabel("")
        sm.qqplot(series_values, line="s", ax=axes[1, col_idx], markersize=2)
        axes[1, col_idx].set_title(f"QQ {col_name}")
    fig.suptitle("Return distributions and QQ-normal plots", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, "return_distributions.png")


def plot_rolling_volatility(
    returns: pd.DataFrame, window: int = ROLLING_VOL_WINDOW,
) -> Path:
    """Annualised rolling volatility for all return series."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(12, 5))
    rolling_vol.plot(ax=ax)
    ax.set_title(f"{window}-day annualised rolling volatility")
    ax.set_ylabel("Annualised vol")
    fig.tight_layout()
    return _save_figure(fig, "rolling_volatility.png")


def plot_drawdowns(prices: pd.DataFrame) -> Path:
    """Drawdown from running peak, one line per series."""
    drawdown = prices / prices.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown.plot(ax=ax)
    ax.set_title("Drawdown from running peak")
    ax.set_ylabel("Drawdown")
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return _save_figure(fig, "drawdowns.png")


def plot_cumulative_returns(returns: pd.DataFrame) -> Path:
    """Cumulative log-return curves for all series."""
    cumulative = returns.cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative.plot(ax=ax)
    ax.set_title("Cumulative log returns")
    ax.set_ylabel("Sum of log returns")
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return _save_figure(fig, "cumulative_returns.png")


def plot_rolling_beta(
    returns: pd.DataFrame, window: int = ROLLING_BETA_WINDOW,
) -> Path:
    """Rolling OLS beta of each semi (and KRW) on gold."""
    fig, ax = plt.subplots(figsize=(12, 5))
    gold_returns = returns["r_gold"]
    gold_variance = gold_returns.rolling(window).var()
    for ticker_name in SEMI_TICKERS + ["krw"]:
        series_values = returns[f"r_{ticker_name}"]
        rolling_cov = series_values.rolling(window).cov(gold_returns)
        rolling_beta = rolling_cov / gold_variance
        ax.plot(
            rolling_beta.index, rolling_beta.values,
            label=f"{ticker_name} vs gold",
            linestyle="--" if ticker_name == "krw" else "-",
            alpha=0.85 if ticker_name == "krw" else 1.0,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{window}-day rolling beta to Gold")
    ax.set_ylabel("Beta")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, "rolling_beta.png")


def plot_rolling_tail_correlation(
    returns: pd.DataFrame, window: int = ROLLING_TAIL_CORR_WINDOW,
) -> Path:
    """Rolling left-quartile correlation between semi and gold returns.

    Inside each rolling window, restricts both series to the worst-quartile
    semi days before computing Pearson correlation. This is the time-varying
    analogue of the static tail-correlation metric and helps reveal whether
    gold's hedging behaviour is stable or regime-dependant.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    gold_returns = returns["r_gold"]
    for ticker_name in SEMI_TICKERS:
        semi_returns = returns[f"r_{ticker_name}"]
        rolling_points: List[Tuple[pd.Timestamp, float]] = []
        date_index = semi_returns.index
        for window_end in range(window, len(semi_returns)):
            semi_window = semi_returns.iloc[window_end - window:window_end]
            gold_window = gold_returns.iloc[window_end - window:window_end]
            threshold = semi_window.quantile(LEFT_QUARTILE_QUANTILE)
            tail_mask = semi_window <= threshold
            if tail_mask.sum() > MIN_TAIL_OBS_FOR_ROLLING_CORR:
                rolling_points.append((
                    date_index[window_end],
                    semi_window[tail_mask].corr(gold_window[tail_mask]),
                ))
        if rolling_points:
            tail_corr_series = pd.Series(dict(rolling_points))
            ax.plot(
                tail_corr_series.index, tail_corr_series.values,
                label=f"{ticker_name} tail-corr vs gold",
            )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Rolling {window}d left-quartile correlation with Gold")
    ax.set_ylabel("Conditional rho")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, "rolling_tail_correlation.png")


def plot_tail_scatter(returns: pd.DataFrame, ticker: str) -> Path:
    """Scatter of semi vs gold returns coloured by KRW return.

    Worst-decile semi days are circled in black; the OLS fit is overlayed in
    red. Saves to tail_scatter_<ticker>.png.
    """
    semi_returns = returns[f"r_{ticker}"]
    gold_returns = returns["r_gold"]
    krw_returns = returns["r_krw"]
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_handle = ax.scatter(
        gold_returns, semi_returns,
        c=krw_returns, cmap="coolwarm",
        s=10, alpha=0.6,
        vmin=krw_returns.quantile(PLOT_COLOUR_QUANTILE_LOW),
        vmax=krw_returns.quantile(PLOT_COLOUR_QUANTILE_HIGH),
    )
    plt.colorbar(
        scatter_handle, ax=ax, label="r_krw (positive = KRW weakens)"
    )
    tail_mask = semi_returns <= semi_returns.quantile(WORST_DECILE_QUANTILE)
    ax.scatter(
        gold_returns[tail_mask], semi_returns[tail_mask],
        facecolors="none", edgecolors="black", s=35, linewidths=0.8,
        label="worst-decile semi days",
    )
    ols_coef = np.polyfit(gold_returns, semi_returns, 1)
    fit_xs = np.linspace(gold_returns.min(), gold_returns.max(), 100)
    ax.plot(
        fit_xs, ols_coef[0] * fit_xs + ols_coef[1],
        color="red", lw=1.5, label=f"OLS fit (beta={ols_coef[0]:+.3f})",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("r_gold")
    ax.set_ylabel(f"r_{ticker}")
    ax.set_title(f"Gold vs {ticker} returns, coloured by KRW")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, f"tail_scatter_{ticker}.png")


def plot_gam_fit_diagnostics(
    gam_model: LinearGAM,
    design_matrix: np.ndarray,
    target_vector: np.ndarray,
    aligned_index: pd.DatetimeIndex,
    ticker: str,
) -> Path:
    """Four-panel GAM diagnostic figure.

    Panels: actual vs predicted scatter, residuals over time, residual
    distribution with KDE, and the residual autocorrelation function up to
    RESIDUAL_ACF_LAGS lags.
    """
    predicted = gam_model.predict(design_matrix)
    residuals = target_vector - predicted

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    actual_vs_pred_ax = axes[0, 0]
    actual_vs_pred_ax.scatter(predicted, target_vector, s=6, alpha=0.4)
    axis_limit = max(abs(target_vector).max(), abs(predicted).max())
    actual_vs_pred_ax.plot(
        [-axis_limit, axis_limit], [-axis_limit, axis_limit],
        color="red", lw=1,
    )
    actual_vs_pred_ax.set_xlabel("Predicted")
    actual_vs_pred_ax.set_ylabel("Actual")
    actual_vs_pred_ax.set_title("Actual vs predicted")

    resid_time_ax = axes[0, 1]
    resid_time_ax.plot(aligned_index, residuals, lw=0.6, color="steelblue")
    resid_time_ax.axhline(0, color="k", lw=0.5)
    resid_time_ax.set_title("Residuals over time")

    resid_hist_ax = axes[1, 0]
    sns.histplot(residuals, bins=60, kde=True, ax=resid_hist_ax, color="steelblue")
    resid_hist_ax.axvline(0, color="k", lw=0.5)
    resid_hist_ax.set_title(
        f"Residual distribution (skew={stats.skew(residuals):+.2f}, "
        f"kurt={stats.kurtosis(residuals):+.2f})"
    )

    resid_acf_ax = axes[1, 1]
    plot_acf(residuals, lags=RESIDUAL_ACF_LAGS, ax=resid_acf_ax)
    resid_acf_ax.set_title(f"Residual ACF (lags 1-{RESIDUAL_ACF_LAGS})")

    fig.suptitle(f"GAM diagnostics | r_{ticker}", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, f"gam_diagnostics_{ticker}.png")


def plot_conditional_means(returns: pd.DataFrame) -> Path:
    """Mean gold and KRW return conditional on each semi's return decile.

    Useful for spotting whether gold systematically moves on semi-stress days,
    which is the empirical signature of a meaningfull tail hedge.
    """
    n_semi = len(SEMI_TICKERS)
    fig, axes = plt.subplots(1, n_semi, figsize=(6.5 * n_semi, 5), sharey=True)
    if n_semi == 1:
        axes = [axes]
    for ax, ticker_name in zip(axes, SEMI_TICKERS):
        semi_returns = returns[f"r_{ticker_name}"]
        decile_labels = pd.qcut(semi_returns, 10, labels=False)
        grouped_df = pd.DataFrame({
            "decile": decile_labels,
            "r_gold": returns["r_gold"].values,
            "r_krw": returns["r_krw"].values,
        })
        mean_by_decile = grouped_df.groupby("decile").mean() * BPS_SCALE
        bar_width = 0.4
        bar_positions = np.arange(10)
        ax.bar(
            bar_positions - bar_width / 2, mean_by_decile["r_gold"],
            width=bar_width, label="mean r_gold", color="goldenrod",
        )
        ax.bar(
            bar_positions + bar_width / 2, mean_by_decile["r_krw"],
            width=bar_width, label="mean r_krw (KRW per USD)", color="steelblue",
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"{ticker_name} return decile (0=worst, 9=best)")
        ax.set_ylabel("Mean return (bps)")
        ax.set_title(f"Conditional means | r_{ticker_name}")
        ax.legend()
    fig.suptitle(
        "Mean gold and KRW returns conditional on semi return decile",
        fontsize=14,
    )
    fig.tight_layout()
    return _save_figure(fig, "conditional_means.png")


def plot_efficient_frontier(returns: pd.DataFrame) -> Path:
    """Risk-return frontier of two-asset semi+gold portfolios across weights.

    For each semi ticker, sweeps gold weight from 0 to 1 across
    FRONTIER_N_WEIGHTS points and plots annualised vol vs annualised mean
    return. A handful of weights are labelled for orientation.
    """
    n_semi = len(SEMI_TICKERS)
    fig, axes = plt.subplots(1, n_semi, figsize=(6.5 * n_semi, 5))
    if n_semi == 1:
        axes = [axes]
    weight_grid = np.linspace(0, 1, FRONTIER_N_WEIGHTS)
    for ax, ticker_name in zip(axes, SEMI_TICKERS):
        pair_df = returns[[f"r_{ticker_name}", "r_gold"]].dropna()
        annualised_means = pair_df.mean().values * TRADING_DAYS_PER_YEAR
        annualised_cov = pair_df.cov().values * TRADING_DAYS_PER_YEAR
        portfolio_vols: List[float] = []
        portfolio_returns: List[float] = []
        for gold_weight in weight_grid:
            weight_vec = np.array([1 - gold_weight, gold_weight])
            portfolio_vols.append(
                float(np.sqrt(weight_vec @ annualised_cov @ weight_vec))
            )
            portfolio_returns.append(float(weight_vec @ annualised_means))
        ax.plot(portfolio_vols, portfolio_returns, "-o", markersize=3)
        for label_weight in FRONTIER_LABELLED_WEIGHTS:
            label_idx = int(label_weight * (len(weight_grid) - 1))
            ax.annotate(
                f"{int(label_weight * 100)}% gold",
                (portfolio_vols[label_idx], portfolio_returns[label_idx]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
        ax.set_xlabel("Annualised vol")
        ax.set_ylabel("Annualised mean return")
        ax.set_title(f"{ticker_name} + gold frontier")
    fig.suptitle("Two-asset risk-return frontier (semi + gold)", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, "efficient_frontier.png")


def plot_monthly_return_heatmap(returns: pd.DataFrame) -> Path:
    """Year x month heatmap of compounded monthly returns, one panel per series."""
    monthly_returns = np.exp(returns.resample("ME").sum()) - 1.0
    n_series = monthly_returns.shape[1]
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 2.6 * n_series))
    if n_series == 1:
        axes = [axes]
    for ax, col_name in zip(axes, monthly_returns.columns):
        col_series = monthly_returns[col_name]
        pivot_table = pd.DataFrame({
            "year": col_series.index.year,
            "month": col_series.index.month,
            "ret": col_series.values * 100,
        }).pivot(index="year", columns="month", values="ret")
        sns.heatmap(
            pivot_table, ax=ax, cmap="RdBu_r", center=0,
            annot=True, fmt=".1f", cbar=True, annot_kws={"size": 7},
        )
        ax.set_title(f"{col_name} monthly returns (%)")
    fig.suptitle("Monthly return heatmap by series", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, "monthly_return_heatmap.png")


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def _to_jsonable(obj):
    """Recursively convert dataclasses / numpy scalars into JSON-friendly types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _to_jsonable(val) for key, val in asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: _to_jsonable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(val) for val in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def run_pipeline(start_date: str, end_date: str) -> PipelineSummary:
    """End-to-end pipeline: fetch data, fit models, render visualisations.

    Inputs:
        start_date - ISO date string (inclusive)
        end_date   - ISO date string (exclusive)
    Output:
        Populated PipelineSummary, also persisted to pipeline_summary.json.
    Method:
        Performs data aquisition, stationarity diagnostics, descriptive stats,
        market-level plots, per-ticker GAM fits and diagnostic plots, hedge /
        diversification metric computation, and an expanding-window backtest.
        All artefacts (PNGs, CSVs, JSON) are written to OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_style(SEABORN_STYLE)

    prices = fetch_prices(start_date, end_date)
    prices.to_csv(OUTPUT_DIR / "prices.csv")
    returns = to_log_returns(prices)
    returns.to_csv(OUTPUT_DIR / "log_returns.csv")

    summary = PipelineSummary(
        run_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        start=str(prices.index.min().date()),
        end=str(prices.index.max().date()),
        n_obs=int(len(returns)),
    )

    log.info("Stationarity diagnostics...")
    summary.stationarity = stationarity_report(returns)

    log.info("Descriptive stats...")
    desc = descriptive_table(returns)
    desc.to_csv(OUTPUT_DIR / "descriptive_stats.csv")

    log.info("Generating market-level visualisations...")
    plot_price_panel(prices)
    plot_cumulative_returns(returns)
    plot_drawdowns(prices)
    plot_rolling_volatility(returns)
    plot_correlation_heatmap(returns)
    plot_rolling_correlations(returns)
    plot_rolling_tail_correlation(returns)
    plot_rolling_beta(returns)
    plot_return_distributions(returns)
    plot_conditional_means(returns)
    plot_efficient_frontier(returns)
    plot_monthly_return_heatmap(returns)
    for ticker_name in SEMI_TICKERS:
        plot_tail_scatter(returns, ticker_name)

    for ticker_name in SEMI_TICKERS:
        target_col = f"r_{ticker_name}"
        log.info("GAM fit for %s ...", ticker_name)
        design_matrix, target_vector, feature_names = build_gam_design(
            returns, target_col
        )
        gam_model = fit_gam(design_matrix, target_vector)

        term_pvalues = gam_significance(gam_model)
        lb_pvalue = residual_ljungbox_pvalue(
            gam_model, design_matrix, target_vector
        )
        pseudo_r2_val = float(
            gam_model.statistics_["pseudo_r2"]["explained_deviance"]
        )
        edof_val = float(gam_model.statistics_["edof"])
        aic_val = float(gam_model.statistics_["AIC"])

        summary.gam_results.append(GAMResult(
            ticker=ticker_name,
            pseudo_r2=pseudo_r2_val,
            edof=edof_val,
            aic=aic_val,
            gold_p=term_pvalues.get(0, np.nan),
            krw_p=term_pvalues.get(1, np.nan),
            interaction_p=term_pvalues.get(4, np.nan),
            residual_ljungbox_p=lb_pvalue,
            n_obs=int(len(target_vector)),
        ))

        plot_partial_dependence(gam_model, feature_names, ticker_name)
        # build_gam_design drops rows for the rolling KRW vol and the lag;
        # align the date index to the final length of the target vector.
        aligned_index = returns.index[-len(target_vector):]
        plot_gam_fit_diagnostics(
            gam_model, design_matrix, target_vector,
            aligned_index, ticker_name,
        )

        log.info("Hedge / diversification metrics for %s ...", ticker_name)
        summary.hedge_metrics.append(
            compute_hedge_metrics(returns, ticker_name)
        )

        log.info("Expanding-window backtest for %s ...", ticker_name)
        oos_pseudo_r2 = expanding_window_backtest(returns, target_col)
        summary.notes.append(
            f"oos_pseudo_r2[{ticker_name}] = {oos_pseudo_r2:.4f}"
        )

    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
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
    print(
        f"Window         : {summary.start} -> {summary.end}  "
        f"(n={summary.n_obs})"
    )
    print("\nGAM results:")
    for gam_row in summary.gam_results:
        print(
            f"  {gam_row.ticker:8s}  R^2={gam_row.pseudo_r2:.3f}  "
            f"edof={gam_row.edof:.1f}  AIC={gam_row.aic:.0f}  "
            f"p(gold)={gam_row.gold_p:.3f}  p(krw)={gam_row.krw_p:.3f}  "
            f"p(te)={gam_row.interaction_p:.3f}  "
            f"LB10 p={gam_row.residual_ljungbox_p:.3f}"
        )
    print("\nHedge / diversification metrics:")
    for hedge_row in summary.hedge_metrics:
        print(
            f"  {hedge_row.ticker:8s}  "
            f"beta_g={hedge_row.full_sample_beta_gold:+.3f}  "
            f"corr_g={hedge_row.full_sample_corr_gold:+.3f}  "
            f"tail_corr_g={hedge_row.tail_corr_gold:+.3f}  "
            f"tail_dep={hedge_row.tail_dependence_lower:.3f}  "
            f"DR(50/50)={hedge_row.diversification_ratio:.3f}"
        )
    for note in summary.notes:
        print(f"  note: {note}")
    print(f"\nArtefacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
