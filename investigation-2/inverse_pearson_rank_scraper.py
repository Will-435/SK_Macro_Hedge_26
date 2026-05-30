"""
inverse_pearson_rank_scraper.py

Scans a broad universe of major industry ETFs, broad index ETFs, country
equity ETFs, US / international bond yields and bond ETFs, plus a curated
set of cryptocurrency tokens and crypto-exposed equities, for assets whose
daily changes have a strong rank or linear correlation with the daily log
returns of the NASDAQ Composite. The intent is to surface both candidate
diversifiers (strongly inverse co-movers) and risk-substitutes (strongly
positive co-movers).

Both Pearson (linear) and Spearman (rank-based) correlations are computed
against NASDAQ over the full sample. The pipeline produces four ladder
diagrams over the top TOP_N_VISUAL_BARS assets in each direction and each
method:

    1. Top positive Spearman : assets that co-move most strongly with NASDAQ
                               in rank space.
    2. Top negative Spearman : assets that move most strongly against NASDAQ
                               in rank space.
    3. Top positive Pearson  : assets that co-move most strongly with NASDAQ
                               linearly.
    4. Top negative Pearson  : assets that move most strongly against NASDAQ
                               linearly.

A bearish NASDAQ regime correlation table (4-week return at or below the
10th percentile AND 21-day annualised realised volatility at or above the
75th percentile) is also saved alongside the full-sample table for
downstream pipeline use.

Spearman rank correlation is used because the universe mixes prices and
yields. Both Spearman and Pearson are computed on daily log returns for
price tickers and on daily first-difference changes for yield tickers so
the resulting numbers are directly comparable across asset categories.

Run:
    python inverse_pearson_rank_scraper.py --start 2015-01-01 --end 2026-05-01
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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Constants

# NASDAQ Composite is the correlation target. Daily log returns of ^IXIC
# are the reference series throughout the file.
NASDAQ_NAME = "nasdaq"
NASDAQ_SYMBOL = "^IXIC"

# Asset universe, grouped by category. Each entry maps a friendly name to a
# Yahoo Finance symbol. PRICE_TICKERS are converted to daily log returns;
# YIELD_TICKERS are converted to daily first-difference changes (basis-point
# moves on yields). Both transformations are then ranked into a Spearman
# correlation against the NASDAQ Composite.
PRICE_TICKERS: Dict[str, Dict[str, str]] = {
    "us_sector_etf": {
        "tech_sector":          "XLK",
        "financials_sector":    "XLF",
        "energy_sector":        "XLE",
        "healthcare_sector":    "XLV",
        "industrials_sector":   "XLI",
        "consumer_disc_sector": "XLY",
        "consumer_stap_sector": "XLP",
        "materials_sector":     "XLB",
        "utilities_sector":     "XLU",
        "real_estate_sector":   "XLRE",
        "comm_services_sector": "XLC",
    },
    "us_industry_etf": {
        "semis_ishares":        "SOXX",
        "semis_vaneck":         "SMH",
        "biotech_ishares":      "IBB",
        "biotech_equal_weight": "XBI",
        "regional_banks":       "KRE",
        "banks":                "KBE",
        "aerospace_defence":    "ITA",
        "homebuilders":         "XHB",
        "gold_miners":          "GDX",
        "oil_gas_exploration":  "XOP",
        "pipelines":            "AMLP",
        "airlines":             "JETS",
    },
    "broad_index_etf": {
        "sp500_spy":            "SPY",
        "sp500_voo":            "VOO",
        "dow_jones":            "DIA",
        "russell_2000":         "IWM",
        "total_us_market":      "VTI",
        "russell_1000":         "IWB",
        "russell_1000_growth":  "IWF",
        "russell_1000_value":   "IWD",
        "momentum_factor":      "MTUM",
        "quality_factor":       "QUAL",
    },
    "country_etf": {
        "japan":         "EWJ",
        "germany":       "EWG",
        "united_kingdom": "EWU",
        "canada":        "EWC",
        "switzerland":   "EWL",
        "south_korea":   "EWY",
        "australia":     "EWA",
        "france":        "EWQ",
        "netherlands":   "EWN",
        "developed_ex_us": "VEA",
        "emerging_markets": "VWO",
        "eafe":          "EFA",
        "all_country":   "ACWI",
    },
    "us_bond_etf": {
        "treasury_20y_plus":    "TLT",
        "treasury_7_10y":       "IEF",
        "treasury_3_7y":        "IEI",
        "treasury_1_3y":        "SHY",
        "us_aggregate_bond":    "AGG",
        "vanguard_total_bond":  "BND",
        "tips_inflation":       "TIP",
        "investment_grade_corp": "LQD",
        "high_yield_corp":      "HYG",
        "us_treasury_broad":    "GOVT",
    },
    "international_bond_etf": {
        "intl_treasury_bwx":      "BWX",
        "intl_treasury_igov":     "IGOV",
        "intl_aggregate_bndx":    "BNDX",
        "em_bonds_usd":           "EMB",
        "em_bonds_local":         "EMLC",
    },
    # Major cryptocurrencies. Yahoo Finance quotes cryptos with a '-USD'
    # suffix; tokens trade 24/7 but yfinance returns one row per UTC day so
    # the resulting series aligns with the equity / bond calendar after the
    # downstream forward-fill.
    "crypto_token": {
        "bitcoin":      "BTC-USD",
        "ethereum":     "ETH-USD",
        "solana":       "SOL-USD",
        "binance_coin": "BNB-USD",
        "ripple":       "XRP-USD",
        "cardano":      "ADA-USD",
        "dogecoin":     "DOGE-USD",
        "avalanche":    "AVAX-USD",
        "polkadot":     "DOT-USD",
        "chainlink":    "LINK-USD",
        "litecoin":     "LTC-USD",
    },
    # Crypto-exposed equities: miners, exchanges, the corporate-treasury
    # bitcoin proxy MicroStrategy, the spot-bitcoin ETFs that launched in
    # 2024, and broader blockchain-equity baskets.
    "crypto_equity": {
        "coinbase":             "COIN",
        "microstrategy":        "MSTR",
        "marathon_digital":     "MARA",
        "riot_platforms":       "RIOT",
        "hut_8_mining":         "HUT",
        "cleanspark":           "CLSK",
        "ishares_bitcoin_etf":  "IBIT",
        "grayscale_bitcoin":    "GBTC",
        "bitwise_crypto_innov": "BITQ",
        "blockchain_amplify":   "BLOK",
        "global_x_blockchain":  "BKCH",
    },
}
YIELD_TICKERS: Dict[str, Dict[str, str]] = {
    "us_bond_yield": {
        "us_10y_yield":  "^TNX",
        "us_5y_yield":   "^FVX",
        "us_30y_yield":  "^TYX",
        "us_3m_yield":   "^IRX",
    },
    # International sovereign yield coverage on Yahoo Finance is patchy.
    # The candidates below are tried defensively and silently skipped if
    # the request returns no data. The country bond ETFs above provide
    # broader, more reliable coverage for the swap-line countries.
    "international_bond_yield_candidate": {
        "germany_10y_yield":      "DE10Y.B",
        "uk_10y_yield":           "GB10Y.B",
        "japan_10y_yield":        "JP10Y.B",
        "canada_10y_yield":       "CA10Y.B",
        "switzerland_10y_yield":  "CH10Y.B",
        "korea_10y_yield":        "KR10Y.B",
        "australia_10y_yield":    "AU10Y.B",
    },
}

# Number of bars shown in each of the four ranked ladder diagrams. The top
# TOP_N_VISUAL_BARS most extreme correlations in each direction (positive
# and negative) for each method (Pearson and Spearman) are plotted. A
# fixed-count rather than threshold-based selection guarantees the charts
# are always populated and easy to compare side by side.
TOP_N_VISUAL_BARS = 20

# Calendar and window sizes.
TRADING_DAYS_PER_YEAR = 252
FOUR_WEEK_TRADING_DAYS = 20
REALISED_VOL_WINDOW = 21

# Bearish regime parameters, applied to the NASDAQ Composite.
BEARISH_QUANTILE = 0.10
HIGH_VOL_QUANTILE = 0.75

# Minimum number of overlapping observations needed to compute a
# correlation. Below this threshold the asset is skipped for that sample.
MIN_OBS_FOR_CORRELATION = 60

# Data alignment.
PANEL_FFILL_LIMIT = 2

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10
BAR_FIGSIZE_BASE = (10, 0.35)

# Plotting: colours. Unconventional red, green and blue tones; the
# matplotlib `color` keyword stays in US spelling because it is the
# library API.
PRIMARY_RED = "#C1272D"
PRIMARY_GREEN = "#386641"
PRIMARY_BLUE = "#1F4E79"
ACCENT_RED = "#E63946"
ACCENT_GREEN = "#52B788"
ACCENT_BLUE = "#457B9D"
DEEP_CRIMSON = "#7C1316"
DEEP_NAVY = "#0B2545"
NEUTRAL_DARK = "#222222"
NEUTRAL_GREY = "#5C5C5C"
CAPTION_COLOUR = "dimgray"

# One colour per asset category for the ladder diagrams.
CATEGORY_COLOURS: Dict[str, str] = {
    "us_sector_etf":                    PRIMARY_RED,
    "us_industry_etf":                  ACCENT_RED,
    "broad_index_etf":                  PRIMARY_BLUE,
    "country_etf":                      ACCENT_BLUE,
    "us_bond_yield":                    PRIMARY_GREEN,
    "us_bond_etf":                      ACCENT_GREEN,
    "international_bond_etf":           NEUTRAL_GREY,
    "international_bond_yield_candidate": NEUTRAL_DARK,
    "crypto_token":                     DEEP_CRIMSON,
    "crypto_equity":                    DEEP_NAVY,
}

# Human-readable category labels for legends and captions.
CATEGORY_DISPLAY_LABELS: Dict[str, str] = {
    "us_sector_etf":                    "US sector ETF",
    "us_industry_etf":                  "US industry ETF",
    "broad_index_etf":                  "Broad index ETF",
    "country_etf":                      "Country / international equity ETF",
    "us_bond_yield":                    "US Treasury yield",
    "us_bond_etf":                      "US bond ETF",
    "international_bond_etf":           "International bond ETF",
    "international_bond_yield_candidate": "International sovereign yield",
    "crypto_token":                     "Cryptocurrency token",
    "crypto_equity":                    "Crypto-exposed equity",
}

# Plotting: line widths, alphas, dpi.
REFERENCE_AXIS_LINEWIDTH = 0.5
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Plotting: ladder-diagram styling. The ladder is drawn as a horizontal
# lollipop where each rung is one asset: a stem from x = 0 to x = Spearman
# rho, with a marker on top of the stem at the rho value.
LADDER_STEM_LINEWIDTH = 1.8
LADDER_MARKER_SIZE = 90
LADDER_MARKER_EDGE_COLOUR = NEUTRAL_DARK
LADDER_MARKER_EDGE_LINEWIDTH = 0.6
LADDER_AXIS_X_MIN = -1.0
LADDER_AXIS_X_MAX = 1.0
LADDER_MIN_FIGURE_HEIGHT = 5.5
LADDER_HEIGHT_PER_RUNG = 0.34
LADDER_FIGURE_WIDTH = 11.0

# Plotting: caption layout.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18

# Filesystem layout. One sub-directory per script keeps each pipeline's
# outputs cleanly separated from any other pipeline in this repository.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "inverse_pearson_rank_scraper"
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
log = logging.getLogger("inverse-pearson-rank-scraper")


# Result containers (dataclasses).

@dataclass
class AssetSpec:
    """
    Lightweight specification for one asset in the scan universe. Tracks
    the friendly name, Yahoo Finance ticker, the category it belongs to,
    and whether it is a yield (first-difference) or a price (log-return)
    series.

    INPUTS:
        * name      : friendly identifier used in output tables
        * symbol    : Yahoo Finance ticker symbol
        * category  : universe category key
        * is_yield  : True for sovereign yield series; False for ETF prices

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    name: str
    symbol: str
    category: str
    is_yield: bool


@dataclass
class CorrelationRecord:
    """
    Pearson and Spearman correlation results for one (asset, sample) pair
    against the NASDAQ Composite. Written to the per-sample tables and
    used as the input to all four ranked ladder diagrams.

    INPUTS:
        * name      : friendly identifier
        * symbol    : Yahoo Finance ticker
        * category  : universe category key
        * n_obs     : number of overlapping observations used
        * pearson   : Pearson linear correlation versus NASDAQ
        * spearman  : Spearman rank correlation versus NASDAQ

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    name: str
    symbol: str
    category: str
    n_obs: int
    pearson: float
    spearman: float


@dataclass
class PipelineSummary:
    """
    Top-level run summary, persisted as JSON in the processed-data
    directory. Records the date window, the count of successfully fetched
    assets, and the number of bars rendered in each of the four ranked
    ladder diagrams.

    INPUTS:
        * run_at
        * start
        * end
        * universe_attempted
        * universe_fetched
        * top_n
        * notes

    OUTPUTS:
        * Dataclass aggregating the run metadata.
    """
    run_at: str
    start: str
    end: str
    universe_attempted: int
    universe_fetched: int
    top_n: int
    notes: List[str] = field(default_factory = list)


# Persistence helpers.

def save_parquet(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to Parquet under the supplied directory, creating
    the directory if it does not yet exist. Parquet is the preferred
    storage format for downstream pipeline reuse because it preserves
    dtypes and is faster to load than CSV.

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
    Write a DataFrame to CSV under the supplied directory. CSV is the
    primary deliverable format requested for the filtered correlation
    outputs in this scraper.

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


# Universe construction.

def build_asset_universe() -> List[AssetSpec]:
    """
    Flatten the PRICE_TICKERS and YIELD_TICKERS nested dictionaries into a
    single list of AssetSpec records so downstream loops can iterate over
    one homogeneous collection. The yield flag is carried through so the
    daily-change calculation knows whether to use log returns or first
    differences.

    INPUTS:
        * None (reads module-level PRICE_TICKERS and YIELD_TICKERS)

    OUTPUTS:
        * List of AssetSpec covering the full scan universe.
    """
    universe: List[AssetSpec] = []
    for category, members in PRICE_TICKERS.items():
        for friendly_name, yahoo_symbol in members.items():
            universe.append(AssetSpec(
                name = friendly_name,
                symbol = yahoo_symbol,
                category = category,
                is_yield = False,
            ))
    for category, members in YIELD_TICKERS.items():
        for friendly_name, yahoo_symbol in members.items():
            universe.append(AssetSpec(
                name = friendly_name,
                symbol = yahoo_symbol,
                category = category,
                is_yield = True,
            ))
    return universe


# Data acquisition.

def download_one_safe(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Attempt to download one Yahoo Finance ticker. Returns None on any
    failure or empty response so the calling loop can continue scraping
    the rest of the universe without aborting the whole run on a single
    missing symbol.

    INPUTS:
        * symbol      : Yahoo Finance ticker symbol
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * DataFrame of OHLCV-like columns, or None if no data is available.
    """
    try:
        raw_frame = yf.download(
            symbol,
            start = start_date,
            end = end_date,
            progress = False,
            auto_adjust = True,
        )
    except Exception as fetch_error:
        log.warning("Skipping %s (download error: %s)", symbol, fetch_error)
        return None
    if raw_frame is None or raw_frame.empty:
        return None
    if isinstance(raw_frame.columns, pd.MultiIndex):
        raw_frame.columns = raw_frame.columns.get_level_values(0)
    return raw_frame


def extract_close_series(raw_frame: pd.DataFrame, friendly_name: str) -> pd.Series:
    """
    Pull the close column out of a raw yfinance DataFrame and rename it to
    the friendly identifier so that the resulting panel column is easy to
    identify downstream.

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


def fetch_universe_panel(
    universe: List[AssetSpec], start_date: str, end_date: str,
) -> Tuple[pd.DataFrame, List[AssetSpec]]:
    """
    Download every asset in the universe defensively. Raw frames are
    cached as Parquet under DATA_RAW_DIR. Symbols that yfinance does not
    return data for are logged and skipped; the returned panel only
    contains assets that produced data.

    INPUTS:
        * universe    : list of AssetSpec records to attempt
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Tuple of (panel DataFrame, list of AssetSpec records that
          fetched successfully). The DataFrame contains one column per
          successful asset, indexed by date.
    """
    fetched_series: List[pd.Series] = []
    fetched_specs: List[AssetSpec] = []

    for asset_spec in universe:
        raw_frame = download_one_safe(asset_spec.symbol, start_date, end_date)
        if raw_frame is None:
            log.info("No data for %s (%s); skipping",
                     asset_spec.name, asset_spec.symbol)
            continue
        save_parquet(raw_frame, DATA_RAW_DIR, f"raw_{asset_spec.name}")
        close_series = extract_close_series(raw_frame, asset_spec.name)
        fetched_series.append(close_series)
        fetched_specs.append(asset_spec)

    if len(fetched_series) == 0:
        return pd.DataFrame(), []

    panel = pd.concat(fetched_series, axis = 1).dropna(how = "all")
    panel = panel.ffill(limit = PANEL_FFILL_LIMIT)

    log.info(
        "Fetched %d of %d universe assets",
        len(fetched_specs), len(universe),
    )
    return panel, fetched_specs


# Returns and yield changes.

def to_daily_changes(panel: pd.DataFrame, asset_specs: List[AssetSpec]) -> pd.DataFrame:
    """
    Convert the price + yield panel into a single daily-change frame.
    Price columns become log returns; yield columns become first
    differences. Spearman rank correlation is invariant to monotone
    transformations so the two scales can sit alongside each other in
    the same frame without distorting rank-based comparisons downstream.

    INPUTS:
        * panel        : DataFrame of close prices and yields
        * asset_specs  : list of AssetSpec matching panel.columns

    OUTPUTS:
        * DataFrame of daily changes; one column per input column.
    """
    spec_by_name: Dict[str, AssetSpec] = {}
    for asset_spec in asset_specs:
        spec_by_name[asset_spec.name] = asset_spec

    change_columns: List[pd.Series] = []
    for column_name in panel.columns:
        column_spec = spec_by_name.get(column_name)
        if column_spec is None:
            continue
        column_series = panel[column_name]
        if column_spec.is_yield:
            change_series = column_series.diff()
        else:
            change_series = np.log(column_series).diff()
        change_columns.append(change_series.rename(column_name))

    changes = pd.concat(change_columns, axis = 1).dropna(how = "all")
    return changes


# Bearish regime mask.

def build_nasdaq_regime_mask(nasdaq_prices: pd.Series) -> pd.Series:
    """
    Build the bearish NASDAQ regime mask. A trading day is in the regime
    when the 4-week (20 trading day) log return is at or below the 10th
    percentile of the full sample AND the 21-day annualised realised
    volatility is at or above the 75th percentile of the full sample.
    Both thresholds are computed once on the full sample so the mask is a
    fixed reference rather than a rolling one.

    INPUTS:
        * nasdaq_prices  : daily close prices of the NASDAQ Composite

    OUTPUTS:
        * Boolean Series aligned to nasdaq_prices.index; True on regime days.
    """
    log_prices = np.log(nasdaq_prices)
    four_week_return = log_prices - log_prices.shift(FOUR_WEEK_TRADING_DAYS)
    daily_returns = log_prices.diff()
    realised_vol = daily_returns.rolling(REALISED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    bearish_threshold = four_week_return.quantile(BEARISH_QUANTILE)
    high_vol_threshold = realised_vol.quantile(HIGH_VOL_QUANTILE)

    bearish = four_week_return <= bearish_threshold
    high_vol = realised_vol >= high_vol_threshold
    regime = (bearish & high_vol).fillna(False)
    return regime


# Spearman correlation computation.

def compute_correlations_against_nasdaq(
    changes: pd.DataFrame,
    asset_specs: List[AssetSpec],
    mask: Optional[pd.Series] = None,
) -> List[CorrelationRecord]:
    """
    Compute both the Pearson and Spearman correlation between the NASDAQ
    daily log return column and every other column in the changes frame.
    If a boolean mask is supplied, the correlations are computed only on
    rows where the mask is True; this is how the bearish regime variant
    is produced. Assets with fewer than MIN_OBS_FOR_CORRELATION overlapping
    observations are skipped so noisy correlations do not pollute the
    output.

    INPUTS:
        * changes      : DataFrame of daily changes
        * asset_specs  : list of AssetSpec for the columns
        * mask         : optional boolean Series; if supplied, rows where
                         the mask is False are dropped before correlating

    OUTPUTS:
        * List of CorrelationRecord, one per asset that met the minimum-
          observation threshold.
    """
    nasdaq_column = NASDAQ_NAME
    if nasdaq_column not in changes.columns:
        raise RuntimeError(
            f"NASDAQ column '{nasdaq_column}' missing from changes frame"
        )

    if mask is not None:
        aligned_mask = mask.reindex(changes.index).fillna(False).astype(bool)
        working_frame = changes.loc[aligned_mask]
    else:
        working_frame = changes

    nasdaq_series = working_frame[nasdaq_column]

    spec_by_name: Dict[str, AssetSpec] = {}
    for asset_spec in asset_specs:
        spec_by_name[asset_spec.name] = asset_spec

    records: List[CorrelationRecord] = []
    for column_name in working_frame.columns:
        if column_name == nasdaq_column:
            continue
        column_spec = spec_by_name.get(column_name)
        if column_spec is None:
            continue
        paired = pd.concat(
            [nasdaq_series, working_frame[column_name]], axis = 1,
        ).dropna()
        if len(paired) < MIN_OBS_FOR_CORRELATION:
            continue
        pearson_value = float(
            paired.iloc[:, 0].corr(paired.iloc[:, 1], method = "pearson")
        )
        spearman_value = float(
            paired.iloc[:, 0].corr(paired.iloc[:, 1], method = "spearman")
        )
        records.append(CorrelationRecord(
            name = column_spec.name,
            symbol = column_spec.symbol,
            category = column_spec.category,
            n_obs = int(len(paired)),
            pearson = pearson_value,
            spearman = spearman_value,
        ))
    return records


def select_top_n_by_attribute(
    records: List[CorrelationRecord],
    attribute_name: str,
    top_n: int,
    descending: bool,
) -> List[CorrelationRecord]:
    """
    Return the TOP N records ranked by one of the correlation fields. When
    descending is True the largest positive values come first; when False
    the most negative values come first. Centralising the sort here keeps
    each of the four ranked-plot functions standalone in its drawing logic
    while reusing a single, well-tested selection step.

    INPUTS:
        * records         : list of CorrelationRecord candidates
        * attribute_name  : 'pearson' or 'spearman'
        * top_n           : maximum number of records to return
        * descending      : True for largest-positive, False for largest-
                            negative

    OUTPUTS:
        * Ranked list of at most top_n CorrelationRecord, sorted so that
          the most extreme value sits first.
    """
    sorted_records = sorted(
        records,
        key = lambda record: getattr(record, attribute_name),
        reverse = descending,
    )
    return sorted_records[:top_n]


def records_to_frame(records: List[CorrelationRecord]) -> pd.DataFrame:
    """
    Convert a list of CorrelationRecord into a tidy DataFrame for CSV /
    Parquet output and for plotting. Columns: name, symbol, category,
    n_obs, pearson, spearman.

    INPUTS:
        * records  : list of CorrelationRecord

    OUTPUTS:
        * DataFrame with one row per record.
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append({
            "name":     record.name,
            "symbol":   record.symbol,
            "category": record.category,
            "n_obs":    record.n_obs,
            "pearson":  record.pearson,
            "spearman": record.spearman,
        })
    return pd.DataFrame(rows)


# Plotting helpers.

def add_caption(fig: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption underneath the figure so each saved PNG
    carries its own self-contained explanation of axes, units and any
    abbreviations used in the chart.

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
    Save a matplotlib figure under VISUALS_DIR at the project default
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


def plot_top_positive_spearman(records: List[CorrelationRecord]) -> Optional[Path]:
    """
    Render the TOP_N_VISUAL_BARS assets with the largest POSITIVE Spearman
    rank correlations against NASDAQ as a horizontal ladder diagram. The
    strongest co-mover sits at the top; each rung is coloured by its asset
    category and a legend in the lower right names the categories present
    in the chart.

    INPUTS:
        * records  : list of CorrelationRecord covering the scan universe

    OUTPUTS:
        * Path to the saved PNG, or None if no records are supplied.
    """
    if len(records) == 0:
        log.info("Top positive Spearman: no records; skipping plot")
        return None

    top_records = select_top_n_by_attribute(
        records, "spearman", TOP_N_VISUAL_BARS, descending = True,
    )
    # Reverse so the largest positive sits at the top of the y axis.
    chart_records = list(reversed(top_records))

    rung_labels: List[str] = []
    rung_values: List[float] = []
    rung_colours: List[str] = []
    present_categories: List[str] = []
    for record in chart_records:
        rung_labels.append(f"{record.name} ({record.symbol})")
        rung_values.append(record.spearman)
        rung_colours.append(CATEGORY_COLOURS.get(record.category, NEUTRAL_DARK))
        if record.category not in present_categories:
            present_categories.append(record.category)

    figure_height = max(
        LADDER_MIN_FIGURE_HEIGHT,
        LADDER_HEIGHT_PER_RUNG * len(chart_records),
    )
    fig, ax = plt.subplots(figsize = (LADDER_FIGURE_WIDTH, figure_height))
    rung_positions = np.arange(len(chart_records))

    ax.hlines(
        y = rung_positions, xmin = 0.0, xmax = rung_values,
        colors = rung_colours, linewidth = LADDER_STEM_LINEWIDTH,
    )
    ax.scatter(
        rung_values, rung_positions,
        c = rung_colours,
        s = LADDER_MARKER_SIZE,
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 3,
    )
    ax.set_yticks(rung_positions)
    ax.set_yticklabels(rung_labels, fontsize = TEXT_FONT)
    ax.axvline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_xlim(left = LADDER_AXIS_X_MIN, right = LADDER_AXIS_X_MAX)
    ax.set_xlabel("Spearman rank correlation vs NASDAQ")
    ax.set_title(
        f"Top {TOP_N_VISUAL_BARS} largest POSITIVE Spearman correlations "
        "vs NASDAQ"
    )

    legend_handles: List[Patch] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = CATEGORY_DISPLAY_LABELS.get(category_key, category_key),
        ))
    ax.legend(
        handles = legend_handles,
        loc = "lower right",
        fontsize = TEXT_FONT,
        title = "Asset category",
    )

    fig.tight_layout()
    add_caption(
        fig,
        "Ladder diagram of the assets whose Spearman rank correlation with "
        "the NASDAQ Composite over the full sample is the largest positive. "
        "These are the assets that move most strongly in the same direction "
        "as NASDAQ in rank space - effectively NASDAQ substitutes rather "
        "than diversifiers. Marker colour encodes the asset category.",
    )
    return save_figure(fig, "top_positive_spearman.png")


def plot_top_negative_spearman(records: List[CorrelationRecord]) -> Optional[Path]:
    """
    Render the TOP_N_VISUAL_BARS assets with the largest NEGATIVE Spearman
    rank correlations against NASDAQ as a horizontal ladder diagram. The
    most strongly inverse co-mover sits at the bottom; each rung is
    coloured by its asset category and a legend in the upper right names
    the categories present in the chart.

    INPUTS:
        * records  : list of CorrelationRecord covering the scan universe

    OUTPUTS:
        * Path to the saved PNG, or None if no records are supplied.
    """
    if len(records) == 0:
        log.info("Top negative Spearman: no records; skipping plot")
        return None

    top_records = select_top_n_by_attribute(
        records, "spearman", TOP_N_VISUAL_BARS, descending = False,
    )
    # Reverse so the most negative sits at the bottom of the y axis.
    chart_records = list(reversed(top_records))

    rung_labels: List[str] = []
    rung_values: List[float] = []
    rung_colours: List[str] = []
    present_categories: List[str] = []
    for record in chart_records:
        rung_labels.append(f"{record.name} ({record.symbol})")
        rung_values.append(record.spearman)
        rung_colours.append(CATEGORY_COLOURS.get(record.category, NEUTRAL_DARK))
        if record.category not in present_categories:
            present_categories.append(record.category)

    figure_height = max(
        LADDER_MIN_FIGURE_HEIGHT,
        LADDER_HEIGHT_PER_RUNG * len(chart_records),
    )
    fig, ax = plt.subplots(figsize = (LADDER_FIGURE_WIDTH, figure_height))
    rung_positions = np.arange(len(chart_records))

    ax.hlines(
        y = rung_positions, xmin = 0.0, xmax = rung_values,
        colors = rung_colours, linewidth = LADDER_STEM_LINEWIDTH,
    )
    ax.scatter(
        rung_values, rung_positions,
        c = rung_colours,
        s = LADDER_MARKER_SIZE,
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 3,
    )
    ax.set_yticks(rung_positions)
    ax.set_yticklabels(rung_labels, fontsize = TEXT_FONT)
    ax.axvline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_xlim(left = LADDER_AXIS_X_MIN, right = LADDER_AXIS_X_MAX)
    ax.set_xlabel("Spearman rank correlation vs NASDAQ")
    ax.set_title(
        f"Top {TOP_N_VISUAL_BARS} largest NEGATIVE Spearman correlations "
        "vs NASDAQ"
    )

    legend_handles: List[Patch] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = CATEGORY_DISPLAY_LABELS.get(category_key, category_key),
        ))
    ax.legend(
        handles = legend_handles,
        loc = "upper right",
        fontsize = TEXT_FONT,
        title = "Asset category",
    )

    fig.tight_layout()
    add_caption(
        fig,
        "Ladder diagram of the assets whose Spearman rank correlation with "
        "the NASDAQ Composite over the full sample is the largest negative. "
        "These are the assets that move most strongly against NASDAQ in "
        "rank space - candidate diversifiers and hedges. Marker colour "
        "encodes the asset category.",
    )
    return save_figure(fig, "top_negative_spearman.png")


def plot_top_positive_pearson(records: List[CorrelationRecord]) -> Optional[Path]:
    """
    Render the TOP_N_VISUAL_BARS assets with the largest POSITIVE Pearson
    linear correlations against NASDAQ as a horizontal ladder diagram. The
    strongest linear co-mover sits at the top; each rung is coloured by
    its asset category and a legend in the lower right names the
    categories present in the chart.

    INPUTS:
        * records  : list of CorrelationRecord covering the scan universe

    OUTPUTS:
        * Path to the saved PNG, or None if no records are supplied.
    """
    if len(records) == 0:
        log.info("Top positive Pearson: no records; skipping plot")
        return None

    top_records = select_top_n_by_attribute(
        records, "pearson", TOP_N_VISUAL_BARS, descending = True,
    )
    chart_records = list(reversed(top_records))

    rung_labels: List[str] = []
    rung_values: List[float] = []
    rung_colours: List[str] = []
    present_categories: List[str] = []
    for record in chart_records:
        rung_labels.append(f"{record.name} ({record.symbol})")
        rung_values.append(record.pearson)
        rung_colours.append(CATEGORY_COLOURS.get(record.category, NEUTRAL_DARK))
        if record.category not in present_categories:
            present_categories.append(record.category)

    figure_height = max(
        LADDER_MIN_FIGURE_HEIGHT,
        LADDER_HEIGHT_PER_RUNG * len(chart_records),
    )
    fig, ax = plt.subplots(figsize = (LADDER_FIGURE_WIDTH, figure_height))
    rung_positions = np.arange(len(chart_records))

    ax.hlines(
        y = rung_positions, xmin = 0.0, xmax = rung_values,
        colors = rung_colours, linewidth = LADDER_STEM_LINEWIDTH,
    )
    ax.scatter(
        rung_values, rung_positions,
        c = rung_colours,
        s = LADDER_MARKER_SIZE,
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 3,
    )
    ax.set_yticks(rung_positions)
    ax.set_yticklabels(rung_labels, fontsize = TEXT_FONT)
    ax.axvline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_xlim(left = LADDER_AXIS_X_MIN, right = LADDER_AXIS_X_MAX)
    ax.set_xlabel("Pearson correlation vs NASDAQ")
    ax.set_title(
        f"Top {TOP_N_VISUAL_BARS} largest POSITIVE Pearson correlations "
        "vs NASDAQ"
    )

    legend_handles: List[Patch] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = CATEGORY_DISPLAY_LABELS.get(category_key, category_key),
        ))
    ax.legend(
        handles = legend_handles,
        loc = "lower right",
        fontsize = TEXT_FONT,
        title = "Asset category",
    )

    fig.tight_layout()
    add_caption(
        fig,
        "Ladder diagram of the assets whose Pearson linear correlation "
        "with the NASDAQ Composite over the full sample is the largest "
        "positive. Pearson responds to magnitudes as well as ordering, so "
        "the top of this chart highlights assets that move both in the "
        "same direction as NASDAQ and by similar daily magnitudes. Marker "
        "colour encodes the asset category.",
    )
    return save_figure(fig, "top_positive_pearson.png")


def plot_top_negative_pearson(records: List[CorrelationRecord]) -> Optional[Path]:
    """
    Render the TOP_N_VISUAL_BARS assets with the largest NEGATIVE Pearson
    linear correlations against NASDAQ as a horizontal ladder diagram. The
    most strongly linear inverse co-mover sits at the bottom; each rung is
    coloured by its asset category and a legend in the upper right names
    the categories present in the chart.

    INPUTS:
        * records  : list of CorrelationRecord covering the scan universe

    OUTPUTS:
        * Path to the saved PNG, or None if no records are supplied.
    """
    if len(records) == 0:
        log.info("Top negative Pearson: no records; skipping plot")
        return None

    top_records = select_top_n_by_attribute(
        records, "pearson", TOP_N_VISUAL_BARS, descending = False,
    )
    chart_records = list(reversed(top_records))

    rung_labels: List[str] = []
    rung_values: List[float] = []
    rung_colours: List[str] = []
    present_categories: List[str] = []
    for record in chart_records:
        rung_labels.append(f"{record.name} ({record.symbol})")
        rung_values.append(record.pearson)
        rung_colours.append(CATEGORY_COLOURS.get(record.category, NEUTRAL_DARK))
        if record.category not in present_categories:
            present_categories.append(record.category)

    figure_height = max(
        LADDER_MIN_FIGURE_HEIGHT,
        LADDER_HEIGHT_PER_RUNG * len(chart_records),
    )
    fig, ax = plt.subplots(figsize = (LADDER_FIGURE_WIDTH, figure_height))
    rung_positions = np.arange(len(chart_records))

    ax.hlines(
        y = rung_positions, xmin = 0.0, xmax = rung_values,
        colors = rung_colours, linewidth = LADDER_STEM_LINEWIDTH,
    )
    ax.scatter(
        rung_values, rung_positions,
        c = rung_colours,
        s = LADDER_MARKER_SIZE,
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 3,
    )
    ax.set_yticks(rung_positions)
    ax.set_yticklabels(rung_labels, fontsize = TEXT_FONT)
    ax.axvline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)
    ax.set_xlim(left = LADDER_AXIS_X_MIN, right = LADDER_AXIS_X_MAX)
    ax.set_xlabel("Pearson correlation vs NASDAQ")
    ax.set_title(
        f"Top {TOP_N_VISUAL_BARS} largest NEGATIVE Pearson correlations "
        "vs NASDAQ"
    )

    legend_handles: List[Patch] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = CATEGORY_DISPLAY_LABELS.get(category_key, category_key),
        ))
    ax.legend(
        handles = legend_handles,
        loc = "upper right",
        fontsize = TEXT_FONT,
        title = "Asset category",
    )

    fig.tight_layout()
    add_caption(
        fig,
        "Ladder diagram of the assets whose Pearson linear correlation "
        "with the NASDAQ Composite over the full sample is the largest "
        "negative. Pearson responds to magnitudes as well as ordering, so "
        "the bottom of this chart highlights assets that move opposite to "
        "NASDAQ and by similar daily magnitudes - the strongest candidates "
        "for a linear hedge. Marker colour encodes the asset category.",
    )
    return save_figure(fig, "top_negative_pearson.png")


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
    Apply project-wide matplotlib defaults: seaborn whitegrid background,
    consistent font sizes, and a red / green / blue colour cycler so
    pandas .plot calls fall back to the same palette as our explicit
    colour assignments.

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
    Execute the end-to-end scrape: fetch the NASDAQ Composite and the
    entire candidate universe (including crypto tokens and crypto-exposed
    equities), convert prices to log returns and yields to first
    differences, compute Pearson and Spearman correlations versus NASDAQ
    over the full sample and over the bearish NASDAQ regime, select the
    top TOP_N_VISUAL_BARS records in each direction and each method,
    persist all of the ranked lists as CSV (primary deliverable) and
    Parquet (for downstream reuse), and render the four ranked ladder
    diagrams.

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

    log.info("Fetching the NASDAQ Composite (%s)", NASDAQ_SYMBOL)
    nasdaq_raw = download_one_safe(NASDAQ_SYMBOL, start_date, end_date)
    if nasdaq_raw is None:
        raise RuntimeError("Could not fetch the NASDAQ Composite; aborting")
    save_parquet(nasdaq_raw, DATA_RAW_DIR, f"raw_{NASDAQ_NAME}")
    nasdaq_close = extract_close_series(nasdaq_raw, NASDAQ_NAME)

    log.info("Fetching the candidate scan universe")
    universe = build_asset_universe()
    universe_panel, fetched_specs = fetch_universe_panel(
        universe, start_date, end_date,
    )

    full_panel = pd.concat([nasdaq_close, universe_panel], axis = 1)
    full_panel = full_panel.dropna(how = "all").ffill(limit = PANEL_FFILL_LIMIT)
    save_parquet(full_panel, DATA_PROCESSED_DIR, "prices_and_yields")

    nasdaq_spec = AssetSpec(
        name = NASDAQ_NAME, symbol = NASDAQ_SYMBOL,
        category = "target", is_yield = False,
    )
    all_specs = [nasdaq_spec] + fetched_specs
    changes = to_daily_changes(full_panel, all_specs)
    save_parquet(changes, DATA_PROCESSED_DIR, "daily_changes")

    log.info("Building the NASDAQ bearish regime mask")
    regime_mask = build_nasdaq_regime_mask(nasdaq_close)
    save_parquet(
        regime_mask.to_frame(name = "regime"),
        DATA_PROCESSED_DIR,
        "nasdaq_regime_mask",
    )

    log.info("Computing Pearson and Spearman correlations across the full sample")
    full_records = compute_correlations_against_nasdaq(changes, fetched_specs)
    full_frame_all = records_to_frame(
        sorted(full_records, key = lambda record: record.spearman, reverse = True)
    )
    save_parquet(full_frame_all, DATA_PROCESSED_DIR, "correlations_all_full_sample")
    save_csv(full_frame_all, DATA_PROCESSED_DIR, "correlations_all_full_sample")

    log.info("Computing Pearson and Spearman correlations on regime days")
    regime_records = compute_correlations_against_nasdaq(
        changes, fetched_specs, mask = regime_mask,
    )
    regime_frame_all = records_to_frame(
        sorted(regime_records, key = lambda record: record.spearman, reverse = True)
    )
    save_parquet(regime_frame_all, DATA_PROCESSED_DIR, "correlations_all_regime")
    save_csv(regime_frame_all, DATA_PROCESSED_DIR, "correlations_all_regime")

    log.info("Selecting top %d records in each direction and method", TOP_N_VISUAL_BARS)
    top_positive_spearman_records = select_top_n_by_attribute(
        full_records, "spearman", TOP_N_VISUAL_BARS, descending = True,
    )
    top_negative_spearman_records = select_top_n_by_attribute(
        full_records, "spearman", TOP_N_VISUAL_BARS, descending = False,
    )
    top_positive_pearson_records = select_top_n_by_attribute(
        full_records, "pearson", TOP_N_VISUAL_BARS, descending = True,
    )
    top_negative_pearson_records = select_top_n_by_attribute(
        full_records, "pearson", TOP_N_VISUAL_BARS, descending = False,
    )

    save_csv(records_to_frame(top_positive_spearman_records),
             DATA_PROCESSED_DIR, "top_positive_spearman")
    save_csv(records_to_frame(top_negative_spearman_records),
             DATA_PROCESSED_DIR, "top_negative_spearman")
    save_csv(records_to_frame(top_positive_pearson_records),
             DATA_PROCESSED_DIR, "top_positive_pearson")
    save_csv(records_to_frame(top_negative_pearson_records),
             DATA_PROCESSED_DIR, "top_negative_pearson")
    save_parquet(records_to_frame(top_positive_spearman_records),
                 DATA_PROCESSED_DIR, "top_positive_spearman")
    save_parquet(records_to_frame(top_negative_spearman_records),
                 DATA_PROCESSED_DIR, "top_negative_spearman")
    save_parquet(records_to_frame(top_positive_pearson_records),
                 DATA_PROCESSED_DIR, "top_positive_pearson")
    save_parquet(records_to_frame(top_negative_pearson_records),
                 DATA_PROCESSED_DIR, "top_negative_pearson")

    log.info("Plotting top positive Spearman ladder")
    plot_top_positive_spearman(full_records)

    log.info("Plotting top negative Spearman ladder")
    plot_top_negative_spearman(full_records)

    log.info("Plotting top positive Pearson ladder")
    plot_top_positive_pearson(full_records)

    log.info("Plotting top negative Pearson ladder")
    plot_top_negative_pearson(full_records)

    summary = PipelineSummary(
        run_at = datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        start = str(full_panel.index.min().date()),
        end = str(full_panel.index.max().date()),
        universe_attempted = len(universe),
        universe_fetched = len(fetched_specs),
        top_n = TOP_N_VISUAL_BARS,
    )
    summary.notes.append(
        "International sovereign bond yield coverage on Yahoo Finance is "
        "sparse; international bond exposure is therefore represented "
        "primarily through bond ETFs (BWX, IGOV, BNDX, EMB, EMLC). Country "
        "equity ETFs cover swap-line countries plus South Korea on the "
        "equity side."
    )
    summary.notes.append(
        "Crypto tokens are quoted on a 7-day calendar by Yahoo Finance but "
        "are aligned here to the equity / bond trading calendar via forward "
        "fill, which introduces a mild attenuation of the daily correlation "
        "estimate during equity holidays."
    )
    summary.notes.append(
        "Pearson and Spearman are computed on daily log returns for price "
        "tickers and on daily first differences for yield tickers. Spearman "
        "is invariant to monotone transformations so the two scales are "
        "directly comparable in rank space; Pearson is sensitive to "
        "magnitudes and to outliers."
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
    Entry point. Runs the full scrape pipeline and prints a one-screen
    summary of how many universe tickers were attempted, how many fetched
    successfully, and how many cleared the Spearman threshold in each
    sample variant.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: figures in VISUALS_DIR, Parquet tables and
          CSV deliverables in DATA_PROCESSED_DIR, raw caches in
          DATA_RAW_DIR, JSON summary in DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\nPipeline summary")
    print(f"Window               : {summary.start} -> {summary.end}")
    print(f"Universe attempted   : {summary.universe_attempted}")
    print(f"Universe fetched ok  : {summary.universe_fetched}")
    print(f"Top N per visual     : {summary.top_n}")
    for note in summary.notes:
        print(f"\n  note: {note}")
    print(f"\nVisuals in : {VISUALS_DIR}")
    print(f"Data in    : {DATA_PROCESSED_DIR}")
    print(f"Raw in     : {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
