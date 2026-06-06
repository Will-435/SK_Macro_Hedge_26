"""
skhynix_correlation_ladder.py

Investigation-3. Same scan idea as in investigation-2 NASDAQ
scraper, but with SK Hynix as the reference instead of the NASDAQ.
Ranks a set of individual assets by the strength of
their daily co-movement with SK Hynix and renders the result as a single
horizontal lollipop ladder of Spearman rank correlations.

Investigation-2 centres on US index funds and assets; investigation-3 centres
purely on SK Hynix. This file is self-contained: the shared universe, the
defensive download helpers and the price-to-change transform are copied in
rather than imported, so the two investigations evolve independently.

USD AGAINST SK HYNIX

SK Hynix is quoted in KRW, but semiconductors are overwhelmingly priced and
contracted in USD, so dollar strength should still impact stock price.
Two distinct dollar measures are carried and both are pinned onto the ladder:

    * dxy (DX-Y.NYB) : the broad US dollar index against major currencies.
                       This is the clean "USD" reading the investigation-2
                       work never isolated.
    * usd_krw (KRW=X): the won pair. Already studied, jus used for continuity.
                       It conflates broad-dollar moves with won-specific moves,
                       which is exactly why the broad index is added alongside.

Sign convention:
    DXY and USD/KRW both rise when the dollar strengthens. A negative Spearman
    against SK Hynix therefore means SK Hynix tends to fall on dollar-strength,
    risk-off days. USD/KRW is quoted KRW per USD, so a positive return on it is
    won depreciation.

Spearman rank correlation is the headline statistic because the set of candidate hedges
mixes price returns and yield changes, rank correlation is invariant to the 
difference in their units. Price tickers enter as daily log returns and yield
tickers as daily first differences.
"""

# Imports
from __future__ import annotations
import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf


# Constants

# SK Hynix is the correlation reference. Its daily log returns are the series
# every other asset is ranked against. 
# These tickers have come from Gemini.
REFERENCE_NAME = "skhynix"
REFERENCE_SYMBOL = "000660.KS"

# Pinned assets: always drawn on the ladder regardless of rank so the dollar
# legs stay visible even when they fall outside the strongest field. The broad
# dollar index and the won pair are both pinned. 
# These tickers have come from Gemini.
PINNED_ASSET_NAMES = ["dxy", "usd_krw"]

# Shared scan universe, copied from the investigation-2 scraper so the two
# investigations stay independent. PRICE_TICKERS become daily log returns and
# YIELD_TICKERS become daily first differences.
# These tickers have come from Gemini.
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
    # These tickers have come from Gemini.
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
    # These tickers have come from Gemini.
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
    # International sovereign yield coverage on Yahoo Finance is bad, so
    # so I have used FRED API instead (the Federal reserve's API key). This is 
    # much better and a complete fix. Naturaly, the API key will be kept
    # from being shared publically on github. These tickers have been taken from 
    # Chat GPT.
    "international_bond_yield_candidate": {
        "germany_10y_yield":      "IRLTLT01DEM156N",
        "uk_10y_yield":           "IRLTLT01GBM156N",
        "japan_10y_yield":        "IRLTLT01JPM156N",
        "canada_10y_yield":       "IRLTLT01CAM156N",
        "switzerland_10y_yield":  "IRLTLT01SEM156N",
        "korea_10y_yield":        "IRLTLT01KRM156N",
        "australia_10y_yield":    "IRLTLT01AUM156N",
    },
}

# Extra price tickers added on top of the shared universe. These are the
# SK-Hynix-specific series the NASDAQ scan does not carry: the two dollar
# measures, the won, and the home equity indices SK Hynix trades inside.
# These tickers have come from Gemini.
EXTRA_PRICE_TICKERS: Dict[str, Dict[str, str]] = {
    "usd_index": {
        "dxy":          "DX-Y.NYB",
        "usd_bull_etf": "UUP",
    },
    "fx": {
        "usd_krw": "KRW=X",
    },
    "home_market_equity": {
        "kospi_index":  "^KS11",
        "kosdaq_index": "^KQ11",
    },
}

# Number of strongest rungs (by absolute Spearman) drawn before the pinned
# dollar legs are guaranteed onto the ladder.
TOP_N_LADDER_BARS = 18

# Minimum overlapping observations needed to keep an asset in the ranking. The
# daily threshold guards the yfinance universe; the lower monthly threshold
# applies to the FRED yields, which are correlated at monthly frequency.
MIN_OBS_FOR_CORRELATION = 60
MIN_OBS_FOR_MONTHLY_CORRELATION = 24
MONTHLY_RESAMPLE_RULE = "ME"

# Data alignment.
PANEL_FFILL_LIMIT = 2
DEFAULT_START_DATE = "2015-01-01"

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: base colours.
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

# Distinct colours for the SK-Hynix-specific categories. 
# The dollar index sits on a deep red, the won on 
# a green and the home indices on a blue.
USD_INDEX_COLOUR = "#9B1D20"
FX_COLOUR = "#2D6A4F"
HOME_MARKET_COLOUR = "#274C77"

# One colour and one label per asset category for the ladder.
LADDER_CATEGORY_COLOURS: Dict[str, str] = {
    "us_sector_etf":                     PRIMARY_RED,
    "us_industry_etf":                   ACCENT_RED,
    "broad_index_etf":                   PRIMARY_BLUE,
    "country_etf":                       ACCENT_BLUE,
    "us_bond_yield":                     PRIMARY_GREEN,
    "us_bond_etf":                       ACCENT_GREEN,
    "international_bond_etf":             NEUTRAL_GREY,
    "international_bond_yield_candidate": NEUTRAL_DARK,
    "crypto_token":                      DEEP_CRIMSON,
    "crypto_equity":                     DEEP_NAVY,
    "usd_index":                         USD_INDEX_COLOUR,
    "fx":                                FX_COLOUR,
    "home_market_equity":                HOME_MARKET_COLOUR,
}
LADDER_CATEGORY_LABELS: Dict[str, str] = {
    "us_sector_etf":                     "US sector ETF",
    "us_industry_etf":                   "US industry ETF",
    "broad_index_etf":                   "Broad index ETF",
    "country_etf":                       "Country / international equity ETF",
    "us_bond_yield":                     "US Treasury yield",
    "us_bond_etf":                       "US bond ETF",
    "international_bond_etf":             "International bond ETF",
    "international_bond_yield_candidate": "International sovereign yield",
    "crypto_token":                      "Cryptocurrency token",
    "crypto_equity":                      "Crypto-exposed equity",
    "usd_index":                         "US dollar index",
    "fx":                                "FX (USD/KRW)",
    "home_market_equity":                "Korean home-market equity index",
}

# Plotting: lollipop styling. Each rung is one asset drawn as a stem from
# x = 0 to its Spearman value with a marker on the end.
LADDER_STEM_LINEWIDTH = 1.8
LADDER_MARKER_SIZE = 90
PINNED_MARKER_SIZE = 320
LADDER_MARKER_EDGE_COLOUR = NEUTRAL_DARK
LADDER_MARKER_EDGE_LINEWIDTH = 0.6
LADDER_MIN_FIGURE_HEIGHT = 5.5
LADDER_HEIGHT_PER_RUNG = 0.34
LADDER_FIGURE_WIDTH = 11.0
REFERENCE_AXIS_LINEWIDTH = 0.5
VALUE_LABEL_PAD = 0.012
AXIS_NEGATIVE_HEADROOM = 0.12
AXIS_POSITIVE_HEADROOM = 0.18
AXIS_MIN_NEGATIVE_FLOOR = -0.2

# Plotting: caption layout, resolution and style.
CAPTION_X_POSITION = 0.5
CAPTION_Y_POSITION = 0.02
CAPTION_BOTTOM_PAD = 0.18
DEFAULT_DPI = 140
SEABORN_STYLE = "whitegrid"

# Filesystem layout. One sub-directory per pipeline keeps these outputs
# separate inside investigation-3.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "skhynix_correlation_ladder"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / SCRIPT_SUBDIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" / SCRIPT_SUBDIR
VISUALS_DIR = PROJECT_ROOT / "visuals" / SCRIPT_SUBDIR

# Final deliverable copy. The curated conclusion directory holds the headline
# PNG for investigation-3.
CONCLUSION_DIR = PROJECT_ROOT / "conclusion_3"
LADDER_FILE_NAME = "skhynix_spearman_ladder.png"

# FRED API access. The international sovereign yields are pulled from the
# Federal Reserve FRED API rather than yfinance, whose coverage of those yields
# is unreliable. Any asset whose category is listed in FRED_CATEGORIES carries
# a FRED series ID instead of a Yahoo symbol. The key is read from a .env file
# (which is git-ignored) so it is never committed; the style-guide placeholder
# is used when no key is configured.
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_MISSING_VALUE_STRING = "."
FRED_CATEGORIES = {"international_bond_yield_candidate"}
FRED_API_PLACEHOLDER = "fred api key goes here"


def load_fred_api_key() -> str:
    """
    Resolve the FRED API key without ever hard-coding it in the source. The
    environment variable FRED_API wins; otherwise a .env file is searched for
    from this file's directory upward to the repository root. Falls back to the
    style-guide placeholder so an un-keyed checkout still imports without error.

    INPUTS:
        * None (reads the environment and any .env on the path to the root)

    OUTPUTS:
        * The FRED API key string, or the placeholder if none is configured.
    """
    environment_value = os.environ.get("FRED_API")
    if environment_value:
        return environment_value
    module_directory = Path(__file__).resolve().parent
    for directory in [module_directory, *module_directory.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text().splitlines():
                stripped_line = raw_line.strip()
                if stripped_line.startswith("FRED_API"):
                    _, _, raw_value = stripped_line.partition("=")
                    return raw_value.strip().strip('"').strip("'")
    return FRED_API_PLACEHOLDER


FRED_API = load_fred_api_key()


# Logging.
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("skhynix-correlation-ladder")


# Result containers (dataclasses).

@dataclass
class AssetSpec:
    """
    Lightweight specification for one asset in the scan universe. Carries the
    friendly name, the Yahoo Finance ticker, the category and whether the
    series is a yield (first-difference) or a price (log-return).

    INPUTS:
        * name      : friendly identifier used in output tables
        * symbol    : Yahoo Finance ticker or FRED series ID for the category
        * category  : universe category key
        * is_yield  : True for sovereign yield series; False for prices

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
    Pearson and Spearman correlation results for one asset against SK Hynix.
    Written to the ranked tables and used as the input to the ladder.

    INPUTS:
        * name      : friendly identifier
        * symbol    : Yahoo Finance ticker or FRED series ID
        * category  : universe category key
        * n_obs     : number of overlapping observations used
        * pearson   : Pearson linear correlation versus SK Hynix
        * spearman  : Spearman rank correlation versus SK Hynix
        * frequency : sampling frequency the correlation was computed at,
                      'daily' for the yfinance universe and 'monthly' for the
                      monthly FRED yields

    OUTPUTS:
        * Dataclass instance with the listed fields.
    """
    name: str
    symbol: str
    category: str
    n_obs: int
    pearson: float
    spearman: float
    frequency: str = "daily"


# Persistence helpers.

def save_parquet(frame: pd.DataFrame, directory: Path, name_stem: str) -> Path:
    """
    Write a DataFrame to Parquet under the supplied directory, creating the
    directory if it does not yet exist. Parquet preserves dtypes and reloads
    faster than CSV, so it is the default on-disk format for downstream reuse.

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
    Write a DataFrame to CSV under the supplied directory. The ranked
    correlation table is also written to CSV because it is the headline
    tabular deliverable that the write-up reads directly.

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

def download_one_safe(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Attempt to download one Yahoo Finance ticker. Returns None on any failure
    or empty response so the calling loop can continue scraping the rest of
    the universe without aborting the whole run on a single missing symbol.

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
    Pull the close column out of a raw yfinance DataFrame and rename it to the
    friendly identifier so the resulting panel column is easy to identify.

    INPUTS:
        * raw_frame      : DataFrame returned by yf.download
        * friendly_name  : output Series name

    OUTPUTS:
        * Close-price Series renamed to friendly_name.
    """
    if "Close" in raw_frame.columns:
            close_col = "Close" 
    else:
        close_col = raw_frame.columns[0]
    close_series = raw_frame[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series.rename(friendly_name)


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Pull one FRED series via the public REST endpoint and return it as a
    date-indexed Series. Returns None on any failure or empty response so the
    calling loop can continue without aborting the run. Used for the
    international sovereign yields, whose FRED series are monthly.

    INPUTS:
        * series_id   : FRED series identifier, for example IRLTLT01KRM156N
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * date-indexed float Series sorted ascending, or None if unavailable.
    """
    request_parameters = {
        "series_id": series_id,
        "api_key": FRED_API,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    try:
        response = requests.get(FRED_BASE_URL, params = request_parameters, timeout = 30)
        response.raise_for_status()
    except Exception as fetch_error:
        log.warning("Skipping FRED %s (request error: %s)", series_id, fetch_error)
        return None
    observations = response.json().get("observations", [])
    parsed_dates: List[pd.Timestamp] = []
    parsed_values: List[float] = []
    for observation in observations:
        if observation["value"] == FRED_MISSING_VALUE_STRING:
            continue
        parsed_dates.append(pd.Timestamp(observation["date"]))
        parsed_values.append(float(observation["value"]))
    if len(parsed_values) == 0:
        return None
    return pd.Series(
        data = np.array(parsed_values, dtype = float),
        index = pd.DatetimeIndex(parsed_dates),
    ).sort_index()


def build_reference_universe() -> List[AssetSpec]:
    """
    Flatten the shared price and yield ticker dictionaries plus the extra
    SK-Hynix-specific price tickers into a single list of AssetSpec records.
    The yield flag is carried through so the change transform knows to use
    first differences for yields and log returns for prices.

    INPUTS:
        * None (reads PRICE_TICKERS, YIELD_TICKERS and EXTRA_PRICE_TICKERS)

    OUTPUTS:
        * List of AssetSpec covering the full SK Hynix scan universe.
    """
    universe: List[AssetSpec] = []
    for ticker_group in (PRICE_TICKERS, EXTRA_PRICE_TICKERS):
        for category, members in ticker_group.items():
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


def fetch_universe_panel(
    universe: List[AssetSpec], start_date: str, end_date: str,
) -> tuple[pd.DataFrame, List[AssetSpec]]:
    """
    Download every asset in the universe defensively. Symbols yfinance does
    not return are logged and skipped so a single missing ticker never aborts
    the run. Raw frames are cached as Parquet under DATA_RAW_DIR.

    INPUTS:
        * universe    : list of AssetSpec records to attempt
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Tuple of (panel DataFrame of close prices, list of AssetSpec that
          fetched successfully).
    """
    fetched_series: List[pd.Series] = []
    fetched_specs: List[AssetSpec] = []
    fred_specs: List[AssetSpec] = []

    for asset_spec in universe:
        # FRED-sourced categories carry FRED series IDs, not Yahoo symbols, and
        # are fetched separately once the daily panel index is established.
        if asset_spec.category in FRED_CATEGORIES:
            fred_specs.append(asset_spec)
            continue
        raw_frame = download_one_safe(asset_spec.symbol, start_date, end_date)
        if raw_frame is None:
            log.info("No data for %s (%s); skipping",
                     asset_spec.name, asset_spec.symbol)
            continue
        save_parquet(raw_frame, DATA_RAW_DIR, f"raw_{asset_spec.name}")
        fetched_series.append(extract_close_series(raw_frame, asset_spec.name))
        fetched_specs.append(asset_spec)

    if len(fetched_series) == 0:
        return pd.DataFrame(), []

    panel = pd.concat(fetched_series, axis = 1).dropna(how = "all")
    panel = panel.ffill(limit = PANEL_FFILL_LIMIT)

    # FRED series are monthly. Reindex each onto the daily panel with forward
    # fill so the latest published monthly value carries across the daily grid.
    for asset_spec in fred_specs:
        fred_series = fetch_fred_series(asset_spec.symbol, start_date, end_date)
        if fred_series is None:
            log.info("No FRED data for %s (%s); skipping",
                     asset_spec.name, asset_spec.symbol)
            continue
        save_parquet(
            fred_series.to_frame(name = asset_spec.name),
            DATA_RAW_DIR, f"raw_{asset_spec.name}",
        )
        panel[asset_spec.name] = fred_series.reindex(panel.index, method = "ffill")
        fetched_specs.append(asset_spec)

    log.info("Fetched %d of %d universe assets", len(fetched_specs), len(universe))
    return panel, fetched_specs


# Returns and yield changes.

def to_daily_changes(panel: pd.DataFrame, asset_specs: List[AssetSpec]) -> pd.DataFrame:
    """
    Convert the price and yield panel into a single daily-change frame. Price
    columns become log returns; yield columns become first differences.
    Spearman rank correlation is invariant to monotone transformations so the
    two scales can sit alongside each other without distorting the ranking.

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


# Correlation computation.

def compute_correlations_against_reference(
    changes: pd.DataFrame,
    asset_specs: List[AssetSpec],
    reference_name: str,
) -> List[CorrelationRecord]:
    """
    Compute the Pearson and Spearman correlation between the reference column
    and every other column in the daily-change frame. Assets with fewer than
    MIN_OBS_FOR_CORRELATION overlapping observations are skipped so noisy
    short-history correlations do not enter the ranking.

    INPUTS:
        * changes         : DataFrame of daily changes, one column per asset
        * asset_specs     : list of AssetSpec for the non-reference columns
        * reference_name  : column name of the reference asset (SK Hynix)

    OUTPUTS:
        * List of CorrelationRecord, one per asset clearing the minimum-
          observation threshold.
    """
    if reference_name not in changes.columns:
        raise RuntimeError(
            f"Reference column '{reference_name}' missing from changes frame"
        )
    reference_series = changes[reference_name]

    spec_by_name: Dict[str, AssetSpec] = {}
    for asset_spec in asset_specs:
        spec_by_name[asset_spec.name] = asset_spec

    records: List[CorrelationRecord] = []
    for column_name in changes.columns:
        if column_name == reference_name:
            continue
        column_spec = spec_by_name.get(column_name)
        if column_spec is None:
            continue
        paired = pd.concat(
            [reference_series, changes[column_name]], axis = 1,
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


def compute_monthly_correlations(
    panel: pd.DataFrame,
    asset_specs: List[AssetSpec],
    reference_name: str,
    reference_is_yield: bool = False,
) -> List[CorrelationRecord]:
    """
    Compute Pearson and Spearman against SK Hynix at monthly frequency for the
    monthly FRED series. Forcing a monthly level onto the daily grid and
    differencing it leaves a column that is zero on about ninety-five per cent
    of days; that block of tied zeros collapses the daily rank correlation
    toward zero regardless of any real relationship. Resampling both sides to
    month-end first removes the artefact and gives the correlation at the
    series' true frequency, at the cost of a far smaller sample.

    INPUTS:
        * panel               : daily LEVEL panel of prices and yields
        * asset_specs         : monthly (FRED) AssetSpec records to score
        * reference_name      : column name of the reference asset (SK Hynix)
        * reference_is_yield  : True if the reference is a yield, else a price

    OUTPUTS:
        * List of CorrelationRecord with frequency set to 'monthly'.
    """
    monthly_levels = panel.resample(MONTHLY_RESAMPLE_RULE).last()
    if reference_is_yield:
        reference_change = monthly_levels[reference_name].diff()
    else:
        reference_change = np.log(monthly_levels[reference_name]).diff()

    records: List[CorrelationRecord] = []
    for asset_spec in asset_specs:
        if asset_spec.name not in monthly_levels.columns:
            continue
        if asset_spec.is_yield:
            asset_change = monthly_levels[asset_spec.name].diff()
        else:
            asset_change = np.log(monthly_levels[asset_spec.name]).diff()
        paired = pd.concat([reference_change, asset_change], axis = 1).dropna()
        if len(paired) < MIN_OBS_FOR_MONTHLY_CORRELATION:
            continue
        pearson_value = float(
            paired.iloc[:, 0].corr(paired.iloc[:, 1], method = "pearson")
        )
        spearman_value = float(
            paired.iloc[:, 0].corr(paired.iloc[:, 1], method = "spearman")
        )
        records.append(CorrelationRecord(
            name = asset_spec.name,
            symbol = asset_spec.symbol,
            category = asset_spec.category,
            n_obs = int(len(paired)),
            pearson = pearson_value,
            spearman = spearman_value,
            frequency = "monthly",
        ))
    return records


def select_ladder_records(
    records: List[CorrelationRecord],
    top_n: int,
    pinned_names: List[str],
) -> List[CorrelationRecord]:
    """
    Select the rungs drawn on the ladder. The strongest top_n assets by
    absolute Spearman are taken first; every pinned asset (the dollar legs) is
    then forced into the set if the strength ranking did not already include
    it. The result is ordered by signed Spearman so the strongest inverse rung
    sits at the bottom and the strongest co-mover at the top.

    INPUTS:
        * records       : full list of CorrelationRecord for the universe
        * top_n         : number of strongest rungs to take by absolute rho
        * pinned_names  : asset names always guaranteed onto the ladder

    OUTPUTS:
        * Ordered list of CorrelationRecord to plot, bottom rung first.
    """
    by_strength = sorted(
        records, key = lambda record: abs(record.spearman), reverse = True,
    )
    chosen = list(by_strength[:top_n])
    chosen_names = {record.name for record in chosen}
    for pinned_name in pinned_names:
        if pinned_name in chosen_names:
            continue
        pinned_record = next(
            (record for record in records if record.name == pinned_name), None,
        )
        if pinned_record is not None:
            chosen.append(pinned_record)
            chosen_names.add(pinned_name)
    chosen.sort(key = lambda record: record.spearman)
    return chosen


def records_to_frame(records: List[CorrelationRecord]) -> pd.DataFrame:
    """
    Convert a list of CorrelationRecord into a tidy DataFrame for CSV and
    Parquet output, sorted strongest co-mover first.

    INPUTS:
        * records  : list of CorrelationRecord

    OUTPUTS:
        * DataFrame with one row per record.
    """
    ordered = sorted(records, key = lambda record: record.spearman, reverse = True)
    rows = []
    for record in ordered:
        rows.append({
            "name":      record.name,
            "symbol":    record.symbol,
            "category":  record.category,
            "frequency": record.frequency,
            "n_obs":     record.n_obs,
            "pearson":   record.pearson,
            "spearman":  record.spearman,
        })
    return pd.DataFrame(rows)


# Plotting.

def add_caption(figure: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption underneath the figure so the saved PNG carries
    a self-contained explanation of the axis, the sign convention and the
    pinned dollar rungs.

    INPUTS:
        * figure        : matplotlib Figure
        * caption_text  : caption string

    OUTPUTS:
        * None. Mutates the figure in place.
    """
    figure.subplots_adjust(bottom = CAPTION_BOTTOM_PAD)
    figure.text(
        CAPTION_X_POSITION, CAPTION_Y_POSITION, caption_text,
        ha = "center", va = "bottom",
        fontsize = TEXT_FONT, color = CAPTION_COLOUR, wrap = True,
    )


def plot_spearman_ladder(
    records: List[CorrelationRecord], pinned_names: List[str],
) -> Path:
    """
    Render the strongest Spearman rank correlations against SK Hynix as a
    horizontal lollipop ladder. Each rung is one asset coloured by category;
    the pinned dollar legs are drawn with star markers so the USD reading is
    unmistakable. Each marker is annotated with its rho value.

    INPUTS:
        * records       : ordered ladder records, bottom rung first
        * pinned_names  : asset names drawn with the highlighted star marker

    OUTPUTS:
        * Path to the saved PNG.
    """
    rung_labels: List[str] = []
    rung_values: List[float] = []
    rung_colours: List[str] = []
    present_categories: List[str] = []
    for record in records:
        rung_labels.append(f"{record.name} ({record.symbol})")
        rung_values.append(record.spearman)
        rung_colours.append(LADDER_CATEGORY_COLOURS.get(record.category, NEUTRAL_DARK))
        if record.category not in present_categories:
            present_categories.append(record.category)

    pinned_positions = [
        index for index, record in enumerate(records) if record.name in pinned_names
    ]
    standard_positions = [
        index for index in range(len(records)) if index not in pinned_positions
    ]

    figure_height = max(
        LADDER_MIN_FIGURE_HEIGHT, LADDER_HEIGHT_PER_RUNG * len(records),
    )
    figure, axis = plt.subplots(figsize = (LADDER_FIGURE_WIDTH, figure_height))
    rung_positions = np.arange(len(records))

    axis.hlines(
        y = rung_positions, xmin = 0.0, xmax = rung_values,
        colors = rung_colours, linewidth = LADDER_STEM_LINEWIDTH,
    )
    axis.scatter(
        [rung_values[index] for index in standard_positions],
        [rung_positions[index] for index in standard_positions],
        c = [rung_colours[index] for index in standard_positions],
        s = LADDER_MARKER_SIZE,
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 3,
    )
    axis.scatter(
        [rung_values[index] for index in pinned_positions],
        [rung_positions[index] for index in pinned_positions],
        c = [rung_colours[index] for index in pinned_positions],
        s = PINNED_MARKER_SIZE,
        marker = "*",
        edgecolors = LADDER_MARKER_EDGE_COLOUR,
        linewidths = LADDER_MARKER_EDGE_LINEWIDTH,
        zorder = 4,
    )

    for index, value in enumerate(rung_values):
        if value >= 0:
            text_x = value + VALUE_LABEL_PAD
            alignment = "left"
        else:
            text_x = value - VALUE_LABEL_PAD
            alignment = "right"
        axis.text(
            text_x, rung_positions[index], f"{value:+.2f}",
            va = "center", ha = alignment, fontsize = TEXT_FONT - 1,
            color = NEUTRAL_DARK,
        )

    axis.set_yticks(rung_positions)
    axis.set_yticklabels(rung_labels, fontsize = TEXT_FONT)
    axis.axvline(0, color = NEUTRAL_DARK, lw = REFERENCE_AXIS_LINEWIDTH)

    axis_left = min(AXIS_MIN_NEGATIVE_FLOOR, min(rung_values) - AXIS_NEGATIVE_HEADROOM)
    axis_right = max(rung_values) + AXIS_POSITIVE_HEADROOM
    axis.set_xlim(left = axis_left, right = axis_right)
    axis.set_xlabel("Spearman rank correlation vs SK Hynix (000660.KS)")
    axis.set_title(
        f"Strongest Spearman rank correlations vs SK Hynix "
        f"(top {TOP_N_LADDER_BARS} by strength, USD legs pinned)"
    )

    legend_handles: List[object] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = LADDER_CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = LADDER_CATEGORY_LABELS.get(category_key, category_key),
        ))
    legend_handles.append(Line2D(
        [0], [0], marker = "*", color = "none",
        markerfacecolor = NEUTRAL_GREY, markeredgecolor = LADDER_MARKER_EDGE_COLOUR,
        markersize = 16, label = "Pinned dollar leg (DXY, USD/KRW)",
    ))
    axis.legend(
        handles = legend_handles, loc = "lower right",
        fontsize = TEXT_FONT, title = "Asset category",
    )

    figure.tight_layout()
    add_caption(
        figure,
        "Lollipop ladder of the assets whose daily Spearman rank correlation "
        "with SK Hynix over the full sample is the strongest by absolute "
        "value, with the broad dollar index (DXY) and USD/KRW pinned on "
        "regardless of rank. Positive rho marks co-movers and substitutes; "
        "negative rho marks inverse movers. Both dollar measures rise on "
        "dollar strength, so a negative rho means SK Hynix tends to fall on "
        "dollar-strength, risk-off days. Prices enter as daily log returns "
        "and yields as first differences.",
    )

    VISUALS_DIR.mkdir(parents = True, exist_ok = True)
    output_path = VISUALS_DIR / LADDER_FILE_NAME
    figure.savefig(output_path, dpi = DEFAULT_DPI, bbox_inches = "tight")
    plt.close(figure)
    return output_path


# Plot styling.

def configure_plot_style() -> None:
    """
    Apply the seaborn whitegrid background and the project font sizes so the
    ladder matches the look of the other figures in the project.

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

def run_pipeline(start_date: str, end_date: str) -> Dict[str, object]:
    """
    Execute the end-to-end scan: fetch SK Hynix and the full universe
    (including the two dollar measures, USD/KRW and the home indices), convert
    prices to log returns and yields to first differences, compute Pearson and
    Spearman against SK Hynix, persist the ranked table, select the strongest
    rungs with the dollar legs pinned, render the ladder and copy it into the
    conclusion directory.

    INPUTS:
        * start_date  : ISO date string, inclusive
        * end_date    : ISO date string, exclusive

    OUTPUTS:
        * Dictionary summary of the run, also written to JSON.
    """
    for required_dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALS_DIR, CONCLUSION_DIR):
        required_dir.mkdir(parents = True, exist_ok = True)
    configure_plot_style()

    log.info("Fetching SK Hynix (%s)", REFERENCE_SYMBOL)
    reference_raw = download_one_safe(REFERENCE_SYMBOL, start_date, end_date)
    if reference_raw is None:
        raise RuntimeError("Could not fetch SK Hynix; aborting")
    save_parquet(reference_raw, DATA_RAW_DIR, f"raw_{REFERENCE_NAME}")
    reference_close = extract_close_series(reference_raw, REFERENCE_NAME)

    log.info("Fetching the candidate scan universe")
    universe = build_reference_universe()
    universe_panel, fetched_specs = fetch_universe_panel(universe, start_date, end_date)

    full_panel = pd.concat([reference_close, universe_panel], axis = 1)
    full_panel = full_panel.dropna(how = "all").ffill(limit = PANEL_FFILL_LIMIT)
    save_parquet(full_panel, DATA_PROCESSED_DIR, "prices_and_yields")

    reference_spec = AssetSpec(
        name = REFERENCE_NAME, symbol = REFERENCE_SYMBOL,
        category = "reference", is_yield = False,
    )
    all_specs = [reference_spec] + fetched_specs
    changes = to_daily_changes(full_panel, all_specs)
    save_parquet(changes, DATA_PROCESSED_DIR, "daily_changes")

    # The yfinance universe is correlated at daily frequency. The FRED yields
    # are monthly, so they are correlated at monthly frequency instead of being
    # forced onto the daily grid, which would bury their signal under tied
    # zeros. The two frequencies are kept apart and labelled in the output.
    daily_specs = [spec for spec in fetched_specs if spec.category not in FRED_CATEGORIES]
    monthly_specs = [spec for spec in fetched_specs if spec.category in FRED_CATEGORIES]

    log.info("Computing daily correlations against SK Hynix")
    records = compute_correlations_against_reference(changes, daily_specs, REFERENCE_NAME)

    log.info("Computing monthly correlations for the FRED yields")
    records = records + compute_monthly_correlations(
        full_panel, monthly_specs, REFERENCE_NAME, reference_is_yield = False,
    )

    full_frame = records_to_frame(records)
    save_csv(full_frame, DATA_PROCESSED_DIR, "correlations_all_full_sample")
    save_parquet(full_frame, DATA_PROCESSED_DIR, "correlations_all_full_sample")

    # The ladder plot is a daily-frequency ranking, so the monthly FRED rows
    # are excluded from it; they live in the correlation table above.
    daily_records = [record for record in records if record.frequency == "daily"]
    ladder_records = select_ladder_records(daily_records, TOP_N_LADDER_BARS, PINNED_ASSET_NAMES)
    save_csv(records_to_frame(ladder_records), DATA_PROCESSED_DIR, "ladder_records")

    log.info("Plotting the SK Hynix Spearman ladder")
    ladder_path = plot_spearman_ladder(ladder_records, PINNED_ASSET_NAMES)

    conclusion_path = CONCLUSION_DIR / LADDER_FILE_NAME
    conclusion_path.write_bytes(ladder_path.read_bytes())
    log.info("Copied ladder to %s", conclusion_path)

    record_by_name = {record.name: record for record in records}
    dollar_leg_summary: Dict[str, object] = {}
    for pinned_name in PINNED_ASSET_NAMES:
        pinned_record = record_by_name.get(pinned_name)
        if pinned_record is not None:
            dollar_leg_summary[pinned_name] = {
                "symbol": pinned_record.symbol,
                "spearman": pinned_record.spearman,
                "pearson": pinned_record.pearson,
                "n_obs": pinned_record.n_obs,
            }

    summary: Dict[str, object] = {
        "run_at": datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        "start": str(full_panel.index.min().date()),
        "end": str(full_panel.index.max().date()),
        "universe_attempted": len(universe),
        "universe_fetched": len(fetched_specs),
        "top_n": TOP_N_LADDER_BARS,
        "dollar_legs": dollar_leg_summary,
    }
    summary_path = DATA_PROCESSED_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent = 2))
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
    parser.add_argument("--end", default = datetime.utcnow().strftime("%Y-%m-%d"))
    return parser.parse_args()


# Main.

def main() -> None:
    """
    Entry point. Runs the SK Hynix ladder scan and prints a one-screen summary
    covering the window, universe counts and the pinned dollar-leg correlations.

    INPUTS:
        * None

    OUTPUTS:
        * None. Side effects: ladder PNG in VISUALS_DIR and CONCLUSION_DIR,
          ranked tables in DATA_PROCESSED_DIR, raw caches in DATA_RAW_DIR,
          JSON summary in DATA_PROCESSED_DIR, stdout summary.
    """
    cli_args = parse_args()
    summary = run_pipeline(cli_args.start, cli_args.end)

    print("\nPipeline summary")
    print(f"Window               : {summary['start']} -> {summary['end']}")
    print(f"Universe attempted   : {summary['universe_attempted']}")
    print(f"Universe fetched ok  : {summary['universe_fetched']}")
    print("\nDollar legs vs SK Hynix:")
    for leg_name, leg_stats in summary["dollar_legs"].items():
        print(
            f"  {leg_name:10s} ({leg_stats['symbol']:9s}): "
            f"Spearman {leg_stats['spearman']:+.3f}  "
            f"Pearson {leg_stats['pearson']:+.3f}  n = {leg_stats['n_obs']}"
        )
    print(f"\nVisuals in    : {VISUALS_DIR}")
    print(f"Conclusion in : {CONCLUSION_DIR}")
    print(f"Data in       : {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
