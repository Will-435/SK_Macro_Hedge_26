"""
pca_factors.py

Builds a stationary, daily frequency panel of factors relevant to a four week
(twenty trading day) USD KRW return forecast, then reduces the panel to
principal components for downstream conditional sampling in bootstrap_pdf.py.

EXCLUDED CANDIDATE FACTORS

Two macro candidates were deliberately excluded because they operate on the
wrong horizon for a four week return model.

Debt to GDP ratios are quarterly and behave as a long run fair value variable.
Forward filling a quarterly series onto a daily or weekly grid produces a
near constant column that contributes no variance to PCA and no information
at a four week horizon. Debt to GDP belongs in a long horizon fair value
model, not in this short horizon return model.

CPI is monthly, sticky, largely priced in before release, and release lagged.
The inflation surprise matters at the release instant but the level itself
is not a four week driver. Excluded for the same horizon mismatch reason.

The Korea Value Up Index was also dropped. It launched in late 2024 with
roughly eighteen months of history, which cannot span a ten year backtest
window without dominating the panel with missing values.

NATURAL GAS PROXY LIMITATION

True global LNG spot is not available on yfinance. Henry Hub front month
futures (NG=F) are used as a proxy. This is a known limitation: Henry Hub
tracks United States gas balances, whereas Korean LNG import costs are
driven by Asian JKM and oil indexed long term contracts. The directionality
is broadly aligned during global energy shocks but the level relationship
is loose.

VIX AS A KOREAN RISK GAUGE

Korea sovereign CDS would be a cleaner risk appetite gauge for the KRW but
it requires a paid feed. VIX is used as the free substitute. This trades
off Korea specific risk pricing for a broad global risk on risk off signal.
"""


import datetime as dt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt


FRED_API = "fred api key goes here"
ECOS_API = "bok ecos api key goes here"

LOOKBACK_YEARS = 10
LOOKBACK_BUFFER_DAYS = 60
DAYS_PER_YEAR_CALENDAR = 365.25

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
ECOS_PAGE_SIZE = 100000
ECOS_MISSING_VALUE_STRING = "."

FIGURE_DPI = 150
SCREE_FIGSIZE = (10, 6)
SCREE_BAR_ALPHA = 0.6

# yfinance tickers. The map key is the internal feature name carried through
# the rest of the pipeline. The value is the yfinance ticker symbol.
YFINANCE_TICKERS = {
    "usdkrw": "KRW=X",
    "dxy": "DX-Y.NYB",
    "usdcnh": "CNH=X",
    "brent": "BZ=F",
    "wti": "CL=F",
    "natgas_henry_hub": "NG=F",
    "kospi": "^KS11",
    "semis_etf": "SOXX",
    "vix": "^VIX",
}

# ECOS statistical table codes. The codes below are illustrative placeholders
# the user must verify against the live ECOS catalogue before running. The
# tuple is (table code, cycle, item code).
ECOS_STAT_CODES = {
    "foreign_equity_flow":  ("731Y004", "D", "1000000"),
    "foreign_bond_flow":    ("731Y004", "D", "2000000"),
    "kr_10y_yield":         ("817Y002", "D", "010210000"),
    "trade_balance":        ("301Y013", "M", "1000"),
    "current_account":      ("301Y013", "M", "1000"),
    "rfc_deposits":         ("104Y014", "M", "BMAA1"),
}

# Calendar lag between a Korean monthly macro reference period end and the
# real world release date. These lags ensure no forward looking information
# leaks into the model.
KOREA_MACRO_RELEASE_LAG_DAYS = {
    "trade_balance":   20,
    "current_account": 35,
    "rfc_deposits":    25,
}

DAILY_ECOS_SERIES = ("foreign_equity_flow", "foreign_bond_flow", "kr_10y_yield")
MONTHLY_ECOS_SERIES = ("trade_balance", "current_account", "rfc_deposits")


def fetch_fred_series(series_code = None, start_date = None, end_date = None):
    """
    Pull a single FRED series via the public REST endpoint and return it
    as a date indexed pandas Series.

    INPUTS:
        * series code as the FRED identifier string, for example DGS10
        * start date as ISO yyyy mm dd string
        * end date as ISO yyyy mm dd string

    OUTPUTS:
        * pandas Series indexed by observation date with float values
    """
    request_parameters = {
        "series_id": series_code,
        "api_key": FRED_API,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    response = requests.get(FRED_BASE_URL, params = request_parameters, timeout = 30)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    parsed_dates = []
    parsed_values = []
    for observation in observations:
        if observation["value"] == ECOS_MISSING_VALUE_STRING:
            continue
        parsed_dates.append(pd.Timestamp(observation["date"]))
        parsed_values.append(float(observation["value"]))
    return pd.Series(
        data = np.array(parsed_values, dtype = float),
        index = pd.DatetimeIndex(parsed_dates),
        name = series_code,
    )


def fetch_ecos_series(stat_code = None, cycle = None, item_code = None, start_date = None, end_date = None):
    """
    Pull a single Bank of Korea ECOS statistical series and return it as a
    pandas Series indexed by the reference period end timestamp.

    INPUTS:
        * stat code as the ECOS table identifier
        * cycle as D for daily or M for monthly
        * item code identifying the row within the table
        * start date as yyyymmdd for daily or yyyymm for monthly
        * end date in the same format as start date

    OUTPUTS:
        * pandas Series indexed by reference period timestamp
    """
    request_url = (
        f"{ECOS_BASE_URL}/{ECOS_API}/json/en/1/{ECOS_PAGE_SIZE}/"
        f"{stat_code}/{cycle}/{start_date}/{end_date}/{item_code}"
    )
    response = requests.get(request_url, timeout = 30)
    response.raise_for_status()
    payload = response.json()
    if "StatisticSearch" not in payload:
        return pd.Series(dtype = float, name = stat_code)
    rows = payload["StatisticSearch"].get("row", [])
    parsed_dates = []
    parsed_values = []
    for row in rows:
        time_string = row["TIME"]
        if cycle == "D":
            period_timestamp = pd.Timestamp(time_string)
        elif cycle == "M":
            year_part = time_string[:4]
            month_part = time_string[4:6]
            period_timestamp = pd.Timestamp(f"{year_part}-{month_part}-01") + pd.offsets.MonthEnd(0)
        else:
            period_timestamp = pd.Timestamp(time_string)
        try:
            parsed_value = float(row["DATA_VALUE"])
        except (ValueError, TypeError):
            continue
        parsed_dates.append(period_timestamp)
        parsed_values.append(parsed_value)
    return pd.Series(
        data = np.array(parsed_values, dtype = float),
        index = pd.DatetimeIndex(parsed_dates),
        name = stat_code,
    ).sort_index()


def lag_macro_to_release_date(monthly_series = None, lag_days = None):
    """
    Shift a monthly macro series from its reference period end timestamp to
    the actual real world release timestamp. This is the central leakage
    guard for monthly Korean macro factors.

    INPUTS:
        * monthly series indexed by reference period end
        * lag in calendar days between reference period end and release

    OUTPUTS:
        * series with the same values but reindexed to release timestamps
    """
    shifted_index = monthly_series.index + pd.Timedelta(days = lag_days)
    return pd.Series(
        data = monthly_series.values,
        index = shifted_index,
        name = monthly_series.name,
    )


def transform_log_returns(price_series = None):
    """
    Compute natural log returns of a strictly positive price series. Used
    for FX, equity indices, energy futures and the VIX level.

    INPUTS:
        * strictly positive price series

    OUTPUTS:
        * log return series with the same index
    """
    return np.log(price_series).diff()


def transform_first_difference(level_series = None):
    """
    Compute the first difference of a level series. Required for yields and
    yield spreads because they can sit near zero or turn negative, so log
    returns are either numerically unstable or mathematically undefined.

    INPUTS:
        * level series of yields or a yield spread

    OUTPUTS:
        * first differenced series
    """
    return level_series.diff()


def transform_standardised_level(signed_series = None):
    """
    Pass a signed quantity through unchanged ready for the panel wide
    standardisation step. Used for flow variables such as foreign equity
    flows, foreign bond flows, trade balance and current account, where the
    sign carries information and a logarithm is mathematically undefined.

    INPUTS:
        * signed level series

    OUTPUTS:
        * series cast to float without further transformation
    """
    return signed_series.astype(float)


def transform_rfc_deposits(deposits_series = None):
    """
    Transform resident foreign currency deposit balances. Apply log
    differences if strictly positive, otherwise fall back to a plain first
    difference. The series is reported in USD billions and is normally
    positive, but the fallback protects against any sign anomaly.

    INPUTS:
        * resident foreign currency deposit level series

    OUTPUTS:
        * stationary deposits series
    """
    if (deposits_series.dropna() > 0).all():
        return np.log(deposits_series).diff()
    return deposits_series.diff()


FEATURE_TRANSFORMS = {
    "usdkrw":               transform_log_returns,
    "dxy":                  transform_log_returns,
    "usdcnh":               transform_log_returns,
    "brent":                transform_log_returns,
    "wti":                  transform_log_returns,
    "natgas_henry_hub":     transform_log_returns,
    "kospi":                transform_log_returns,
    "semis_etf":            transform_log_returns,
    "vix":                  transform_log_returns,
    "us_10y_yield":         transform_first_difference,
    "kr_10y_yield":         transform_first_difference,
    "kr_us_yield_spread":   transform_first_difference,
    "foreign_equity_flow":  transform_standardised_level,
    "foreign_bond_flow":    transform_standardised_level,
    "trade_balance":        transform_standardised_level,
    "current_account":      transform_standardised_level,
    "rfc_deposits":         transform_rfc_deposits,
}


def fetch_yfinance_panel(start_date = None, end_date = None):
    """
    Download all configured yfinance tickers in a single batched call and
    return their close prices aligned by business day.

    INPUTS:
        * start date as ISO yyyy mm dd string
        * end date as ISO yyyy mm dd string

    OUTPUTS:
        * dataframe of close prices with one column per internal feature name
    """
    ticker_symbols = list(YFINANCE_TICKERS.values())
    ticker_to_feature = {ticker: feature for feature, ticker in YFINANCE_TICKERS.items()}
    raw_download = yf.download(
        tickers = ticker_symbols,
        start = start_date,
        end = end_date,
        auto_adjust = False,
        progress = False,
    )
    close_frame = raw_download["Close"].copy()
    close_frame = close_frame.rename(columns = ticker_to_feature)
    close_frame.index = pd.DatetimeIndex(close_frame.index)
    return close_frame


def build_raw_factor_panel(lookback_years = LOOKBACK_YEARS):
    """
    Assemble all factor sources onto a common business day grid. Monthly
    macro series are lagged to their release date before being forward
    filled onto the daily grid. The derived KR US yield spread is
    constructed inside this function.

    INPUTS:
        * lookback length in years for the backtest window

    OUTPUTS:
        * dataframe of raw factor levels indexed by business day
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days = int(lookback_years * DAYS_PER_YEAR_CALENDAR) + LOOKBACK_BUFFER_DAYS)

    market_prices = fetch_yfinance_panel(
        start_date = start_date.isoformat(),
        end_date = end_date.isoformat(),
    )

    us_10y_yield = fetch_fred_series(
        series_code = "DGS10",
        start_date = start_date.isoformat(),
        end_date = end_date.isoformat(),
    )
    us_10y_yield.name = "us_10y_yield"

    ecos_start_daily = start_date.strftime("%Y%m%d")
    ecos_end_daily = end_date.strftime("%Y%m%d")
    ecos_start_monthly = start_date.strftime("%Y%m")
    ecos_end_monthly = end_date.strftime("%Y%m")

    daily_ecos_frames = {}
    for series_name in DAILY_ECOS_SERIES:
        stat_code, cycle, item_code = ECOS_STAT_CODES[series_name]
        daily_series = fetch_ecos_series(
            stat_code = stat_code,
            cycle = cycle,
            item_code = item_code,
            start_date = ecos_start_daily,
            end_date = ecos_end_daily,
        )
        daily_series.name = series_name
        daily_ecos_frames[series_name] = daily_series

    monthly_ecos_frames = {}
    for series_name in MONTHLY_ECOS_SERIES:
        stat_code, cycle, item_code = ECOS_STAT_CODES[series_name]
        monthly_raw = fetch_ecos_series(
            stat_code = stat_code,
            cycle = cycle,
            item_code = item_code,
            start_date = ecos_start_monthly,
            end_date = ecos_end_monthly,
        )
        monthly_raw.name = series_name
        monthly_ecos_frames[series_name] = lag_macro_to_release_date(
            monthly_series = monthly_raw,
            lag_days = KOREA_MACRO_RELEASE_LAG_DAYS[series_name],
        )

    business_day_index = pd.bdate_range(start = start_date, end = end_date)
    factor_panel = pd.DataFrame(index = business_day_index)

    for column_name in market_prices.columns:
        factor_panel[column_name] = market_prices[column_name].reindex(business_day_index)

    factor_panel["us_10y_yield"] = us_10y_yield.reindex(business_day_index).ffill()

    for series_name, series_data in daily_ecos_frames.items():
        factor_panel[series_name] = series_data.reindex(business_day_index)

    for series_name, series_data in monthly_ecos_frames.items():
        factor_panel[series_name] = series_data.reindex(business_day_index).ffill()

    factor_panel["kr_us_yield_spread"] = factor_panel["kr_10y_yield"] - factor_panel["us_10y_yield"]
    return factor_panel


def apply_feature_transforms(raw_panel = None):
    """
    Apply each feature's per type stationarity transform to convert the
    raw factor panel into a stationary panel ready for standardisation.

    INPUTS:
        * raw factor panel indexed by business day

    OUTPUTS:
        * stationary panel with the same index and feature columns
    """
    stationary_panel = pd.DataFrame(index = raw_panel.index)
    for feature_name, transform_function in FEATURE_TRANSFORMS.items():
        if feature_name in raw_panel.columns:
            stationary_panel[feature_name] = transform_function(raw_panel[feature_name])
    return stationary_panel


def standardise_panel(stationary_panel = None):
    """
    Centre each column to zero mean and scale to unit variance. PCA is
    scale sensitive so standardisation is mandatory before fitting.

    INPUTS:
        * stationary panel of transformed features

    OUTPUTS:
        * standardised panel
        * series of per column means used for standardisation
        * series of per column standard deviations used for standardisation
    """
    column_means = stationary_panel.mean()
    column_stds = stationary_panel.std()
    standardised_panel = (stationary_panel - column_means) / column_stds
    return standardised_panel, column_means, column_stds


def fit_pca(standardised_panel = None):
    """
    Fit PCA via singular value decomposition on the standardised panel.
    Rows with any missing value are dropped before the decomposition.
    Returns the components, scores, explained variance ratio and the
    surviving column ordering for later projection.

    INPUTS:
        * standardised panel of zero mean unit variance columns

    OUTPUTS:
        * dictionary with keys scores, loadings, explained variance ratio,
          feature columns
    """
    clean_panel = standardised_panel.dropna(how = "any")
    centred_matrix = clean_panel.values - clean_panel.values.mean(axis = 0, keepdims = True)
    sample_count = centred_matrix.shape[0]
    _, singular_values, right_singular_t = np.linalg.svd(centred_matrix, full_matrices = False)
    explained_variance = (singular_values ** 2) / (sample_count - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    components_matrix = right_singular_t
    scores_matrix = centred_matrix @ components_matrix.T
    component_labels = [f"pc{component_index + 1}" for component_index in range(scores_matrix.shape[1])]
    scores_frame = pd.DataFrame(
        data = scores_matrix,
        index = clean_panel.index,
        columns = component_labels,
    )
    loadings_frame = pd.DataFrame(
        data = components_matrix,
        index = component_labels,
        columns = clean_panel.columns,
    )
    return {
        "scores": scores_frame,
        "loadings": loadings_frame,
        "explained_variance_ratio": explained_variance_ratio,
        "feature_columns": list(clean_panel.columns),
    }


def project_into_pc_space(standardised_row = None, loadings = None):
    """
    Project a single standardised feature vector into principal component
    space using the fitted loadings. Exposed so bootstrap_pdf.py can map a
    fresh factor reading into the same PC coordinates.

    INPUTS:
        * standardised row as a 1d array or Series
        * loadings as a dataframe with shape (components, features)

    OUTPUTS:
        * 1d numpy array of length equal to number of components
    """
    feature_vector = np.asarray(standardised_row).ravel()
    return loadings.values @ feature_vector


def plot_scree(explained_variance_ratio = None, output_path = None):
    """
    Render the PCA scree plot with both individual and cumulative explained
    variance so the user can choose how many components to retain.

    INPUTS:
        * 1d array of explained variance ratios in descending order
        * output file path for the scree image

    OUTPUTS:
        * none, file is written to disk
    """
    component_indices = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    figure_handle, axis_handle = plt.subplots(figsize = SCREE_FIGSIZE)
    axis_handle.bar(
        component_indices, explained_variance_ratio,
        color = "blue", alpha = SCREE_BAR_ALPHA, label = "individual",
    )
    axis_handle.plot(
        component_indices, cumulative_variance,
        color = "red", marker = "o", label = "cumulative",
    )
    axis_handle.set_xlabel("principal component")
    axis_handle.set_ylabel("explained variance ratio")
    axis_handle.set_title("PCA scree and cumulative explained variance")
    axis_handle.legend()
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)
