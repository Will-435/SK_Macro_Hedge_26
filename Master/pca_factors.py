"""
pca_factors.py

Builds a stationary, daily frequency panel of factors relevant to a four week
(twenty trading day) USD KRW return forecast, then reduces the panel to
principal components for downstream conditional sampling in bootstrap_pdf.py.

DATA SOURCE ACCESS CONSTRAINT

The Bank of Korea ECOS API is restricted to Korean residents and nationals,
so it cannot be used here. Every factor is therefore drawn from sources that
are free and open to non Korean users, namely yfinance and the Federal
Reserve FRED API. The consequences of this constraint are recorded below.

FACTORS REPLACED OR DROPPED RELATIVE TO AN ECOS BUILD

Daily foreign equity and bond flows are an ECOS only daily series. They are
replaced by the United States listed iShares MSCI South Korea ETF (EWY) as a
free daily proxy for foreign appetite toward Korean equities. EWY is priced
in dollars and moves with foreign positioning, so its log return is a usable
stand in for net foreign flow direction. It is a price based proxy, not a
true flow measurement, which is a known limitation.

The Korea ten year government bond yield is daily on ECOS. The free
substitute is the OECD ten year government bond yield for Korea on FRED
(IRLTLT01KRM156N), which is monthly. This is a frequency downgrade: the daily
yield path is lost and the monthly value is forward filled across the daily
grid, so the level and the derived yield spread only step once per month.

Trade balance, current account and resident foreign currency deposits are
ECOS monthly series with no free, currently maintained equivalent for non
Korean users. The OECD trade balance series for Korea on FRED is discontinued
(it stops in late 2024) and the current account series is quarterly, so both
would be stale or near constant across the recent window that the model
conditions on. They are dropped rather than wired up as misleading stale
columns.

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


FRED_API = "cfa3f4e2b7a6b802ab2df38002ecca10"

LOOKBACK_YEARS = 10
LOOKBACK_BUFFER_DAYS = 60
DAYS_PER_YEAR_CALENDAR = 365.25

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_MISSING_VALUE_STRING = "."

FIGURE_DPI = 150
SCREE_FIGSIZE = (10, 6)
SCREE_BAR_ALPHA = 0.6

# yfinance tickers. The map key is the internal feature name carried through
# the rest of the pipeline. The value is the yfinance ticker symbol. EWY is
# the iShares MSCI South Korea ETF, used as a free daily proxy for foreign
# appetite toward Korean equities in place of the ECOS only foreign flow data.
YFINANCE_TICKERS = {
    "usdkrw": "KRW=X",
    "dxy": "DX-Y.NYB",
    # Offshore USD CNH (CNH=X) is not currently served by yfinance, so onshore
    # USD CNY (CNY=X) is used instead. The two track within a few pips and are
    # interchangeable as a renminbi co movement signal for the won.
    "usdcny": "CNY=X",
    "brent": "BZ=F",
    "wti": "CL=F",
    "natgas_henry_hub": "NG=F",
    "kospi": "^KS11",
    "semis_etf": "SOXX",
    "vix": "^VIX",
    "korea_equity_etf": "EWY",
}

# FRED daily series. The map key is the internal feature name and the value is
# the FRED series identifier.
FRED_DAILY_SERIES = {
    "us_10y_yield": "DGS10",
}

# FRED monthly macro series. The Korea ten year yield is the only free,
# currently maintained Korean macro series available to non Korean users. It
# is monthly and must be lagged to release and forward filled onto the daily
# grid.
FRED_MONTHLY_MACRO_SERIES = {
    "kr_10y_yield": "IRLTLT01KRM156N",
}

# Calendar lag between a monthly macro reference period end and the real world
# release date. This guards against forward looking leakage from a monthly
# value being stamped onto dates before it was actually published. Thirty days
# is a conservative allowance for the OECD republication delay on FRED.
KOREA_MACRO_RELEASE_LAG_DAYS = {
    "kr_10y_yield": 30,
}


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
        if observation["value"] == FRED_MISSING_VALUE_STRING:
            continue
        parsed_dates.append(pd.Timestamp(observation["date"]))
        parsed_values.append(float(observation["value"]))
    return pd.Series(
        data = np.array(parsed_values, dtype = float),
        index = pd.DatetimeIndex(parsed_dates),
        name = series_code,
    )


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
    # Non positive prints are data feed glitches, not real prices. Mask them to
    # missing so the logarithm stays defined and no spurious return is created.
    positive_prices = price_series.where(price_series > 0)
    return np.log(positive_prices).diff()


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


# Strictly positive price and index series take log returns. Yields and the
# derived yield spread take first differences because they can sit near zero
# or turn negative, where a logarithm is unstable or undefined. No signed flow
# transform is needed because the ECOS flow series have been replaced by the
# EWY price proxy, which is itself a strictly positive price.
FEATURE_TRANSFORMS = {
    "usdkrw":               transform_log_returns,
    "dxy":                  transform_log_returns,
    "usdcny":               transform_log_returns,
    "brent":                transform_log_returns,
    "wti":                  transform_log_returns,
    "natgas_henry_hub":     transform_log_returns,
    "kospi":                transform_log_returns,
    "semis_etf":            transform_log_returns,
    "vix":                  transform_log_returns,
    "korea_equity_etf":     transform_log_returns,
    "us_10y_yield":         transform_first_difference,
    "kr_10y_yield":         transform_first_difference,
    "kr_us_yield_spread":   transform_first_difference,
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

    fred_daily_frames = {}
    for series_name, series_code in FRED_DAILY_SERIES.items():
        daily_series = fetch_fred_series(
            series_code = series_code,
            start_date = start_date.isoformat(),
            end_date = end_date.isoformat(),
        )
        daily_series.name = series_name
        fred_daily_frames[series_name] = daily_series

    fred_monthly_frames = {}
    for series_name, series_code in FRED_MONTHLY_MACRO_SERIES.items():
        monthly_raw = fetch_fred_series(
            series_code = series_code,
            start_date = start_date.isoformat(),
            end_date = end_date.isoformat(),
        )
        # FRED stamps a monthly observation at the first of the month. Snap to
        # the month end reference period before applying the release lag.
        monthly_raw.index = monthly_raw.index + pd.offsets.MonthEnd(0)
        monthly_raw.name = series_name
        fred_monthly_frames[series_name] = lag_macro_to_release_date(
            monthly_series = monthly_raw,
            lag_days = KOREA_MACRO_RELEASE_LAG_DAYS[series_name],
        )

    business_day_index = pd.bdate_range(start = start_date, end = end_date)
    factor_panel = pd.DataFrame(index = business_day_index)

    for column_name in market_prices.columns:
        factor_panel[column_name] = market_prices[column_name].reindex(business_day_index)

    for series_name, series_data in fred_daily_frames.items():
        factor_panel[series_name] = series_data.reindex(business_day_index).ffill()

    for series_name, series_data in fred_monthly_frames.items():
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
    # Drop any column that is entirely missing before the row wise drop. A
    # single dead data feed would otherwise turn every row into a row with a
    # missing value and silently empty the whole panel.
    populated_panel = standardised_panel.dropna(axis = 1, how = "all")
    clean_panel = populated_panel.dropna(how = "any")
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
