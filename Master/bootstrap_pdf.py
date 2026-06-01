"""
bootstrap_pdf.py

Builds the conditional probability density of the four week (twenty
trading day) USD KRW log return given the current factor state. The
conditioning mechanism is a nearest neighbour lookup in principal
component space using the PCA artefacts from pca_factors.py, followed by
a stationary block bootstrap on the realised forward returns at the
matched dates.

RELOCATED STATIONARITY ASSUMPTION

Two historical dates that share PC coordinates are assumed to share an
outcome distribution for the next twenty trading days. Equivalently, the
joint distribution of factors and forward USD KRW return is assumed
stable over the lookback window. If the regime drifts, for example a
structural intervention by the Bank of Korea or a sustained dollar bull
cycle that did not appear in the backtest, this assumption fails and the
conditional distribution will be miscalibrated.

NEIGHBOURHOOD RADIUS BIAS VARIANCE TRADE OFF

The number of neighbours retained from history is a bias variance trade
off. A small neighbourhood gives the most relevant historical analogues
but the resulting sample is tiny and the empirical density is noisy. A
large neighbourhood loses specificity and pulls the conditional density
back towards the unconditional distribution. The radius is exposed as
the constant N_NEIGHBOURS for tuning.

BLOCK BOOTSTRAP REASONING

The conditioning sample consists of overlapping twenty trading day
forward returns. Adjacent dates share most of their forward window so
the sample is strongly serially correlated. An independent identically
distributed bootstrap would treat these draws as independent and would
underestimate the spread, producing intervals that are systematically
too tight. The stationary bootstrap of Politis and Romano preserves
local dependence by resampling contiguous blocks whose lengths follow a
geometric distribution, recovering the correct uncertainty for the
endpoint distribution.

FAN CHART INTERPRETATION

A four week return is a single endpoint quantity. Every ray drawn on
the fan chart is a straight line from spot to a sampled endpoint price.
It is a fan of endpoints, not an intra period path simulation. No claim
is made about the shape of the trajectory between today and the
twentieth trading day.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis


FAN_CHART_RAY_COUNT = 400
DENSITY_GRID_RESOLUTION = 256
DENSITY_NORMALISATION_EPSILON = 1.0e-12
RANDOM_SEED = 20260601

FIGURE_DPI = 150
FAN_CHART_FIGSIZE = (12, 7)
PDF_FIGSIZE = (10, 6)
RAY_ALPHA = 0.25
RAY_LINEWIDTH = 0.8
MEAN_RAY_LINEWIDTH = 2.5
PDF_FILL_ALPHA = 0.3


def load_pc_history(scores_parquet_path = None):
    """
    Load the principal component score history persisted by the PCA stage.

    INPUTS:
        * path to the PC scores Parquet file

    OUTPUTS:
        * dataframe of PC scores indexed by date
    """
    return pd.read_parquet(scores_parquet_path)


def load_raw_panel(raw_panel_parquet_path = None):
    """
    Load the raw factor panel persisted by the PCA stage. Required to
    recover the USD KRW spot price series used to compute forward returns
    and to anchor the fan chart.

    INPUTS:
        * path to the raw factor panel Parquet file

    OUTPUTS:
        * dataframe of raw factor levels indexed by date
    """
    return pd.read_parquet(raw_panel_parquet_path)


def compute_forward_log_returns(price_series = None, horizon_days = None):
    """
    Compute the forward log return at every date. The value at time t is
    the log of price at t plus horizon minus the log of price at t. Dates
    near the end of the sample where the horizon extends beyond the
    available data become NaN and are dropped downstream.

    INPUTS:
        * price series of USD KRW close levels
        * forecast horizon in trading days

    OUTPUTS:
        * series of forward log returns aligned to the start date t
    """
    return np.log(price_series.shift(- horizon_days)) - np.log(price_series)


def build_conditioning_sample(pc_scores = None, forward_returns = None, n_components = None):
    """
    Align PC scores with the realised forward return at each date. Returns
    only those dates where both the PC vector and the forward return are
    observable, which excludes the last horizon worth of trading days.

    INPUTS:
        * dataframe of PC scores
        * series of forward log returns
        * number of leading principal components to retain

    OUTPUTS:
        * dataframe with the first n components and a forward log return
          column, indexed by date
    """
    selected_components = pc_scores.iloc[:, :n_components]
    aligned_frame = selected_components.join(
        forward_returns.rename("forward_log_return"),
        how = "inner",
    )
    return aligned_frame.dropna()


def find_neighbourhood(history_pcs = None, current_pc = None, n_neighbours = None):
    """
    Find the n historical dates nearest to the current PC vector using
    Mahalanobis distance. Mahalanobis whitens the PC covariance so a
    high variance component does not dominate the metric. This is the
    eigenvalue scaled distance recommended for PCA neighbour searches.

    INPUTS:
        * dataframe of historical PC vectors indexed by date
        * current PC vector as a 1d array
        * neighbourhood size

    OUTPUTS:
        * DatetimeIndex of the nearest historical dates
        * 1d array of the corresponding Mahalanobis distances
    """
    history_matrix = history_pcs.values
    sample_covariance = np.cov(history_matrix, rowvar = False)
    inverse_covariance = np.linalg.pinv(sample_covariance)
    current_vector = np.asarray(current_pc).ravel()
    row_count = history_matrix.shape[0]
    distances = np.empty(row_count)
    for row_index in range(row_count):
        distances[row_index] = mahalanobis(
            history_matrix[row_index], current_vector, inverse_covariance,
        )
    sorted_positions = np.argsort(distances)
    nearest_positions = sorted_positions[:n_neighbours]
    return history_pcs.index[nearest_positions], distances[nearest_positions]


def stationary_bootstrap_sample(observations = None, sample_count = None, expected_block_length = None, random_state = None):
    """
    Generate stationary block bootstrap resamples of a one dimensional
    observation array. Each resample has the same length as the input.
    Block boundaries are drawn from a geometric distribution with mean
    expected block length, which preserves local serial dependence while
    keeping the resampling stationary.

    INPUTS:
        * observations as a 1d array of forward returns
        * number of bootstrap resamples to generate
        * expected geometric block length L, geometric probability is 1 / L
        * integer seed for the numpy random generator

    OUTPUTS:
        * 2d array of shape (sample count, observation count) of resamples
    """
    random_generator = np.random.default_rng(random_state)
    observations_array = np.asarray(observations)
    observation_count = observations_array.size
    geometric_probability = 1.0 / expected_block_length
    output_samples = np.empty((sample_count, observation_count))
    for sample_index in range(sample_count):
        current_position = int(random_generator.integers(0, observation_count))
        for time_index in range(observation_count):
            output_samples[sample_index, time_index] = observations_array[current_position]
            if random_generator.random() < geometric_probability:
                current_position = int(random_generator.integers(0, observation_count))
            else:
                current_position = (current_position + 1) % observation_count
    return output_samples


def conditional_pdf_from_neighbours(neighbour_forward_returns = None, n_bootstrap_samples = None, expected_block_length = None):
    """
    Apply the stationary block bootstrap to the neighbour subset of
    forward log returns and fit a Gaussian kernel density to the pooled
    bootstrap output. The pooled distribution is the empirical conditional
    PDF of the four week USD KRW log return.

    INPUTS:
        * neighbour forward returns sorted by historical date
        * number of bootstrap resamples
        * expected block length for the stationary bootstrap

    OUTPUTS:
        * 1d array of the pooled bootstrap sample
        * fitted scipy gaussian kde object
    """
    bootstrap_matrix = stationary_bootstrap_sample(
        observations = neighbour_forward_returns,
        sample_count = n_bootstrap_samples,
        expected_block_length = expected_block_length,
        random_state = RANDOM_SEED,
    )
    pooled_returns = bootstrap_matrix.ravel()
    kernel_density = stats.gaussian_kde(pooled_returns)
    return pooled_returns, kernel_density


def plot_fan_chart(spot_price = None, sampled_log_returns = None, kernel_density = None, horizon_days = None, output_path = None):
    """
    Render the fan chart. Each ray is a straight line from spot to a
    sampled endpoint price coloured by bootstrap density on a red to
    green gradient. A single black ray marks the mean endpoint. Rays are
    endpoints only, not intra period paths.

    INPUTS:
        * current USD KRW spot price
        * pooled bootstrap log returns
        * fitted kernel density on the same pooled sample
        * forecast horizon in trading days
        * output file path

    OUTPUTS:
        * none, file is written to disk
    """
    random_generator = np.random.default_rng(RANDOM_SEED)
    selected_count = min(FAN_CHART_RAY_COUNT, sampled_log_returns.size)
    selected_positions = random_generator.choice(
        sampled_log_returns.size, size = selected_count, replace = False,
    )
    selected_log_returns = sampled_log_returns[selected_positions]
    density_values = kernel_density(selected_log_returns)
    density_range = density_values.max() - density_values.min() + DENSITY_NORMALISATION_EPSILON
    normalised_densities = (density_values - density_values.min()) / density_range

    figure_handle, axis_handle = plt.subplots(figsize = FAN_CHART_FIGSIZE)
    time_axis = np.arange(0, horizon_days + 1)

    for ray_index in range(selected_count):
        endpoint_price = spot_price * np.exp(selected_log_returns[ray_index])
        ray_levels = np.linspace(spot_price, endpoint_price, horizon_days + 1)
        density_weight = normalised_densities[ray_index]
        # Red green gradient: high density maps to green, low density to red.
        ray_colour = (1.0 - density_weight, density_weight, 0.0)
        axis_handle.plot(
            time_axis, ray_levels,
            color = ray_colour, alpha = RAY_ALPHA, linewidth = RAY_LINEWIDTH,
        )

    mean_log_return = float(np.mean(sampled_log_returns))
    mean_endpoint_price = spot_price * np.exp(mean_log_return)
    axis_handle.plot(
        time_axis,
        np.linspace(spot_price, mean_endpoint_price, horizon_days + 1),
        color = "black", linewidth = MEAN_RAY_LINEWIDTH, label = "mean endpoint ray",
    )

    axis_handle.set_xlabel("trading days ahead")
    axis_handle.set_ylabel("USD KRW level")
    axis_handle.set_title("USD KRW endpoint fan from spot, coloured by bootstrap density")
    axis_handle.legend()
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)


def plot_conditional_pdf(sampled_log_returns = None, kernel_density = None, output_path = None):
    """
    Render the conditional PDF of the four week log return. The mean of
    the bootstrap pool is overlaid as a dashed vertical reference line.

    INPUTS:
        * pooled bootstrap log returns
        * fitted kernel density on the same pooled sample
        * output file path

    OUTPUTS:
        * none, file is written to disk
    """
    log_return_grid = np.linspace(
        sampled_log_returns.min(), sampled_log_returns.max(),
        DENSITY_GRID_RESOLUTION,
    )
    density_values = kernel_density(log_return_grid)
    figure_handle, axis_handle = plt.subplots(figsize = PDF_FIGSIZE)
    axis_handle.fill_between(log_return_grid, density_values, color = "blue", alpha = PDF_FILL_ALPHA)
    axis_handle.plot(log_return_grid, density_values, color = "blue")
    axis_handle.axvline(
        float(np.mean(sampled_log_returns)),
        color = "black", linestyle = "--", label = "mean",
    )
    axis_handle.set_xlabel("four week USD KRW log return")
    axis_handle.set_ylabel("density")
    axis_handle.set_title("conditional pdf of twenty trading day USD KRW log return")
    axis_handle.legend()
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)
