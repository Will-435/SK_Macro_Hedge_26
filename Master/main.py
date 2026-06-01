"""
main.py

Orchestrates the two stage USD KRW four week (twenty trading day) return
model. Stage one builds the stationary factor panel and fits PCA using the
pure functions in pca_factors.py. Stage two conditions on the current factor
state, runs the neighbourhood block bootstrap and produces the conditional
PDF and fan chart using the pure functions in bootstrap_pdf.py.

This file owns all input output: directory creation, Parquet persistence and
the run parameters. pca_factors.py and bootstrap_pdf.py contain only function
and constant definitions and perform no side effects when imported.
"""


import os
import numpy as np
import pandas as pd

import pca_factors
import bootstrap_pdf


# Run parameters for the modelling pipeline.
LOOKBACK_YEARS = 10
FORECAST_HORIZON_DAYS = 20
N_PCS_USED = 3
N_NEIGHBOURS = 200
N_BOOTSTRAP_SAMPLES = 5000
EXPECTED_BLOCK_LENGTH_DAYS = 5
LOWER_INTERVAL_QUANTILE = 0.10
UPPER_INTERVAL_QUANTILE = 0.90

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

RAW_PANEL_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "raw_factor_panel.parquet")
STANDARDISED_PANEL_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "standardised_panel.parquet")
SCALER_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "scaler_stats.parquet")
SCORES_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "pc_scores.parquet")
LOADINGS_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "pc_loadings.parquet")
EXPLAINED_VARIANCE_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "explained_variance.parquet")
NEIGHBOUR_RETURNS_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "neighbour_forward_returns.parquet")
PDF_PARQUET_PATH = os.path.join(DATA_PROCESSED_DIR, "conditional_pdf.parquet")
SCREE_PLOT_PATH = os.path.join(VISUALS_DIR, "pca_scree.png")
FAN_CHART_PATH = os.path.join(VISUALS_DIR, "usdkrw_fan_chart.png")
CONDITIONAL_PDF_PATH = os.path.join(VISUALS_DIR, "usdkrw_conditional_pdf.png")


def ensure_output_directories():
    """
    Create the data and visuals output directories if they do not exist.

    INPUTS:
        * none

    OUTPUTS:
        * none, directories are created on disk as a side effect
    """
    for directory_path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALS_DIR):
        os.makedirs(directory_path, exist_ok = True)


def run_pca_stage():
    """
    Build the raw factor panel, apply the per feature stationarity
    transforms, standardise, fit PCA and persist every artefact plus the
    scree visual. Returns the fitted PCA results for optional inspection.

    INPUTS:
        * none

    OUTPUTS:
        * dictionary of PCA results from pca_factors.fit_pca
    """
    raw_panel = pca_factors.build_raw_factor_panel(lookback_years = LOOKBACK_YEARS)
    raw_panel.to_parquet(RAW_PANEL_PARQUET_PATH)

    stationary_panel = pca_factors.apply_feature_transforms(raw_panel = raw_panel)

    standardised_panel, feature_means, feature_stds = pca_factors.standardise_panel(
        stationary_panel = stationary_panel,
    )
    standardised_panel.to_parquet(STANDARDISED_PANEL_PARQUET_PATH)

    scaler_stats = pd.DataFrame({"mean": feature_means, "std": feature_stds})
    scaler_stats.to_parquet(SCALER_PARQUET_PATH)

    pca_results = pca_factors.fit_pca(standardised_panel = standardised_panel)
    pca_results["scores"].to_parquet(SCORES_PARQUET_PATH)
    pca_results["loadings"].to_parquet(LOADINGS_PARQUET_PATH)

    explained_variance_ratio = pca_results["explained_variance_ratio"]
    explained_variance_frame = pd.DataFrame({
        "component": [f"pc{component_index + 1}" for component_index in range(len(explained_variance_ratio))],
        "explained_variance_ratio": explained_variance_ratio,
    })
    explained_variance_frame.to_parquet(EXPLAINED_VARIANCE_PARQUET_PATH)

    pca_factors.plot_scree(
        explained_variance_ratio = explained_variance_ratio,
        output_path = SCREE_PLOT_PATH,
    )

    print("PCA stage completed")
    print(f"  raw panel:           {RAW_PANEL_PARQUET_PATH}")
    print(f"  standardised panel:  {STANDARDISED_PANEL_PARQUET_PATH}")
    print(f"  scaler stats:        {SCALER_PARQUET_PATH}")
    print(f"  pc scores:           {SCORES_PARQUET_PATH}")
    print(f"  loadings:            {LOADINGS_PARQUET_PATH}")
    print(f"  explained variance:  {EXPLAINED_VARIANCE_PARQUET_PATH}")
    print(f"  scree plot:          {SCREE_PLOT_PATH}")
    print(f"  variance ratios:     {np.round(explained_variance_ratio, 4)}")
    return pca_results


def run_bootstrap_stage():
    """
    Load the PCA artefacts, find the neighbourhood of the current factor
    state in PC space, run the stationary block bootstrap on the matched
    forward returns and persist the conditional PDF, the neighbour set, the
    fan chart and the PDF plot.

    INPUTS:
        * none

    OUTPUTS:
        * none, all results are written to data/processed and visuals
    """
    pc_scores = bootstrap_pdf.load_pc_history(scores_parquet_path = SCORES_PARQUET_PATH)
    raw_panel = bootstrap_pdf.load_raw_panel(raw_panel_parquet_path = RAW_PANEL_PARQUET_PATH)

    usdkrw_prices = raw_panel["usdkrw"].dropna()
    forward_log_returns = bootstrap_pdf.compute_forward_log_returns(
        price_series = usdkrw_prices,
        horizon_days = FORECAST_HORIZON_DAYS,
    )

    conditioning_frame = bootstrap_pdf.build_conditioning_sample(
        pc_scores = pc_scores,
        forward_returns = forward_log_returns,
        n_components = N_PCS_USED,
    )

    history_component_columns = [f"pc{component_index + 1}" for component_index in range(N_PCS_USED)]
    history_pcs = conditioning_frame[history_component_columns]
    forward_returns_aligned = conditioning_frame["forward_log_return"]

    current_pc_vector = pc_scores.iloc[-1, :N_PCS_USED].values

    neighbour_dates, neighbour_distances = bootstrap_pdf.find_neighbourhood(
        history_pcs = history_pcs,
        current_pc = current_pc_vector,
        n_neighbours = N_NEIGHBOURS,
    )

    date_sort_positions = neighbour_dates.argsort()
    sorted_neighbour_dates = neighbour_dates[date_sort_positions]
    sorted_neighbour_distances = neighbour_distances[date_sort_positions]
    neighbour_returns = forward_returns_aligned.loc[sorted_neighbour_dates]

    neighbour_frame = pd.DataFrame({
        "date": sorted_neighbour_dates,
        "forward_log_return": neighbour_returns.values,
        "mahalanobis_distance": sorted_neighbour_distances,
    })
    neighbour_frame.to_parquet(NEIGHBOUR_RETURNS_PARQUET_PATH, index = False)

    pooled_returns, kernel_density = bootstrap_pdf.conditional_pdf_from_neighbours(
        neighbour_forward_returns = neighbour_returns.values,
        n_bootstrap_samples = N_BOOTSTRAP_SAMPLES,
        expected_block_length = EXPECTED_BLOCK_LENGTH_DAYS,
    )

    pdf_grid = np.linspace(pooled_returns.min(), pooled_returns.max(), bootstrap_pdf.DENSITY_GRID_RESOLUTION)
    pdf_frame = pd.DataFrame({
        "log_return": pdf_grid,
        "density": kernel_density(pdf_grid),
    })
    pdf_frame.to_parquet(PDF_PARQUET_PATH, index = False)

    spot_price = float(usdkrw_prices.iloc[-1])

    bootstrap_pdf.plot_fan_chart(
        spot_price = spot_price,
        sampled_log_returns = pooled_returns,
        kernel_density = kernel_density,
        horizon_days = FORECAST_HORIZON_DAYS,
        output_path = FAN_CHART_PATH,
    )

    bootstrap_pdf.plot_conditional_pdf(
        sampled_log_returns = pooled_returns,
        kernel_density = kernel_density,
        output_path = CONDITIONAL_PDF_PATH,
    )

    central_interval = np.quantile(pooled_returns, [LOWER_INTERVAL_QUANTILE, UPPER_INTERVAL_QUANTILE])
    mean_log_return = float(np.mean(pooled_returns))

    print("bootstrap stage completed")
    print(f"  neighbour returns:   {NEIGHBOUR_RETURNS_PARQUET_PATH}")
    print(f"  conditional pdf:     {PDF_PARQUET_PATH}")
    print(f"  fan chart:           {FAN_CHART_PATH}")
    print(f"  conditional pdf plot:{CONDITIONAL_PDF_PATH}")
    print(f"  spot:                {spot_price:.4f}")
    print(f"  mean 4w log return:  {mean_log_return:.4f}")
    print(f"  80 percent interval: [{central_interval[0]:.4f}, {central_interval[1]:.4f}]")


def main():
    """
    Run the full pipeline end to end: prepare directories, build and fit
    PCA, then condition and bootstrap the four week USD KRW return density.

    INPUTS:
        * none

    OUTPUTS:
        * none, all results are written to disk
    """
    ensure_output_directories()
    run_pca_stage()
    run_bootstrap_stage()


if __name__ == "__main__":
    main()
