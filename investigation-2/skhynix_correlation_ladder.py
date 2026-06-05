"""
skhynix_correlation_ladder.py

Ranks a broad universe of individual assets by the strength of their daily
co-movement with SK Hynix (000660.KS), the headline KOSPI semiconductor name,
and renders the result as a single horizontal lollipop ladder of Spearman rank
correlations.

The universe is the same cross-asset set used for the NASDAQ scan (US sector,
industry and broad-index ETFs, country equity ETFs, US and international bond
ETFs and yields, crypto tokens and crypto-exposed equities) extended with the
series that matter for a Korean semiconductor position: USD/KRW and the home
KOSPI / KOSDAQ indices. The shared universe, the defensive download helpers
and the price-to-change transform are imported from the NASDAQ scraper so the
two ladders are built from one identical source.

USD/KRW is pinned onto the ladder regardless of where it ranks by strength. A
currency hedge for an SK Hynix position has to be read off the SK Hynix axis
directly, so the won is always drawn even though its daily rank correlation is
weak. It is shown with a star marker to separate it from the ranked field.

Sign convention:
    USD/KRW is quoted as KRW per USD, so a positive daily log return is KRW
    depreciation against the dollar. A negative Spearman against SK Hynix
    therefore means SK Hynix tends to fall on won-weakening, risk-off days.

Spearman rank correlation is the headline statistic because the universe mixes
price returns and yield changes; rank correlation is invariant to the monotone
difference in their scales. Price tickers enter as daily log returns and yield
tickers as daily first differences.

Run:
    python skhynix_correlation_ladder.py --start 2015-01-01 --end 2026-05-01
"""

# Imports
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

# Shared universe, fetch helpers and transform reused from the NASDAQ scraper
# so both ladders scan one identical asset set.
from inverse_pearson_rank_scraper import (
    PRICE_TICKERS,
    YIELD_TICKERS,
    AssetSpec,
    CorrelationRecord,
    download_one_safe,
    extract_close_series,
    to_daily_changes,
    CATEGORY_COLOURS,
    CATEGORY_DISPLAY_LABELS,
    MIN_OBS_FOR_CORRELATION,
    PANEL_FFILL_LIMIT,
)


# Constants

# SK Hynix is the correlation reference. Its daily log returns are the series
# every other asset is ranked against.
REFERENCE_NAME = "skhynix"
REFERENCE_SYMBOL = "000660.KS"

# USD/KRW is the pinned asset: always drawn on the ladder so the currency leg
# is visible even when it ranks outside the strongest field.
PINNED_ASSET_NAME = "usd_krw"

# Extra price tickers added on top of the shared universe. These are the
# Korea-specific series the NASDAQ scan does not carry: the won and the two
# home equity indices SK Hynix actually trades inside.
EXTRA_PRICE_TICKERS: Dict[str, Dict[str, str]] = {
    "fx": {
        "usd_krw": "KRW=X",
    },
    "home_market_equity": {
        "kospi_index":  "^KS11",
        "kosdaq_index": "^KQ11",
    },
}

# Number of strongest rungs (by absolute Spearman) drawn before the pinned
# USD/KRW row is guaranteed onto the ladder.
TOP_N_LADDER_BARS = 18

# Data alignment.
DEFAULT_START_DATE = "2015-01-01"

# Plotting: global font sizes.
TITLE_FONT = 14
TEXT_FONT = 10

# Plotting: colours for the two extra categories, kept inside the red / green
# / blue family used by the shared palette. The won sits on a distinct green
# and the home indices on a distinct blue.
FX_COLOUR = "#2D6A4F"
HOME_MARKET_COLOUR = "#274C77"
NEUTRAL_DARK = "#222222"
CAPTION_COLOUR = "dimgray"

# Merged colour and label maps: shared palette plus the two extra categories.
LADDER_CATEGORY_COLOURS: Dict[str, str] = {
    **CATEGORY_COLOURS,
    "fx":                 FX_COLOUR,
    "home_market_equity": HOME_MARKET_COLOUR,
}
LADDER_CATEGORY_LABELS: Dict[str, str] = {
    **CATEGORY_DISPLAY_LABELS,
    "fx":                 "FX (USD/KRW)",
    "home_market_equity": "Korean home-market equity index",
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
# separate from the NASDAQ scan that shares the universe definition.
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUBDIR = "skhynix_correlation_ladder"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / SCRIPT_SUBDIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" / SCRIPT_SUBDIR
VISUALS_DIR = PROJECT_ROOT / "visuals" / SCRIPT_SUBDIR

# Final deliverable copy. The curated conclusion directory holds the single
# headline PNG alongside the other investigation-2 visuals.
CONCLUSION_DIR = PROJECT_ROOT / "conclusion_2"
LADDER_FILE_NAME = "skhynix_spearman_ladder.png"


# Logging.
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("skhynix-correlation-ladder")


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


# Universe construction.

def build_reference_universe() -> List[AssetSpec]:
    """
    Flatten the shared price and yield ticker dictionaries plus the extra
    Korea-specific price tickers into a single list of AssetSpec records. The
    yield flag is carried through so the change transform knows to use first
    differences for yields and log returns for prices.

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


# Data acquisition.

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

    for asset_spec in universe:
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
    log.info("Fetched %d of %d universe assets", len(fetched_specs), len(universe))
    return panel, fetched_specs


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


def select_ladder_records(
    records: List[CorrelationRecord],
    top_n: int,
    pinned_name: str,
) -> List[CorrelationRecord]:
    """
    Select the rungs drawn on the ladder. The strongest top_n assets by
    absolute Spearman are taken first; the pinned asset (USD/KRW) is then
    forced into the set if the strength ranking did not already include it.
    The result is ordered by signed Spearman so the strongest inverse rung
    sits at the bottom and the strongest co-mover at the top.

    INPUTS:
        * records      : full list of CorrelationRecord for the universe
        * top_n        : number of strongest rungs to take by absolute rho
        * pinned_name  : asset name always guaranteed onto the ladder

    OUTPUTS:
        * Ordered list of CorrelationRecord to plot, bottom rung first.
    """
    by_strength = sorted(
        records, key = lambda record: abs(record.spearman), reverse = True,
    )
    chosen = list(by_strength[:top_n])
    chosen_names = {record.name for record in chosen}
    if pinned_name not in chosen_names:
        pinned_record = next(
            (record for record in records if record.name == pinned_name), None,
        )
        if pinned_record is not None:
            chosen.append(pinned_record)
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
            "name":     record.name,
            "symbol":   record.symbol,
            "category": record.category,
            "n_obs":    record.n_obs,
            "pearson":  record.pearson,
            "spearman": record.spearman,
        })
    return pd.DataFrame(rows)


# Plotting.

def add_caption(figure: plt.Figure, caption_text: str) -> None:
    """
    Place a descriptive caption underneath the figure so the saved PNG carries
    a self-contained explanation of the axis, the sign convention and the
    pinned USD/KRW rung.

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
    records: List[CorrelationRecord], pinned_name: str,
) -> Path:
    """
    Render the strongest Spearman rank correlations against SK Hynix as a
    horizontal lollipop ladder. Each rung is one asset coloured by category;
    the pinned USD/KRW rung is drawn with a star marker so the currency leg
    is unmistakable. Each marker is annotated with its rho value.

    INPUTS:
        * records      : ordered ladder records, bottom rung first
        * pinned_name  : asset name drawn with the highlighted star marker

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
        index for index, record in enumerate(records) if record.name == pinned_name
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
        f"(top {TOP_N_LADDER_BARS} by strength, USD/KRW pinned)"
    )

    legend_handles: List[Patch] = []
    for category_key in present_categories:
        legend_handles.append(Patch(
            facecolor = LADDER_CATEGORY_COLOURS.get(category_key, NEUTRAL_DARK),
            edgecolor = LADDER_MARKER_EDGE_COLOUR,
            label = LADDER_CATEGORY_LABELS.get(category_key, category_key),
        ))
    legend_handles.append(Line2D(
        [0], [0], marker = "*", color = "none",
        markerfacecolor = FX_COLOUR, markeredgecolor = LADDER_MARKER_EDGE_COLOUR,
        markersize = 16, label = "USD/KRW (pinned FX hedge candidate)",
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
        "value, with USD/KRW pinned on regardless of its rank. Positive rho "
        "marks co-movers and substitutes; negative rho marks inverse movers "
        "and hedge candidates. USD/KRW is quoted KRW per USD, so its negative "
        "rho means SK Hynix tends to fall on won-weakening risk-off days. "
        "Prices enter as daily log returns and yields as first differences.",
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
    ladder matches the look of the other investigation-2 figures.

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
    (including USD/KRW and the home indices), convert prices to log returns
    and yields to first differences, compute Pearson and Spearman against SK
    Hynix, persist the ranked table as CSV and Parquet, select the strongest
    rungs with USD/KRW pinned, render the ladder and copy it into the
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

    log.info("Computing Pearson and Spearman correlations against SK Hynix")
    records = compute_correlations_against_reference(changes, fetched_specs, REFERENCE_NAME)
    full_frame = records_to_frame(records)
    save_csv(full_frame, DATA_PROCESSED_DIR, "correlations_all_full_sample")
    save_parquet(full_frame, DATA_PROCESSED_DIR, "correlations_all_full_sample")

    ladder_records = select_ladder_records(records, TOP_N_LADDER_BARS, PINNED_ASSET_NAME)
    save_csv(records_to_frame(ladder_records), DATA_PROCESSED_DIR, "ladder_records")

    log.info("Plotting the SK Hynix Spearman ladder")
    ladder_path = plot_spearman_ladder(ladder_records, PINNED_ASSET_NAME)

    conclusion_path = CONCLUSION_DIR / LADDER_FILE_NAME
    conclusion_path.write_bytes(ladder_path.read_bytes())
    log.info("Copied ladder to %s", conclusion_path)

    pinned_record = next(
        (record for record in records if record.name == PINNED_ASSET_NAME), None,
    )
    summary: Dict[str, object] = {
        "run_at": datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        "start": str(full_panel.index.min().date()),
        "end": str(full_panel.index.max().date()),
        "universe_attempted": len(universe),
        "universe_fetched": len(fetched_specs),
        "top_n": TOP_N_LADDER_BARS,
        "usd_krw_spearman": None if pinned_record is None else pinned_record.spearman,
        "usd_krw_pearson": None if pinned_record is None else pinned_record.pearson,
        "usd_krw_n_obs": None if pinned_record is None else pinned_record.n_obs,
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
    covering the window, universe counts and the pinned USD/KRW correlation.

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
    print(f"USD/KRW Spearman     : {summary['usd_krw_spearman']}")
    print(f"USD/KRW Pearson      : {summary['usd_krw_pearson']}")
    print(f"\nVisuals in    : {VISUALS_DIR}")
    print(f"Conclusion in : {CONCLUSION_DIR}")
    print(f"Data in       : {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
