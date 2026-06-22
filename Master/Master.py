"""
Master.py

Renders the cross-asset Pearson correlation matrix as a heatmap PNG for the
report appendix. The numerical matrix is computed in rep_metrics.cpp over the
trailing 252 trading days of daily log returns and written to
rep_corr_matrix.csv; this file only loads that output and draws it, so every
value originates from the May_Rep pipeline and nothing is recomputed here.
"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parent.parent
CORRELATION_CSV_PATH = REPO_ROOT / "rep_corr_matrix.csv"
VISUALS_DIR = Path(__file__).resolve().parent / "visuals"
CORRELATION_PNG_PATH = VISUALS_DIR / "rep_correlation_matrix.png"

FIGURE_DPI = 150
FIGURE_SIZE = (8.0, 6.5)
CORRELATION_FLOOR = -1.0
CORRELATION_CEILING = 1.0
DARK_CELL_THRESHOLD = 0.5

# Red for negative, white for zero, green for positive correlation.
NEGATIVE_COLOUR = "#C1272D"
NEUTRAL_COLOUR = "#FFFFFF"
POSITIVE_COLOUR = "#386641"


def load_correlation_matrix(csv_path = None):
    """
    Load the correlation matrix written by rep_metrics.cpp.

    INPUTS:
        * path to rep_corr_matrix.csv

    OUTPUTS:
        * dataframe indexed and columned by asset name
    """
    return pd.read_csv(csv_path, index_col = 0)


def plot_correlation_heatmap(matrix = None, output_path = None):
    """
    Draw the correlation matrix as an annotated red to white to green heatmap.
    Cell text flips to white on strongly coloured cells so every coefficient
    stays legible.

    INPUTS:
        * correlation dataframe
        * output PNG path

    OUTPUTS:
        * none, a PNG is written to disk
    """
    colour_map = LinearSegmentedColormap.from_list(
        "red_white_green", [NEGATIVE_COLOUR, NEUTRAL_COLOUR, POSITIVE_COLOUR],
    )
    labels = list(matrix.columns)
    values = matrix.values

    figure_handle, axis_handle = plt.subplots(figsize = FIGURE_SIZE)
    heatmap_image = axis_handle.imshow(
        values, cmap = colour_map, vmin = CORRELATION_FLOOR, vmax = CORRELATION_CEILING,
    )
    axis_handle.set_xticks(range(len(labels)))
    axis_handle.set_yticks(range(len(labels)))
    axis_handle.set_xticklabels(labels, rotation = 45, ha = "right")
    axis_handle.set_yticklabels(labels)

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            cell_value = values[row_index, column_index]
            text_colour = "white" if abs(cell_value) > DARK_CELL_THRESHOLD else "black"
            axis_handle.text(
                column_index, row_index, f"{cell_value:+.2f}",
                ha = "center", va = "center", color = text_colour, fontsize = 9,
            )

    axis_handle.set_title("Cross-asset Pearson correlation, trailing 252 trading days")
    colour_bar = figure_handle.colorbar(heatmap_image, ax = axis_handle, fraction = 0.046, pad = 0.04)
    colour_bar.set_label("Pearson correlation")
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)


def main():
    """
    Load the C++ correlation output and render the heatmap PNG into the Master
    visuals directory.

    INPUTS:
        * none

    OUTPUTS:
        * none, the PNG is written to disk
    """
    VISUALS_DIR.mkdir(parents = True, exist_ok = True)
    correlation_matrix = load_correlation_matrix(csv_path = CORRELATION_CSV_PATH)
    plot_correlation_heatmap(matrix = correlation_matrix, output_path = CORRELATION_PNG_PATH)
    print(f"wrote {CORRELATION_PNG_PATH}")


if __name__ == "__main__":
    main()
