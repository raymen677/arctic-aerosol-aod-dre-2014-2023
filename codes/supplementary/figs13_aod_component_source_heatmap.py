"""Figure S13: AOD component and emission-region contribution heatmap.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: codes/main/fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True

# Resolve shared figure modules from either code subdirectory.
from pathlib import Path as _EPPath
_ep_root = _EPPath(__file__).resolve().parents[2]
for _ep_code_dir in (_ep_root / "codes" / "main", _ep_root / "codes" / "supplementary"):
    if str(_ep_code_dir) not in sys.path:
        sys.path.insert(0, str(_ep_code_dir))
from fig1_region import save_ep_figure, install_offline_geography
install_offline_geography()
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from fig2_arctic_aod import (
    DEFAULT_BASE,
    DEFAULT_NATURAL,
    FIGURE1_SOURCE_SPECS,
    ROOT_DIR,
    format_scientific_multiplier,
    get_species_field,
    nice_ceiling,
    scalar_weighted_mean,
    scientific_power,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "supplementary"
FIGURE_STEM = "fig2_anth_source_component_heatmap"

COMPONENT_LABELS = ["SU", "NI", "OC", "BC", "SS", "DU"]
COMPONENT_TO_SPECIES = {
    "SU": "SU",
    "NI": "AM+NI",
    "OC": "OC",
    "BC": "BC",
    "SS": "SS",
    "DU": "DU",
}
ANTHROPOGENIC_COMPONENT_LABELS = {"SU", "NI", "OC", "BC"}

SOURCE_REGION_LABELS = [
    "Total",
    "NAT",
    "ANT",
    "Local",
    "N. Europe",
    "Other Local",
    "Non-Local",
    "S. Europe",
    "E. Asia",
    "N. America",
    "Other Non-Local",
]

FONT_SIZES = {
    "tick": 11,
    "annotation": 10,
    "colorbar": 10,
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "fig2_source_component",
    ["#FBFAF6", "#E6E5DA", "#CBB37A", "#C1765B", "#583B3B"],
)
NAN_CELL_COLOR = "#EFECE5"
TEXT_COLOR = "#2E2A26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2 heatmap of Arctic AOD by source region and aerosol component."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Path to aod_gc_base.nc")
    parser.add_argument(
        "--natural",
        type=Path,
        default=DEFAULT_NATURAL,
        help="Path to aod_gc_noall.nc, used to calculate ANT as base - noall.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for exported figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1000,
        help="Export DPI for PNG files and embedded rasters in vector output.",
    )
    return parser.parse_args()


def build_heatmap_matrix(base: xr.Dataset, natural: xr.Dataset) -> np.ndarray:
    base_fields = {
        label: get_species_field(base, species)
        for label, species in COMPONENT_TO_SPECIES.items()
    }
    natural_fields = {
        label: get_species_field(natural, species)
        for label, species in COMPONENT_TO_SPECIES.items()
    }
    source_specs = {str(spec["label"]): spec for spec in FIGURE1_SOURCE_SPECS}

    rows = []
    for source_label in SOURCE_REGION_LABELS:
        if source_label == "Total":
            rows.append(
                [
                    scalar_weighted_mean(
                        base_fields[label]
                        if label in ANTHROPOGENIC_COMPONENT_LABELS
                        else natural_fields[label]
                    )
                    for label in COMPONENT_LABELS
                ]
            )
            continue

        if source_label == "NAT":
            rows.append([scalar_weighted_mean(natural_fields[label]) for label in COMPONENT_LABELS])
            continue

        if source_label == "ANT":
            rows.append(
                [
                    scalar_weighted_mean(base_fields[label] - natural_fields[label])
                    if label in ANTHROPOGENIC_COMPONENT_LABELS
                    else np.nan
                    for label in COMPONENT_LABELS
                ]
            )
            continue

        with xr.open_dataset(source_specs[source_label]["path"]) as removed:
            removed_fields = {
                label: get_species_field(removed, COMPONENT_TO_SPECIES[label])
                for label in ANTHROPOGENIC_COMPONENT_LABELS
            }
            rows.append(
                [
                    scalar_weighted_mean(base_fields[label] - removed_fields[label])
                    if label in ANTHROPOGENIC_COMPONENT_LABELS
                    else np.nan
                    for label in COMPONENT_LABELS
                ]
            )

    return np.asarray(rows, dtype=float)


def create_figure(matrix: np.ndarray, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    power = scientific_power(matrix)
    display_matrix = matrix / (10.0**power)
    vmax = nice_ceiling(float(np.nanmax(display_matrix)) * 1.04)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(NAN_CELL_COLOR)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    image = ax.imshow(display_matrix, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(COMPONENT_LABELS)))
    ax.set_xticklabels(COMPONENT_LABELS)
    ax.set_yticks(np.arange(len(SOURCE_REGION_LABELS)))
    ax.set_yticklabels(SOURCE_REGION_LABELS)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"], length=0, colors=TEXT_COLOR)
    ax.set_xticks(np.arange(-0.5, len(COMPONENT_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SOURCE_REGION_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in (0.5, 2.5, 5.5):
        ax.axhline(y, color="#746D64", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#FCFBF8")

    for row, col in np.ndindex(display_matrix.shape):
        value = display_matrix[row, col]
        if np.isnan(value):
            text = " - "
            color = "#7A746D"
        else:
            text = f"{value:.2f}"
            color = "white" if vmax > 0 and value / vmax >= 0.56 else TEXT_COLOR
        ax.text(
            col,
            row,
            text,
            ha="center",
            va="center",
            fontsize=FONT_SIZES["annotation"],
            color=color,
        )

    ax.text(
        1.0,
        1.025,
        format_scientific_multiplier(power),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZES["colorbar"],
        color=TEXT_COLOR,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.14,
        aspect=34,
        shrink=0.68,
    )
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=FONT_SIZES["colorbar"], width=0.5, length=2.2, pad=2)
    colorbar.ax.text(
        1.0,
        1.35,
        format_scientific_multiplier(power),
        transform=colorbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZES["colorbar"],
        color=TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.19)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = save_ep_figure(fig, output_dir / "figs13_aod_component_source_heatmap")
    plt.close(fig)
    return png_path, pdf_path


def print_matrix(matrix: np.ndarray) -> None:
    print("Arctic (>50 deg N) area-weighted mean AOD contribution")
    print("Rows: " + ", ".join(SOURCE_REGION_LABELS))
    print("Columns: " + ", ".join(COMPONENT_LABELS) + " (NI is AM+NI)")
    label_width = max(14, max(map(len, SOURCE_REGION_LABELS)) + 2)
    print("source".ljust(label_width) + "".join(label.rjust(12) for label in COMPONENT_LABELS))
    for label, row in zip(SOURCE_REGION_LABELS, matrix):
        values = [" - ".rjust(12) if np.isnan(value) else f"{value:12.5e}" for value in row]
        print(label.ljust(label_width) + "".join(values))


def main() -> None:
    args = parse_args()
    base = xr.open_dataset(args.base)
    natural = xr.open_dataset(args.natural)
    try:
        matrix = build_heatmap_matrix(base, natural)
        print_matrix(matrix)
        png_path, pdf_path = create_figure(matrix, args.output_dir, args.dpi)
        print(f"Saved figure 2 to {png_path}")
        print(f"Saved figure 2 to {pdf_path}")
    finally:
        base.close()
        natural.close()


if __name__ == "__main__":
    main()
