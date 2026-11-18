"""Figure 5: Surface and TOA shortwave and longwave component DRE heatmaps.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""
#!/usr/bin/env python3

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from fig1_region import save_ep_figure, install_offline_geography
install_offline_geography()

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image

warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_DIR = Path(__file__).resolve().parent / "data" / "radiation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "main"
MM_PER_INCH = 20
NATURE_DOUBLE_COLUMN_WIDTH_MM = 183.0
NATURE_MAX_HEIGHT_MM = 170.0
FIGURE_WIDTH_IN = NATURE_DOUBLE_COLUMN_WIDTH_MM / MM_PER_INCH
FIGURE_HEIGHT_IN = NATURE_MAX_HEIGHT_MM / MM_PER_INCH
HEATMAP_WIDTH_STRETCH = 0.72
SW_COLORBAR_TICK_STEP = 0.2
LW_COLORBAR_TICK_STEP = 0.02

FILES = {
    "base": BASE_DIR / "rad_gc_L0_base.nc",
    "noall": BASE_DIR / "rad_gc_L1_noall.nc",
    "local_nonp": BASE_DIR / "rad_gc_L2_local_nonp.nc",
    "trans_nonp_sup": BASE_DIR / "rad_gc_L2_trans_nonp_sup.nc",
    "local_no_north_europe": BASE_DIR / "rad_gc_L3_local_no_north_europe.nc",
    "local_no_north_europe_sup": BASE_DIR / "rad_gc_L3_local_no_north_europe_sup.nc",
    "trans_no_south_europe": BASE_DIR / "rad_gc_L3_trans_no_south_europe.nc",
    "trans_no_east_asia": BASE_DIR / "rad_gc_L3_trans_no_east_asia.nc",
    "trans_no_north_america": BASE_DIR / "rad_gc_L3_trans_no_north_america.nc",
    "trans_no_three_regions_sup": BASE_DIR / "rad_gc_L3_trans_no_three_regions_sup.nc",
}

CONTRIB_SCENARIOS = [
    "noall",
    "local_nonp",
    "local_no_north_europe",
    "local_no_north_europe_sup",
    "trans_nonp_sup",
    "trans_no_south_europe",
    "trans_no_east_asia",
    "trans_no_north_america",
    "trans_no_three_regions_sup",
]
HEATMAP_SCENARIOS = ["total", "natural", *CONTRIB_SCENARIOS]
CONTRIB_SHORT_LABELS = {
    "total": "Total",
    "natural": "NAT",
    "noall": "ANT",
    "local_nonp": "Local",
    "trans_nonp_sup": "Non-Local",
    "local_no_north_europe": "N. Europe",
    "local_no_north_europe_sup": "Other Local",
    "trans_no_south_europe": "S. Europe",
    "trans_no_east_asia": "E. Asia",
    "trans_no_north_america": "N. America",
    "trans_no_three_regions_sup": "Other Non-Local",
}
HEATMAP_GROUPS = [
    {"label": "Major", "start": 0, "end": 2, "facecolor": "#FBFAF7"},
    {"label": "Local", "start": 3, "end": 5, "facecolor": "#F1F6FB"},
    {"label": "Trans", "start": 6, "end": 10, "facecolor": "#FEF6EC"},
]
RADIATION_TYPES = [
    ("SW-Surf", "SW", "Surf"),
    ("SW-TOA", "SW", "TOA"),
    ("LW-Surf", "LW", "Surf"),
    ("LW-TOA", "LW", "TOA"),
]
RAD_LABELS = {
    "SW-Surf": "SW Surf",
    "SW-TOA": "SW TOA",
    "LW-Surf": "LW Surf",
    "LW-TOA": "LW TOA",
}
PLOTTED_COMPONENTS = ["PM", "BC", "OA", "SU", "AM+NI"]
NATURAL_ONLY_COMPONENTS = {"DU", "SS"}
HEATMAP_COMPONENTS = ["SU", "AM+NI", "OA", "BC", "SS", "DU"]
COMPONENT_DISPLAY_LABELS = {
    "PM": "Total",
    "BC": "BC",
    "OA": "OC",
    "SU": "SU",
    "AM+NI": "NI",
    "DU": "DU",
    "SS": "SS",
}
COMPONENT_COLORS = {
    "PM": "#000000",
    "BC": "#D55E00",
    "OA": "#009E73",
    "SU": "#0072B2",
    "AM+NI": "#CC79A7",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "nature_diverging",
    ["#2B6F97", "#F7F4EF", "#B44B48"],
    N=256,
)
FONT_SIZES = {
    "base": 15.0,
    "panel_label": 15.0,
    "axes_title": 15.0,
    "axes_label": 15.0,
    "default_xtick": 10.0,
    "default_ytick": 10.0,
    "heatmap_xtick": 10.0,
    "heatmap_ytick": 10.0,
    "heatmap_group_label": 9.0,
    "heatmap_scale_annotation": 10.0,
    "heatmap_cell": 10.0,
    "colorbar_label": 10.0,
    "sw_colorbar_tick": 10.0,
    "lw_colorbar_tick": 10.0,
    "lw_colorbar_scale_title": 10.0,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZES["base"],
        "text.usetex": False,
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.titlesize": FONT_SIZES["axes_title"],
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZES["axes_label"],
        "axes.linewidth": 0.4,
        "axes.unicode_minus": False,
        "xtick.labelsize": FONT_SIZES["default_xtick"],
        "ytick.labelsize": FONT_SIZES["default_ytick"],
        "xtick.major.width": 0.35,
        "ytick.major.width": 0.35,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2 for Arctic radiative effects.")
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
        help="Export DPI for PNG files and embedded rasters.",
    )
    return parser.parse_args()


def load_datasets() -> dict[str, xr.Dataset]:
    datasets: dict[str, xr.Dataset] = {}
    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input dataset: {path}")
        print(f"Loading {name}: {path.name}")
        datasets[name] = xr.open_dataset(path)
    return datasets


def close_datasets(datasets: dict[str, xr.Dataset]) -> None:
    for dataset in datasets.values():
        dataset.close()


def arctic_subset(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    return data.where(data["lat"] > 50.0, drop=True)


def area_weighted_mean(data: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(data["lat"]))
    return data.weighted(weights).mean(dim=("lat", "lon"))


def get_re(dataset: xr.Dataset, wave: str, location: str, component: str) -> xr.DataArray:
    if component == "AM+NI":
        return dataset[f"RadAllSky{wave}{location}_AM"] + dataset[f"RadAllSky{wave}{location}_NI"]
    return dataset[f"RadAllSky{wave}{location}_{component}"]


def annual_mean_value(dataset: xr.Dataset, wave: str, location: str, component: str) -> float:
    field = get_re(dataset, wave, location, component)
    scalar = area_weighted_mean(arctic_subset(field)).mean()
    return float(scalar.values)


def annual_mean_contribution(
    base_dataset: xr.Dataset,
    sensitivity_dataset: xr.Dataset,
    wave: str,
    location: str,
    component: str,
) -> float:
    return annual_mean_value(base_dataset, wave, location, component) - annual_mean_value(
        sensitivity_dataset,
        wave,
        location,
        component,
    )


def build_contribution_matrix(datasets: dict[str, xr.Dataset], label: str, components: list[str]) -> np.ndarray:
    wave = next(wave for rad_label, wave, _ in RADIATION_TYPES if rad_label == label)
    location = next(location for rad_label, _, location in RADIATION_TYPES if rad_label == label)

    matrix = np.zeros((len(components), len(CONTRIB_SCENARIOS)))
    for row, component in enumerate(components):
        for col, scenario in enumerate(CONTRIB_SCENARIOS):
            matrix[row, col] = annual_mean_contribution(
                datasets["base"],
                datasets[scenario],
                wave,
                location,
                component,
            )
    return matrix


def build_heatmap_matrix(datasets: dict[str, xr.Dataset], label: str, components: list[str]) -> np.ndarray:
    wave = next(wave for rad_label, wave, _ in RADIATION_TYPES if rad_label == label)
    location = next(location for rad_label, _, location in RADIATION_TYPES if rad_label == label)

    matrix = np.zeros((len(HEATMAP_SCENARIOS), len(components)))
    for row, scenario in enumerate(HEATMAP_SCENARIOS):
        for col, component in enumerate(components):
            if scenario == "total":
                matrix[row, col] = annual_mean_value(
                    datasets["noall"] if component in NATURAL_ONLY_COMPONENTS else datasets["base"],
                    wave,
                    location,
                    component,
                )
            elif scenario == "natural":
                matrix[row, col] = annual_mean_value(
                    datasets["noall"],
                    wave,
                    location,
                    component,
                )
            elif component in NATURAL_ONLY_COMPONENTS:
                matrix[row, col] = np.nan
            else:
                matrix[row, col] = annual_mean_contribution(
                    datasets["base"],
                    datasets[scenario],
                    wave,
                    location,
                    component,
                )
    return matrix


def symmetric_limit(arrays: list[np.ndarray], quantile: float = 98.0, floor: float = 0.01) -> float:
    finite_values: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        finite = np.abs(values[np.isfinite(values)])
        if finite.size:
            finite_values.append(finite)

    if not finite_values:
        return floor

    merged = np.concatenate(finite_values)
    return max(float(np.nanpercentile(merged, quantile)), floor)


def symmetric_colorbar_ticks(limit: float, step: float) -> np.ndarray:
    # Round outward to a tick boundary without adding a step for floating-point noise.
    intervals = max(1, int(np.ceil(np.nextafter(limit / step, -np.inf))))
    return np.arange(-intervals, intervals + 1, dtype=float) * step


def padded_limits(arrays: list[np.ndarray], floor: float = 0.01, pad_fraction: float = 0.12) -> tuple[float, float]:
    finite_values: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)

    if not finite_values:
        return -floor, floor

    merged = np.concatenate(finite_values)
    lower = min(float(np.nanmin(merged)), 0.0)
    upper = max(float(np.nanmax(merged)), 0.0)

    if np.isclose(lower, upper):
        spread = max(abs(lower), floor)
        return lower - spread, upper + spread

    padding = max((upper - lower) * pad_fraction, floor)
    return lower - padding, upper + padding


def format_cell_value(value: float, scale: float | None = None) -> str:
    numeric = float(value)
    if scale is not None:
        numeric = numeric / scale

    rounded = np.round(numeric, 2)
    if abs(rounded) < 0.005:
        rounded = 0.0
    return f"{rounded:.2f}"


def format_panel_title(label: str, matrix_label: str) -> str:
    return RAD_LABELS[matrix_label]


def heatmap_component_labels() -> list[str]:
    return [
        "NI" if component == "AM+NI" else COMPONENT_DISPLAY_LABELS[component]
        for component in HEATMAP_COMPONENTS
    ]


def heatmap_scenario_labels() -> list[str]:
    return [CONTRIB_SHORT_LABELS[name] for name in HEATMAP_SCENARIOS]


def save_figure(fig, output_dir, stem, dpi):
    save_ep_figure(fig, output_dir / "fig5_dre_component_source_heatmap", bbox_inches=None, pad_inches=0)


def remove_old_bar_outputs(output_dir):
    pass


def plot_fig2(datasets: dict[str, xr.Dataset], output_dir: Path, dpi: int) -> None:
    print("\n[Figure 5] Annual-mean contribution matrices")
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_old_bar_outputs(output_dir)

    heatmap_matrices = {
        label: build_heatmap_matrix(datasets, label, HEATMAP_COMPONENTS)
        for label, _, _ in RADIATION_TYPES
    }
    sw_labels = [label for label, _, _ in RADIATION_TYPES if label.startswith("SW")]
    lw_labels = [label for label, _, _ in RADIATION_TYPES if label.startswith("LW")]
    lw_cell_scale = 1e-2
    sw_vmax = symmetric_limit([heatmap_matrices[label] for label in sw_labels], floor=0.02)
    lw_vmax = symmetric_limit([heatmap_matrices[label] for label in lw_labels], floor=0.002)
    sw_ticks = symmetric_colorbar_ticks(sw_vmax, SW_COLORBAR_TICK_STEP)
    lw_ticks = symmetric_colorbar_ticks(lw_vmax, LW_COLORBAR_TICK_STEP)
    sw_vmax = float(sw_ticks[-1])
    lw_vmax = float(lw_ticks[-1])

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white")
    fig.subplots_adjust(left=0.118, right=0.91, top=0.94, bottom=0.055, wspace=0.50, hspace=0.2)
    heatmap_box_aspect = (len(HEATMAP_SCENARIOS) / len(HEATMAP_COMPONENTS)) * HEATMAP_WIDTH_STRETCH

    sw_heatmap_image = None
    lw_heatmap_image = None
    scenario_labels = heatmap_scenario_labels()
    component_labels = heatmap_component_labels()
    ordered_labels = ["SW-Surf", "SW-TOA", "LW-Surf", "LW-TOA"]
    for axis, (panel_label, matrix_label) in zip(axes.flat, zip("abcd", ordered_labels)):
        matrix = heatmap_matrices[matrix_label]
        is_lw_panel = matrix_label.startswith("LW")
        vmax = sw_vmax if matrix_label.startswith("SW") else lw_vmax
        axis.set_facecolor("#FBFAF7")
        for group in HEATMAP_GROUPS:
            axis.axhspan(
                float(group["start"]) - 0.5,
                float(group["end"]) + 0.5,
                color=str(group["facecolor"]),
                zorder=0,
            )
        heatmap_image = axis.imshow(
            matrix,
            cmap=HEATMAP_CMAP,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
            aspect="auto",
            interpolation="nearest",
            rasterized=True,
        )
        if matrix_label.startswith("SW"):
            sw_heatmap_image = heatmap_image
        else:
            lw_heatmap_image = heatmap_image
        axis.set_box_aspect(heatmap_box_aspect)

        axis.set_title(format_panel_title(panel_label, matrix_label), loc="left", pad=3.5)
        axis.text(
            -0.115,
            1.015,
            panel_label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZES["panel_label"],
            fontweight="bold",
            color="#1A1A1A",
            clip_on=False,
        )
        axis.set_xticks(np.arange(len(HEATMAP_COMPONENTS)))
        axis.set_xticklabels(component_labels)
        axis.set_yticks(np.arange(len(HEATMAP_SCENARIOS)))
        axis.set_yticklabels(scenario_labels)
        axis.tick_params(
            axis="x",
            pad=1.2,
            width=0.35,
            length=2.0,
            labelsize=FONT_SIZES["heatmap_xtick"],
            colors="#303030",
        )
        axis.tick_params(
            axis="y",
            pad=1.2,
            width=0.35,
            length=2.0,
            labelsize=FONT_SIZES["heatmap_ytick"],
            colors="#303030",
        )
        axis.set_xticks(np.arange(-0.5, len(HEATMAP_COMPONENTS), 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, len(HEATMAP_SCENARIOS), 1.0), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35)
        axis.tick_params(which="minor", bottom=False, left=False)
        for yline in (2.5, 5.5):
            axis.axhline(yline, color="#5E564D", linewidth=0.55, zorder=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.4)
        axis.spines["bottom"].set_linewidth(0.4)
        axis.spines["left"].set_color("#707070")
        axis.spines["bottom"].set_color("#707070")
        if is_lw_panel:
            axis.text(
                0.98,
                1.012,
                r"$\times 10^{-2}$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=FONT_SIZES["heatmap_scale_annotation"],
                color="#2F2F2F",
            )

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if abs(value) > vmax * 0.48 else "#1A1A1A"
                axis.text(
                    col,
                    row,
                    "-" if np.isnan(value) else format_cell_value(float(value), scale=lw_cell_scale if is_lw_panel else None),
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZES["heatmap_cell"],
                    fontweight="normal",
                    color=text_color,
                )

    assert sw_heatmap_image is not None and lw_heatmap_image is not None
    sw_colorbar = fig.colorbar(
        sw_heatmap_image,
        ax=axes[0, :].tolist(),
        location="right",
        fraction=0.032,
        pad=0.07,
        aspect=18,
        shrink=0.72,
    )
    sw_colorbar.set_label(r"$\Delta$SW (W m$^{-2}$)", fontsize=FONT_SIZES["colorbar_label"], labelpad=4.0)
    sw_colorbar.set_ticks(sw_ticks, labels=[f"{tick:.1f}" for tick in sw_ticks])
    sw_colorbar.ax.tick_params(labelsize=FONT_SIZES["sw_colorbar_tick"], width=0.35, length=2.0, pad=1.0)
    sw_colorbar.outline.set_linewidth(0.35)

    lw_colorbar = fig.colorbar(
        lw_heatmap_image,
        ax=axes[1, :].tolist(),
        location="right",
        fraction=0.032,
        pad=0.07,
        aspect=18,
        shrink=0.72,
    )
    lw_colorbar.set_label(r"$\Delta$LW (W m$^{-2}$)", fontsize=FONT_SIZES["colorbar_label"], labelpad=4.0)
    lw_colorbar.set_ticks(lw_ticks, labels=[f"{tick / lw_cell_scale:.0f}" for tick in lw_ticks])
    lw_colorbar.ax.set_title(r"$\times 10^{-2}$", fontsize=FONT_SIZES["lw_colorbar_scale_title"], pad=3.0)
    lw_colorbar.ax.tick_params(labelsize=FONT_SIZES["lw_colorbar_tick"], width=0.35, length=2.0, pad=1.0)
    lw_colorbar.outline.set_linewidth(0.35)

    save_figure(fig, output_dir, "fig2_heatmap", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    datasets = load_datasets()
    try:
        plot_fig2(datasets, args.output_dir, args.dpi)
    finally:
        close_datasets(datasets)


if __name__ == "__main__":
    main()
