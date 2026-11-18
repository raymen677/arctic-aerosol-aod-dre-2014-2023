"""Figure S12: Natural sea-salt and dust AOD.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from fig1_region import save_ep_figure, install_offline_geography
install_offline_geography()
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import xarray as xr

from fig2_arctic_aod import (
    DATASET_COLORS,
    DEFAULT_NATURAL,
    DEFAULT_OUTPUT_DIR,
    DISPLAY_SPECIES,
    FIGURE1_STEM_PREFIX,
    NATURAL_SPECIES,
    OVERVIEW_CMAPS,
    annual_mean_series,
    decadal_mean_map,
    draw_map_panel,
    draw_natural_variability_panel,
    format_title_scientific,
    get_species_field,
    nice_ceiling,
    scalar_weighted_mean,
    seasonal_climatology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 1 natural aerosol species panels.")
    parser.add_argument("--natural", type=Path, default=DEFAULT_NATURAL, help="Path to aod_gc_noall.nc")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures" / "supplementary", help="Directory used for exported figures")
    parser.add_argument("--dpi", type=int, default=1000, help="Export DPI for PNG/PDF output")
    parser.add_argument("--species", nargs="+", choices=NATURAL_SPECIES, default=NATURAL_SPECIES)
    return parser.parse_args()


def draw_natural_component_row(fig, map_spec, series_spec, natural: xr.Dataset, species: str, map_panel_letter: str, series_panel_letter: str) -> None:
    natural_field = get_species_field(natural, species)
    natural_map = decadal_mean_map(natural_field)
    vmax = nice_ceiling(float(natural_map.max().values))
    colorbar_ticks = [0.0, vmax / 2.0, vmax]
    seasonal_values = seasonal_climatology(natural_field)
    seasonal_max = nice_ceiling(float(seasonal_values.max().values) * 1.12)

    ax_map = fig.add_subplot(map_spec, projection=ccrs.NorthPolarStereo(central_longitude=20))
    ax_series = fig.add_subplot(series_spec)

    title = f"NAT {DISPLAY_SPECIES[species]} AOD ({format_title_scientific(scalar_weighted_mean(natural_field))})"
    draw_map_panel(
        ax_map,
        natural_map,
        Normalize(vmin=0.0, vmax=vmax),
        OVERVIEW_CMAPS["natural"],
        map_panel_letter,
        title,
        colorbar_ticks,
        [f"{tick:.3f}" for tick in colorbar_ticks],
        seasonal_values,
        seasonal_max,
        DATASET_COLORS["natural"],
    )

    ax_series.text(0.5, 1.045, f"({series_panel_letter}) NAT {DISPLAY_SPECIES[species]} Variability", transform=ax_series.transAxes, ha="center", va="bottom", fontsize=15.0, color="#111111")
    ax_series.set_xticks([])
    ax_series.set_yticks([])
    for spine in ax_series.spines.values():
        spine.set_visible(False)
    ax_series.set_facecolor("white")
    draw_natural_variability_panel(ax_series, {"NAT": annual_mean_series(natural_field)})




def create_combined_natural_component_figure(natural: xr.Dataset, output_dir: Path, dpi: int, species_list: list[str]) -> Path:
    fig = plt.figure(figsize=(10.1, 9.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.975, top=0.94, bottom=0.07, width_ratios=[1.0, 0.8], wspace=0.01, hspace=0.38)
    panel_letters = [("a", "b"), ("c", "d")]
    for row, species in enumerate(species_list):
        draw_natural_component_row(fig, gs[row, 0], gs[row, 1], natural, species, *panel_letters[row])

    combined_species = "+".join(species_list)
    png_path, _ = save_ep_figure(fig, output_dir / "figs12_natural_sea_salt_dust_aod")
    plt.close(fig)
    return png_path


def print_summary(natural: xr.Dataset, species_list: list[str]) -> None:
    print(r"Arctic (>50 deg N) area-weighted ten-year mean natural AOD")
    for species in species_list:
        natural_mean = scalar_weighted_mean(get_species_field(natural, species))
        print(f"  {DISPLAY_SPECIES[species]:>5}: natural={natural_mean:.5f}")


def main() -> None:
    args = parse_args()
    with xr.open_dataset(args.natural) as natural:
        print_summary(natural, ["SS", "DU"])
        create_combined_natural_component_figure(natural, args.output_dir, args.dpi, ["SS", "DU"])


if __name__ == "__main__":
    main()
