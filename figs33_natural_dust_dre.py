"""Figure S33: Natural dust surface and TOA shortwave and longwave DRE.

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

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from fig4_surface_shortwave_dre import (
    ARCTIC_LAT_MIN,
    DATASET_FILES,
    DEFAULT_OUTPUT_DIR,
    FONT_SIZES,
    NATURAL_SOURCE_SPEC,
    PANEL_SPECS,
    Figure2Plotter,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


NATURAL_COMPONENTS = ["DU", "SS"]
COMPONENT_DISPLAY_LABELS = {
    "DU": "DU",
    "SS": "SS",
}


class NaturalDuSsFigure1Plotter(Figure2Plotter):
    """Generate simplified Figure 1 panels for natural DU and SS."""

    def load_datasets(self) -> None:
        path = DATASET_FILES["noall"]
        if not path.exists():
            raise FileNotFoundError(f"Missing input dataset: {path}")
        print(f"Loading natural aerosol dataset: {path.name}")
        self.datasets["noall"] = xr.open_dataset(path)

    def prepare_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self.load_datasets()
        try:
            self.prepare_output_directory()
            print("\n[DU/SS Figure 1] Natural spatial maps and NAT time series")
            for component in NATURAL_COMPONENTS:
                print(f"  Plotting {component}")
                self.plot_component_figure(component)
        finally:
            self.close_datasets()

    def build_natural_panel_data(self, component: str, panel_spec: dict[str, object]) -> dict[str, object]:
        natural_field = self.get_component_field(self.datasets["noall"], panel_spec, component)
        natural_map = natural_field.mean("time").load()
        natural_series = self.annual_mean_series(natural_field)
        seasonal_values = self.seasonal_climatology(natural_field)

        map_values = natural_map.where(natural_map["lat"] > ARCTIC_LAT_MIN, drop=True).values
        finite_abs = np.abs(np.asarray(map_values, dtype=float)[np.isfinite(map_values)])
        map_vmax = self.nice_ceiling(
            max(float(np.nanpercentile(finite_abs, 98.0)), float(panel_spec["floor"]))
            if finite_abs.size
            else float(panel_spec["floor"])
        )
        map_power = self.display_power([np.array([-map_vmax, map_vmax], dtype=float)])
        map_mean = float(self.area_weighted_mean(natural_map).values)
        line_power = self.display_power([natural_series.values])
        series_ylim = self.compute_axis_limits(
            [natural_series.values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.16,
            enforce_floor=False,
        )
        seasonal_ylim = self.compute_axis_limits(
            [seasonal_values.values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.23,
            rounded_lower=True,
        )

        return {
            "map": natural_map,
            "series": natural_series,
            "seasonal": seasonal_values,
            "map_vmax": map_vmax,
            "map_power": map_power,
            "map_mean": map_mean,
            "line_power": line_power,
            "series_ylim": series_ylim,
            "seasonal_ylim": seasonal_ylim,
        }

    def plot_component_figure(self, component: str) -> None:
        component_label = COMPONENT_DISPLAY_LABELS[component]
        panel_data = [self.build_natural_panel_data(component, panel_spec) for panel_spec in PANEL_SPECS]

        fig = plt.figure(figsize=(11.2, 15.6), constrained_layout=False)
        grid = fig.add_gridspec(
            len(PANEL_SPECS),
            2,
            left=0.060,
            right=0.970,
            top=0.965,
            bottom=0.045,
            width_ratios=[1.08, 1.0],
            hspace=0.46,
            wspace=0.24,
        )

        projection = ccrs.NorthPolarStereo(central_longitude=20)
        panel_tags = iter("abcdefgh")
        color = str(NATURAL_SOURCE_SPEC["color"])

        for row, (panel_spec, data) in enumerate(zip(PANEL_SPECS, panel_data)):
            ylabel = str(panel_spec["ylabel"])
            map_axis = fig.add_subplot(grid[row, 0], projection=projection)
            map_axis.set_box_aspect(1.0)
            map_unit = " W/m²"
            map_title = (
                f"{component_label} {ylabel} NAT Spatial "
                f"({self.format_scientific_value(float(data['map_mean']))}{map_unit})"
            )
            mesh = self.plot_map(
                map_axis,
                data["map"],
                float(data["map_vmax"]),
                next(panel_tags),
                map_title,
            )
            colorbar_ticks = [-float(data["map_vmax"]), 0.0, float(data["map_vmax"])]
            colorbar_labels = [self.format_scaled_value(tick, int(data["map_power"])) for tick in colorbar_ticks]
            self.add_inline_colorbar(
                map_axis,
                mesh,
                colorbar_ticks,
                colorbar_labels,
                int(data["map_power"]),
            )
            self.add_seasonal_inset(
                map_axis,
                data["seasonal"],
                data["seasonal_ylim"],
                color,
                int(data["map_power"]),
                annotate_scale=True,
            )

            line_axis = fig.add_subplot(grid[row, 1])
            self.plot_natural_timeseries(
                line_axis,
                data["series"],
                data["series_ylim"],
                int(data["line_power"]),
            )
            self.add_panel_title(line_axis, next(panel_tags), f"{component_label} {ylabel} NAT Timeseries")

        self.save_figure(fig, self.output_dir / component, component)
        plt.close(fig)

    def plot_natural_timeseries(
        self,
        ax: plt.Axes,
        series: xr.DataArray,
        ylim: tuple[float, float],
        scale_power: int,
    ) -> None:
        color = str(NATURAL_SOURCE_SPEC["color"])
        ax.plot(
            series["year"].values,
            series.values,
            color=color,
            linewidth=float(NATURAL_SOURCE_SPEC["linewidth"]),
            linestyle=str(NATURAL_SOURCE_SPEC["linestyle"]),
            marker=str(NATURAL_SOURCE_SPEC["marker"]),
            markersize=float(NATURAL_SOURCE_SPEC["markersize"]) + 0.4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="NAT",
            zorder=5,
        )
        ax.axhline(
            float(series.mean().values),
            color=color,
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.8,
            alpha=0.36,
            zorder=1,
        )

        self.style_timeseries_axis(
            ax,
            np.asarray(series["year"].values),
            ylim,
            right_ticks=True,
            scale_power=scale_power,
            annotate_scale=True,
        )
        ax.legend(
            frameon=False,
            fontsize=FONT_SIZES["source_line_legend"] + 0.6,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            handlelength=1.9,
            borderaxespad=0.0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot natural DU and SS Figure 1 panels.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "supplementary",
        help="Directory used for exported figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1000,
        help="Export DPI for PNG files and embedded rasters.",
    )
    return parser.parse_args()


def main() -> None:
    render_natural_dre("DU", "figs33_natural_dust_dre")




def render_natural_dre(component, stem):
    directory = Path(__file__).resolve().parent / "figures" / "supplementary"
    plotter = NaturalDuSsFigure1Plotter(directory, 1000)
    plotter.output_stem = stem
    plotter.load_datasets()
    try:
        plotter.plot_component_figure(component)
    finally:
        plotter.close_datasets()

if __name__ == "__main__":
    main()
