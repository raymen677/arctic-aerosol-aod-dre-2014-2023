"""Figure 4: Surface shortwave aerosol direct radiative effects.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: codes/main/fig1_region.py.
"""
#!/usr/bin/env python3

from __future__ import annotations

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
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "radiation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "main"
ARCTIC_LAT_MIN = 50.0
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SCIENTIFIC_DECIMALS = 2


DATASET_FILES = {
    "base": BASE_DIR / "rad_gc_L0_base.nc",
    "noall": BASE_DIR / "rad_gc_L1_noall.nc",
    "local_nonp": BASE_DIR / "rad_gc_L2_local_nonp.nc",
    "local_no_north_europe": BASE_DIR / "rad_gc_L3_local_no_north_europe.nc",
    "local_no_north_europe_sup": BASE_DIR / "rad_gc_L3_local_no_north_europe_sup.nc",
    "trans_nonp_sup": BASE_DIR / "rad_gc_L2_trans_nonp_sup.nc",
    "trans_no_south_europe": BASE_DIR / "rad_gc_L3_trans_no_south_europe.nc",
    "trans_no_east_asia": BASE_DIR / "rad_gc_L3_trans_no_east_asia.nc",
    "trans_no_north_america": BASE_DIR / "rad_gc_L3_trans_no_north_america.nc",
    "trans_no_three_regions_sup": BASE_DIR / "rad_gc_L3_trans_no_three_regions_sup.nc",
}

PLOTTED_COMPONENTS = ["PM", "BC", "OA", "SU", "AM+NI"]
COMPONENT_DISPLAY_LABELS = {
    "PM": "Total",
    "BC": "BC",
    "OA": "OC",
    "SU": "SU",
    "AM+NI": "NI",
}

GROUP_ORDER = ["NAT", "ANT", "Local", "Non-Local"]
MAJOR_SOURCE_LABELS = ["ANT", "NAT", "Local", "Non-Local"]
ANT_NAT_SOURCE_LABELS = ["ANT", "NAT"]
LOCAL_TRANS_MAJOR_LABELS = ["Local", "Non-Local"]
LOCAL_SOURCE_LABELS = ["N. Europe", "Other Local"]
LOCAL_SOURCE_DISPLAY_LABELS = ["N. Europe", "Other"]
TRANS_SOURCE_LABELS = ["S. Europe", "E. Asia", "N. America", "Other Non-Local"]
TRANS_SOURCE_DISPLAY_LABELS = ["S. Europe", "E. Asia", "N. America", "Other"]

MAP_COLUMN_SPECS = [
    {"key": "total", "title": "Total", "color": "#9A6B38"},
    {"key": "anthropogenic", "title": "ANT", "color": "#C46A4D"},
    {"key": "natural", "title": "NAT", "color": "#6E8B63"},
]

FONT_SIZES = {
    "base": 15.0,
    "map_title": 15.0,
    "column_title": 22.0,
    "timeseries_xticks": 15.6,
    "timeseries_yticks": 15.6,
    "bar_xticks": 15.0,
    "bar_yticks": 15.0,
    "y_label": 18.0,
    "group_label": 15.0,
    "bar_value": 15.0,
    "source_line_legend": 14.4,
    "inline_colorbar_ticks": 12.6,
    "seasonal_xticks": 9.8,
    "seasonal_yticks": 15.0,
    "scientific_multiplier": 15.6,
}

BAR_VALUE_LAYOUT = {
    "base_offset_fraction": 0.01,
    "close_threshold_fraction": 0.01,
    "stagger_step_fraction": 0.01,
    "edge_padding_fraction": 0.01,
}

SOURCE_SPECS = [
    {
        "key": "noall",
        "label": "ANT",
        "tick_label": "ANT",
        "group": "ANT",
        "color": "#B24B3A",
        "marker": "D",
        "linewidth": 2.4,
        "markersize": 4.0,
        "linestyle": "-",
        "zorder": 6,
    },
    {
        "key": "local_nonp",
        "label": "Local",
        "tick_label": "Local",
        "group": "Local",
        "color": "#5E84A7",
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 3.6,
        "linestyle": "-",
        "zorder": 5,
    },
    {
        "key": "local_no_north_europe",
        "label": "N. Europe",
        "tick_label": "N. Europe",
        "group": "Local",
        "color": "#3F678D",
        "marker": "o",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "-",
        "zorder": 4,
    },
    {
        "key": "local_no_north_europe_sup",
        "label": "Other Local",
        "tick_label": "Other",
        "group": "Local",
        "color": "#9DBDD9",
        "marker": "o",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "--",
        "zorder": 3,
    },
    {
        "key": "trans_nonp_sup",
        "label": "Non-Local",
        "tick_label": "Non-Local",
        "group": "Non-Local",
        "color": "#C48A4E",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 3.6,
        "linestyle": "-",
        "zorder": 5,
    },
    {
        "key": "trans_no_south_europe",
        "label": "S. Europe",
        "tick_label": "S. Europe",
        "group": "Non-Local",
        "color": "#7FA46B",
        "marker": "s",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "-",
        "zorder": 4,
    },
    {
        "key": "trans_no_east_asia",
        "label": "E. Asia",
        "tick_label": "E. Asia",
        "group": "Non-Local",
        "color": "#4C9A95",
        "marker": "s",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "-",
        "zorder": 4,
    },
    {
        "key": "trans_no_north_america",
        "label": "N. America",
        "tick_label": "N. America",
        "group": "Non-Local",
        "color": "#D2AA58",
        "marker": "s",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "-",
        "zorder": 4,
    },
    {
        "key": "trans_no_three_regions_sup",
        "label": "Other Non-Local",
        "tick_label": "Other",
        "group": "Non-Local",
        "color": "#8E8375",
        "marker": "s",
        "linewidth": 1.45,
        "markersize": 3.1,
        "linestyle": "--",
        "zorder": 3,
    },
]

NATURAL_SOURCE_SPEC = {
    "key": "natural",
    "label": "NAT",
    "tick_label": "NAT",
    "group": "NAT",
    "color": "#6E8B63",
    "marker": "^",
    "linewidth": 2.2,
    "markersize": 3.8,
    "linestyle": "-",
    "zorder": 5,
}
LINE_SOURCE_SPECS = [SOURCE_SPECS[0], NATURAL_SOURCE_SPEC, *SOURCE_SPECS[1:]]

PANEL_SPECS = [
    {
        "panel_name": "SW-Surf",
        "wave": "SW",
        "location": "Surf",
        "ylabel": "SW Surf",
        "floor": 0.02,
    },
    {
        "panel_name": "SW-TOA",
        "wave": "SW",
        "location": "TOA",
        "ylabel": "SW TOA",
        "floor": 0.02,
    },
    {
        "panel_name": "LW-Surf",
        "wave": "LW",
        "location": "Surf",
        "ylabel": "LW Surf",
        "floor": 0.002,
    },
    {
        "panel_name": "LW-TOA",
        "wave": "LW",
        "location": "TOA",
        "ylabel": "LW TOA",
        "floor": 0.002,
    },
]

RE_CMAP = LinearSegmentedColormap.from_list(
    "arctic_re_diverging",
    ["#2F5D82", "#D9E3EC", "#FBF8F3", "#E9C1B4", "#A64A3A"],
)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FONT_SIZES["base"],
        "text.usetex": False,
        "figure.dpi": 100,
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


class Figure2Plotter:
    """Generate all Figure 1 outputs."""

    def __init__(self, output_dir: Path, dpi: int) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        self.output_stem = "fig4_surface_shortwave_dre"
        self.datasets: dict[str, xr.Dataset] = {}

    def load_datasets(self) -> None:
        for name, path in DATASET_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing input dataset: {path}")
            print(f"Loading {name}: {path.name}")
            self.datasets[name] = xr.open_dataset(path)

    def close_datasets(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()

    def run(self) -> None:
        self.load_datasets()
        try:
            self.plot_component_panel_figure("PM", PANEL_SPECS[0])
        finally:
            self.close_datasets()

    def prepare_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_component_field(
        self,
        dataset: xr.Dataset,
        panel_spec: dict[str, object],
        component: str,
    ) -> xr.DataArray:
        wave = str(panel_spec["wave"])
        location = str(panel_spec["location"])
        if component == "AM+NI":
            return dataset[f"RadAllSky{wave}{location}_AM"] + dataset[f"RadAllSky{wave}{location}_NI"]
        return dataset[f"RadAllSky{wave}{location}_{component}"]

    def area_weighted_mean(self, field: xr.DataArray) -> xr.DataArray:
        subset = field.where(field["lat"] > ARCTIC_LAT_MIN, drop=True)
        weights = np.cos(np.deg2rad(subset["lat"]))
        return subset.weighted(weights).mean(dim=("lat", "lon"))

    def annual_mean_series(self, field: xr.DataArray) -> xr.DataArray:
        return self.area_weighted_mean(field).groupby("time.year").mean().load()

    def seasonal_climatology(self, field: xr.DataArray) -> xr.DataArray:
        return self.area_weighted_mean(field).groupby("time.season").mean().reindex(season=SEASON_ORDER).load()

    def annual_contribution_series(
        self,
        source_key: str,
        panel_spec: dict[str, object],
        component: str,
    ) -> xr.DataArray:
        base_field = self.get_component_field(self.datasets["base"], panel_spec, component)
        sensitivity_field = self.get_component_field(self.datasets[source_key], panel_spec, component)
        return self.annual_mean_series(base_field - sensitivity_field)

    def build_panel_data(self, component: str, panel_spec: dict[str, object]) -> dict[str, object]:
        series_by_source: dict[str, xr.DataArray] = {}
        mean_values: dict[str, float] = {}

        for source_spec in SOURCE_SPECS:
            source_key = str(source_spec["key"])
            series = self.annual_contribution_series(source_key, panel_spec, component)
            series_by_source[source_key] = series
            mean_values[source_key] = float(series.mean().values)

        natural_field = self.get_component_field(self.datasets["noall"], panel_spec, component)
        natural_series = self.annual_mean_series(natural_field)
        series_by_source[str(NATURAL_SOURCE_SPEC["key"])] = natural_series
        mean_values[str(NATURAL_SOURCE_SPEC["key"])] = float(natural_series.mean().values)

        bar_values = np.array(
            [
                mean_values[str(NATURAL_SOURCE_SPEC["key"])],
                *[mean_values[str(source_spec["key"])] for source_spec in SOURCE_SPECS],
            ],
            dtype=float,
        )

        return {
            "series_by_source": series_by_source,
            "mean_values": mean_values,
            "years": next(iter(series_by_source.values()))["year"].values,
            "bar_ylim": self.compute_axis_limits([bar_values], floor=float(panel_spec["floor"]), pad_fraction=0.22),
        }

    def build_map_data(
        self,
        component: str,
        panel_spec: dict[str, object],
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        total_field = self.get_component_field(self.datasets["base"], panel_spec, component)
        natural_field = self.get_component_field(self.datasets["noall"], panel_spec, component)
        anthropogenic_field = total_field - natural_field

        maps = {
            "total": total_field.mean("time").load(),
            "anthropogenic": anthropogenic_field.mean("time").load(),
            "natural": natural_field.mean("time").load(),
        }
        seasonal = {
            "total": self.seasonal_climatology(total_field),
            "anthropogenic": self.seasonal_climatology(anthropogenic_field),
            "natural": self.seasonal_climatology(natural_field),
        }
        return maps, seasonal

    def compute_axis_limits(
        self,
        arrays: list[np.ndarray],
        floor: float,
        pad_fraction: float = 0.18,
        enforce_floor: bool = True,
        rounded_lower: bool = False,
    ) -> tuple[float, float]:
        finite_values: list[np.ndarray] = []
        for array in arrays:
            values = np.asarray(array, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)

        if not finite_values:
            return (-self.nice_ceiling(floor), self.nice_ceiling(floor)) if rounded_lower else (-1.0, 1.0)

        merged = np.concatenate(finite_values)
        lower = float(np.nanmin(merged))
        upper = float(np.nanmax(merged))
        tolerance_scale = floor if enforce_floor else max(abs(lower), abs(upper), np.finfo(float).eps)
        tolerance = tolerance_scale * 1e-6

        def upper_limit(value: float) -> float:
            if enforce_floor:
                value = max(value, floor)
            elif np.isclose(value, 0.0):
                value = floor
            return self.nice_ceiling(value)

        def lower_limit(value: float) -> float:
            padded = value * (1.0 + pad_fraction)
            if rounded_lower:
                return self.nice_floor(padded)
            return -upper_limit(abs(value) * (1.0 + pad_fraction))

        if upper <= tolerance:
            return lower_limit(lower), 0.0
        if lower >= -tolerance:
            return 0.0, upper_limit(upper * (1.0 + pad_fraction))
        return lower_limit(lower), upper_limit(upper * (1.0 + pad_fraction))

    def plot_component_panel_figure(self, component: str, panel_spec: dict[str, object]) -> None:
        panel_data = self.build_panel_data(component, panel_spec)
        maps, seasonal = self.build_map_data(component, panel_spec)
        component_label = COMPONENT_DISPLAY_LABELS[component]
        map_keys = [str(spec["key"]) for spec in MAP_COLUMN_SPECS]
        map_arrays = [maps[key].where(maps[key]["lat"] > ARCTIC_LAT_MIN, drop=True).values for key in map_keys]
        map_finite = np.concatenate([
            np.abs(np.asarray(array, dtype=float)[np.isfinite(array)])
            for array in map_arrays
            if np.isfinite(array).any()
        ])
        map_vmax = self.nice_ceiling(max(float(np.nanpercentile(map_finite, 98.0)), float(panel_spec["floor"])))
        map_power = self.display_power([np.array([-map_vmax, map_vmax], dtype=float)])
        map_means = {key: float(self.area_weighted_mean(maps[key]).values) for key in map_keys}
        line_power = self.display_power(
            [
                series.values
                for series in dict(panel_data["series_by_source"]).values()
            ]
        )
        bar_power = self.display_power([np.array(list(dict(panel_data["mean_values"]).values()), dtype=float)])
        seasonal_ylim = self.compute_axis_limits(
            [seasonal[key].values for key in map_keys],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.23,
            rounded_lower=True,
        )

        fig = plt.figure(figsize=(14.0, 10.8), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            3,
            left=0.055,
            right=0.985,
            top=0.93,
            bottom=0.075,
            height_ratios=[1.0, 0.98],
            hspace=0.34,
            wspace=0.30,
        )

        panel_tags = iter("abcde")
        projection = ccrs.NorthPolarStereo(central_longitude=20)
        colorbar_ticks = [-map_vmax, 0.0, map_vmax]
        colorbar_labels = [self.format_scaled_value(tick, map_power) for tick in colorbar_ticks]

        for col, column_spec in enumerate(MAP_COLUMN_SPECS):
            map_key = str(column_spec["key"])
            map_axis = fig.add_subplot(grid[0, col], projection=projection)
            map_axis.set_box_aspect(1.0)
            map_title = str(column_spec["title"])
            # For PM ("Total"), don't append the component label to ANT/NAT map titles
            if component != "PM" and map_title != component_label:
                map_title = f"{map_title} {component_label}"
            map_unit = " W/m²"
            map_title = f"{map_title} ({self.format_scientific_value(map_means[map_key])}{map_unit})"
            mesh = self.plot_map(
                map_axis,
                maps[map_key],
                map_vmax,
                next(panel_tags),
                map_title,
            )
            self.add_inline_colorbar(map_axis, mesh, colorbar_ticks, colorbar_labels, map_power)
            self.add_seasonal_inset(
                map_axis,
                seasonal[map_key],
                seasonal_ylim,
                str(column_spec["color"]),
                map_power,
                annotate_scale=True,
            )

        bottom_grid = grid[1, :].subgridspec(1, 2, wspace=0.17)
        line_axis = fig.add_subplot(bottom_grid[0, 0])
        bar_axis = fig.add_subplot(bottom_grid[0, 1])

        line_label = "" if component == "PM" else f"{component_label} "
        self.add_panel_title(
            line_axis,
            next(panel_tags),
            f"{line_label}{panel_spec['ylabel']} Source Temporal Variations (W/m²)",
        )
        self.plot_timeseries_family_panels(line_axis, panel_data, panel_spec, line_power)

        bar_label = "" if component == "PM" else f"{component_label} "
        self.add_panel_title(
            bar_axis,
            next(panel_tags),
            f"{bar_label}{panel_spec['ylabel']} Source Mean Changes (W/m²)",
        )
        self.plot_bar_panel(
            bar_axis,
            dict(panel_data["mean_values"]),
            tuple(panel_data["bar_ylim"]),
            bar_power,
        )

        component_dir = self.output_dir / component
        self.save_figure(fig, component_dir, str(panel_spec["ylabel"]))
        plt.close(fig)

    def plot_map(
        self,
        ax: plt.Axes,
        field_2d: xr.DataArray,
        vmax: float,
        panel_letter: str,
        title: str,
    ):
        self.setup_arctic_axis(ax)
        lon, lat, data = self.cyclic_lon_lat_data(field_2d)
        mesh = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=RE_CMAP,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
            shading="auto",
            rasterized=True,
            alpha=0.97,
            zorder=1.5,
        )
        self.add_panel_title(ax, panel_letter, title)
        return mesh

    def setup_arctic_axis(self, ax: plt.Axes) -> None:
        ax.set_extent([-180, 180, ARCTIC_LAT_MIN, 90], crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor="#F6F1E8",
            edgecolor="none",
            zorder=0,
        )
        ax.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor="#FCFBF8",
            edgecolor="none",
            zorder=0,
        )
        gridlines = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            xlocs=np.arange(-180, 181, 60),
            ylocs=[60, 70, 80],
            linewidth=0.45,
            color="#BBB3A8",
            alpha=0.22,
            linestyle="--",
            zorder=1,
        )
        gridlines.n_steps = 100
        lons = np.linspace(-180, 180, 721)
        ax.plot(
            lons,
            np.full_like(lons, ARCTIC_LAT_MIN),
            transform=ccrs.PlateCarree(),
            color="#D3CBC0",
            linewidth=0.8,
            zorder=2,
        )
        ax.coastlines(
            resolution="50m",
            linewidth=0.55,
            color="#8A847B",
            zorder=3,
        )
        self.add_circular_boundary(ax)
        geo_spine = ax.spines.get("geo")
        if geo_spine is not None:
            geo_spine.set_linewidth(0.65)
            geo_spine.set_edgecolor("#A79E93")

    def add_panel_title(self, ax: plt.Axes, panel_letter: str, title: str) -> None:
        ax.text(
            0.5,
            1.115,
            f"({panel_letter}) {title}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["map_title"],
            color="#111111",
            clip_on=False,
        )

    def add_inline_colorbar(
        self,
        ax: plt.Axes,
        mappable,
        ticks: list[float],
        ticklabels: list[str],
        scale_power: int,
    ) -> None:
        colorbar_axis = ax.inset_axes([0.18, -0.07, 0.64, 0.035])
        colorbar_axis.set_facecolor((1, 1, 1, 0.78))
        colorbar = plt.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(ticklabels)
        colorbar.outline.set_linewidth(0.55)
        colorbar.outline.set_edgecolor("#796C5C")
        colorbar.ax.tick_params(
            labelsize=FONT_SIZES["inline_colorbar_ticks"],
            width=0.45,
            length=1.8,
            pad=1.2,
            colors="#3F3A35",
        )
        self.add_scale_label(colorbar_axis, scale_power, y=2.20, fontsize=FONT_SIZES["inline_colorbar_ticks"], color="#3F3A35")

    def add_scale_label(
        self,
        ax: plt.Axes,
        scale_power: int,
        *,
        y: float = 0.98,
        va: str = "top",
        fontsize: float | None = None,
        color: str = "#3A3530",
    ) -> None:
        multiplier = self.format_multiplier(scale_power)
        if not multiplier:
            return
        ax.text(
            0.02,
            y,
            multiplier,
            transform=ax.transAxes,
            ha="left",
            va=va,
            fontsize=fontsize if fontsize is not None else FONT_SIZES["scientific_multiplier"],
            color=color,
            zorder=10,
            clip_on=False,
        )

    def add_seasonal_inset(
        self,
        ax: plt.Axes,
        seasonal_values: xr.DataArray,
        ylim: tuple[float, float],
        color: str,
        scale_power: int,
        annotate_scale: bool = True,
    ) -> None:
        inset = ax.inset_axes([0.02, 0.17, 0.45, 0.25])
        inset.set_zorder(4.5)
        self.plot_seasonal_bars(inset, seasonal_values, ylim, color, scale_power, annotate_scale)

    def plot_seasonal_bars(
        self,
        ax: plt.Axes,
        seasonal_values: xr.DataArray,
        ylim: tuple[float, float],
        color: str,
        scale_power: int,
        annotate_scale: bool = True,
    ) -> None:
        values = np.array([float(seasonal_values.sel(season=season).values) for season in SEASON_ORDER], dtype=float)
        ax.bar(
            np.arange(len(SEASON_ORDER)),
            values,
            width=0.74,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.84,
            zorder=3,
        )
        ax.axhline(0.0, color=(0.18, 0.18, 0.18, 0.72), linewidth=0.45, zorder=2)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(len(SEASON_ORDER)))
        ax.set_xticklabels(SEASON_ORDER)
        yticks = self.dedup_ticks([ylim[0], 0.0, ylim[1]])
        ax.set_yticks(yticks)
        ax.set_yticklabels([self.format_scaled_value(tick, scale_power) for tick in yticks])
        if annotate_scale:
            has_positive_values = np.nanmax(values) > 0.0
            self.add_scale_label(
                ax,
                scale_power,
                y=0.98 if has_positive_values else 0.02,
                va="top" if has_positive_values else "bottom",
                fontsize=FONT_SIZES["seasonal_yticks"],
            )
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=(1, 1, 1, 0.72), linestyle="-", linewidth=0.4)
        ax.tick_params(axis="x", labelsize=FONT_SIZES["seasonal_xticks"], length=0.0, pad=0.8)
        ax.tick_params(axis="y", labelsize=FONT_SIZES["seasonal_yticks"], length=1.2, pad=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.45)
        ax.spines["bottom"].set_linewidth(0.45)
        ax.spines["left"].set_color((0.15, 0.15, 0.15, 0.75))
        ax.spines["bottom"].set_color((0.15, 0.15, 0.15, 0.75))
        ax.set_facecolor((1, 1, 1, 0.68))

    def plot_source_lines(
        self,
        ax: plt.Axes,
        series_by_source: dict[str, xr.DataArray],
        labels: list[str],
        alpha: float,
        draw_mean: bool,
        emphasize: bool,
    ) -> dict[str, object]:
        selected_labels = set(labels)
        handles: dict[str, object] = {}

        for source_spec in LINE_SOURCE_SPECS:
            label = str(source_spec["label"])
            if label not in selected_labels:
                continue

            source_key = str(source_spec["key"])
            series = series_by_source[source_key]
            linewidth = float(source_spec["linewidth"])

            line, = ax.plot(
                series["year"].values,
                series.values,
                color=source_spec["color"],
                linewidth=linewidth,
                linestyle=str(source_spec["linestyle"]),
                marker=str(source_spec["marker"]),
                markersize=float(source_spec["markersize"]),
                markerfacecolor=source_spec["color"],
                markeredgecolor="white",
                markeredgewidth=0.55 if emphasize else 0.4,
                alpha=alpha,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=int(source_spec["zorder"]),
            )
            handles[label] = line

            if emphasize:
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=linewidth + 1.25, foreground=(1, 1, 1, 0.95)),
                        pe.Normal(),
                    ]
                )
            if draw_mean:
                ax.axhline(
                    float(series.mean().values),
                    color=source_spec["color"],
                    linestyle=(0, (2.0, 2.0)),
                    linewidth=0.7,
                    alpha=0.28,
                    zorder=1,
                )

        return handles

    def plot_timeseries_family_panels(
        self,
        container_axis: plt.Axes,
        panel_data: dict[str, object],
        panel_spec: dict[str, object],
        scale_power: int,
    ) -> None:
        series_by_source = dict(panel_data["series_by_source"])
        years = np.asarray(panel_data["years"])

        container_axis.set_xticks([])
        container_axis.set_yticks([])
        for spine in container_axis.spines.values():
            spine.set_visible(False)
        container_axis.set_facecolor("white")
        self.add_scale_label(container_axis, scale_power, y=1.08)

        ant_nat_axis = container_axis.inset_axes([0.01, 0.54, 0.44, 0.45])
        local_trans_axis = container_axis.inset_axes([0.50, 0.54, 0.44, 0.45])
        local_axis = container_axis.inset_axes([0.01, 0.02, 0.44, 0.45])
        trans_axis = container_axis.inset_axes([0.50, 0.02, 0.44, 0.45])

        ant_nat_values = np.concatenate(
            [
                series_by_source[str(source_spec["key"])].values
                for source_spec in LINE_SOURCE_SPECS
                if str(source_spec["label"]) in ANT_NAT_SOURCE_LABELS
            ]
        )
        local_trans_values = np.concatenate(
            [
                series_by_source[str(source_spec["key"])].values
                for source_spec in LINE_SOURCE_SPECS
                if str(source_spec["label"]) in LOCAL_TRANS_MAJOR_LABELS
            ]
        )
        local_detail_values = np.concatenate(
            [
                series_by_source[str(source_spec["key"])].values
                for source_spec in SOURCE_SPECS
                if str(source_spec["label"]) in LOCAL_SOURCE_LABELS
            ]
        )
        trans_detail_values = np.concatenate(
            [
                series_by_source[str(source_spec["key"])].values
                for source_spec in SOURCE_SPECS
                if str(source_spec["label"]) in TRANS_SOURCE_LABELS
            ]
        )
        ant_nat_ylim = self.compute_axis_limits(
            [ant_nat_values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.16,
            enforce_floor=False,
        )
        local_trans_ylim = self.compute_axis_limits(
            [local_trans_values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.16,
            enforce_floor=False,
        )
        local_ylim = self.compute_axis_limits(
            [local_detail_values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.18,
            enforce_floor=False,
        )
        trans_ylim = self.compute_axis_limits(
            [trans_detail_values],
            floor=float(panel_spec["floor"]),
            pad_fraction=0.18,
            enforce_floor=False,
        )
        ant_nat_handles = self.plot_source_lines(
            ant_nat_axis,
            series_by_source,
            labels=ANT_NAT_SOURCE_LABELS,
            alpha=0.98,
            draw_mean=True,
            emphasize=True,
        )
        self.style_timeseries_axis(
            ant_nat_axis,
            years,
            ant_nat_ylim,
            show_xticklabels=False,
            scale_power=scale_power,
            annotate_scale=False,
        )
        self.add_axis_corner_title(ant_nat_axis, None)
        ant_nat_axis.legend(
            [ant_nat_handles[label] for label in ANT_NAT_SOURCE_LABELS],
            ANT_NAT_SOURCE_LABELS,
            frameon=False,
            fontsize=FONT_SIZES["source_line_legend"],
            loc="upper center",
            bbox_to_anchor=(0.56, 1.18),
            ncol=2,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

        local_trans_axis.axhspan(
            0,
            1,
            transform=local_trans_axis.get_yaxis_transform(),
            color="#FBF4EA",
            zorder=0,
        )
        local_trans_handles = self.plot_source_lines(
            local_trans_axis,
            series_by_source,
            labels=LOCAL_TRANS_MAJOR_LABELS,
            alpha=0.98,
            draw_mean=True,
            emphasize=True,
        )
        self.style_timeseries_axis(
            local_trans_axis,
            years,
            local_trans_ylim,
            right_ticks=True,
            show_xticklabels=False,
            scale_power=scale_power,
            annotate_scale=False,
        )
        self.add_axis_corner_title(local_trans_axis, None)
        local_trans_axis.legend(
            [local_trans_handles[label] for label in LOCAL_TRANS_MAJOR_LABELS],
            LOCAL_TRANS_MAJOR_LABELS,
            frameon=False,
            fontsize=FONT_SIZES["source_line_legend"],
            loc="upper center",
            bbox_to_anchor=(0.56, 1.18),
            ncol=2,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

        local_axis.axhspan(0, 1, transform=local_axis.get_yaxis_transform(), color="#F1F6FB", zorder=0)
        local_handles = self.plot_source_lines(
            local_axis,
            series_by_source,
            labels=LOCAL_SOURCE_LABELS,
            alpha=0.92,
            draw_mean=False,
            emphasize=False,
        )
        self.style_timeseries_axis(
            local_axis,
            years,
            local_ylim,
            facecolor="#F8FBFE",
            scale_power=scale_power,
            annotate_scale=False,
        )
        self.add_axis_corner_title(local_axis, "Local")
        local_axis.legend(
            [local_handles[label] for label in LOCAL_SOURCE_LABELS],
            LOCAL_SOURCE_DISPLAY_LABELS,
            frameon=False,
            fontsize=FONT_SIZES["source_line_legend"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

        trans_axis.axhspan(0, 1, transform=trans_axis.get_yaxis_transform(), color="#FEF6EC", zorder=0)
        trans_handles = self.plot_source_lines(
            trans_axis,
            series_by_source,
            labels=TRANS_SOURCE_LABELS,
            alpha=0.92,
            draw_mean=False,
            emphasize=False,
        )
        self.style_timeseries_axis(
            trans_axis,
            years,
            trans_ylim,
            facecolor="#FEFBF7",
            right_ticks=True,
            scale_power=scale_power,
            annotate_scale=False,
        )
        self.add_axis_corner_title(trans_axis, "Non-Local")
        trans_axis.legend(
            [trans_handles[label] for label in TRANS_SOURCE_LABELS],
            TRANS_SOURCE_DISPLAY_LABELS,
            frameon=False,
            fontsize=FONT_SIZES["source_line_legend"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.55),
            ncol=2,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

    def style_timeseries_axis(
        self,
        ax: plt.Axes,
        years: np.ndarray,
        ylim: tuple[float, float],
        facecolor: str = "#FCFBF8",
        right_ticks: bool = False,
        show_ylabels: bool = True,
        show_xticklabels: bool = True,
        scale_power: int = 0,
        annotate_scale: bool = False,
    ) -> None:
        ax.set_xlim(float(years.min()) - 0.2, float(years.max()) + 0.2)
        ax.set_ylim(*ylim)
        tick_years = np.array([2015, 2018, 2021])
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(int(year)) for year in tick_years])
        ax.axhline(0.0, color="#6A6258", linewidth=0.65, zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E0D7", linestyle="-", linewidth=0.55)
        ax.grid(axis="x", color="#F1ECE4", linestyle="-", linewidth=0.35)
        ax.tick_params(
            axis="x",
            labelsize=FONT_SIZES["timeseries_xticks"],
            width=0.55,
            length=2.2,
            pad=1.2,
            colors="#3A3530",
        )
        ax.tick_params(
            axis="y",
            labelsize=FONT_SIZES["timeseries_yticks"],
            width=0.55,
            length=2.2,
            pad=1.2,
            colors="#3A3530",
        )
        ax.spines["top"].set_visible(False)
        if right_ticks:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_linewidth(0.75)
            ax.spines["right"].set_color("#7A7268")
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        else:
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.75)
            ax.spines["left"].set_color("#7A7268")
        ax.spines["bottom"].set_linewidth(0.75)
        ax.spines["bottom"].set_color("#7A7268")
        ax.set_facecolor(facecolor)
        self.apply_y_axis_format(
            ax,
            scale_power,
            annotate_scale=annotate_scale,
        )
        if not show_ylabels:
            ax.set_yticklabels([])
        if not show_xticklabels:
            ax.set_xticklabels([])

    def add_axis_corner_title(self, ax: plt.Axes, title: str | None) -> None:
        if title is None:
            return
        ax.text(
            0.02,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZES["group_label"],
            fontweight="bold",
            color="#2E2A26",
        )

    def plot_bar_panel(
        self,
        ax: plt.Axes,
        mean_values: dict[str, float],
        ylim: tuple[float, float],
        scale_power: int = 0,
    ) -> None:
        bar_source_specs = [NATURAL_SOURCE_SPEC, *SOURCE_SPECS]
        x = np.arange(len(bar_source_specs))
        values = [mean_values[str(spec["key"])] for spec in bar_source_specs]
        labels = [str(spec["tick_label"]) for spec in bar_source_specs]
        colors = [str(spec["color"]) for spec in bar_source_specs]

        bars = ax.bar(
            x,
            values,
            width=0.72,
            color=colors,
            edgecolor=(1, 1, 1, 0.95),
            linewidth=0.6,
            alpha=0.96,
            zorder=3,
        )
        ax.axhline(0.0, color="#6A6258", linewidth=0.65, zorder=1)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=FONT_SIZES["bar_xticks"],
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E0D7", linestyle="-", linewidth=0.55)
        ax.tick_params(axis="x", length=1.8, pad=1.0, colors="#3A3530")
        ax.tick_params(
            axis="y",
            labelsize=FONT_SIZES["bar_yticks"],
            length=1.8,
            pad=1.0,
            colors="#3A3530",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.spines["left"].set_color("#7A7268")
        ax.spines["bottom"].set_color("#7A7268")
        ax.set_facecolor("white")
        self.apply_y_axis_format(ax, scale_power, annotate_scale=True)

        group_ranges: dict[str, list[int]] = {}
        for index, source_spec in enumerate(bar_source_specs):
            group_ranges.setdefault(str(source_spec["group"]), []).append(index)

        for left_group, right_group in zip(GROUP_ORDER[:-1], GROUP_ORDER[1:]):
            if left_group not in group_ranges or right_group not in group_ranges:
                continue
            xpos = (group_ranges[left_group][-1] + group_ranges[right_group][0]) * 0.5
            ax.axvline(
                xpos,
                ymin=0.05,
                ymax=0.90,
                color="#D7D0C6",
                linewidth=1.8,
                zorder=4,
            )

        span = ylim[1] - ylim[0]
        place_labels_at_bottom = max(values) <= 0.0
        if place_labels_at_bottom:
            line_y = ylim[0] + span * 0.035
            text_y = line_y + span * 0.012
        else:
            line_y = ylim[1] - span * 0.10
            text_y = line_y + span * 0.035

        for group, indices in group_ranges.items():
            x0 = indices[0] - 0.34
            x1 = indices[-1] + 0.34
            ax.plot([x0, x1], [line_y, line_y], color="#B8AB9A", linewidth=0.8, clip_on=False, zorder=4)
            ax.text(
                (x0 + x1) * 0.5,
                text_y,
                group,
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["group_label"],
                color="#6A6258",
                clip_on=False,
            )

        label_positions = self.compute_bar_label_positions(values, tuple(ylim))
        for bar, value, y in zip(bars, values, label_positions):
            value_text = self.format_bar_value(value, scale_power)
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                y,
                value_text,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=FONT_SIZES["bar_value"],
                color="#4A433B",
                zorder=5,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=(1, 1, 1, 0.95))],
            )


    # 科学计数法放在右上角
    def apply_y_axis_format(
        self,
        ax: plt.Axes,
        scale_power: int = 0,
        *,
        annotate_scale: bool = False,
    ) -> None:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: self.format_scaled_value(value, scale_power)))
        ax.yaxis.get_offset_text().set_visible(False)
        if not annotate_scale:
            return

        if annotate_scale:
            self.add_scale_label(ax, scale_power, y=1.08)

    def compute_bar_label_positions(self, values: list[float], ylim: tuple[float, float]) -> list[float]:
        span = ylim[1] - ylim[0]
        if span <= 0:
            return values

        base_offset = span * BAR_VALUE_LAYOUT["base_offset_fraction"]
        close_threshold = span * BAR_VALUE_LAYOUT["close_threshold_fraction"]
        stagger_step = span * BAR_VALUE_LAYOUT["stagger_step_fraction"]
        edge_padding = span * BAR_VALUE_LAYOUT["edge_padding_fraction"]

        extra_offsets = [0.0] * len(values)
        close_groups = self.find_close_value_groups(values, close_threshold)

        for group in close_groups:
            if len(group) < 2:
                continue

            group_diffs = [abs(values[right] - values[left]) for left, right in zip(group[:-1], group[1:])]
            mean_diff = float(np.mean(group_diffs)) if group_diffs else 0.0
            closeness = 1.0 - min(mean_diff / close_threshold, 1.0) if close_threshold > 0 else 1.0
            group_step = stagger_step * max(closeness, 0.25)

            for level, index in enumerate(group):
                extra_offsets[index] = level * group_step

        positions: list[float] = []
        for value, extra_offset in zip(values, extra_offsets):
            if value >= 0:
                positions.append(value + base_offset + extra_offset)
            else:
                positions.append(value - base_offset - extra_offset)

        for group in close_groups:
            if not group:
                continue

            if all(values[index] >= 0 for index in group):
                max_position = max(positions[index] for index in group)
                overflow = max_position - (ylim[1] - edge_padding)
                if overflow > 0:
                    for index in group:
                        positions[index] -= overflow

                sorted_group = sorted(group, key=lambda idx: values[idx])
                for left, right in zip(sorted_group[:-1], sorted_group[1:]):
                    min_gap = max(stagger_step * 0.8, span * 0.01)
                    if positions[right] - positions[left] < min_gap:
                        positions[right] = positions[left] + min_gap

                max_position = max(positions[index] for index in group)
                overflow = max_position - (ylim[1] - edge_padding)
                if overflow > 0:
                    for index in group:
                        positions[index] -= overflow

            elif all(values[index] < 0 for index in group):
                min_position = min(positions[index] for index in group)
                overflow = (ylim[0] + edge_padding) - min_position
                if overflow > 0:
                    for index in group:
                        positions[index] += overflow

                sorted_group = sorted(group, key=lambda idx: values[idx], reverse=True)
                for left, right in zip(sorted_group[:-1], sorted_group[1:]):
                    min_gap = max(stagger_step * 0.8, span * 0.01)
                    if positions[left] - positions[right] < min_gap:
                        positions[right] = positions[left] - min_gap

                min_position = min(positions[index] for index in group)
                overflow = (ylim[0] + edge_padding) - min_position
                if overflow > 0:
                    for index in group:
                        positions[index] += overflow

        return positions

    def find_close_value_groups(self, values: list[float], close_threshold: float) -> list[list[int]]:
        if not values:
            return []

        groups: list[list[int]] = []
        current_group = [0]

        for index in range(1, len(values)):
            previous_value = values[index - 1]
            current_value = values[index]
            same_sign = (previous_value >= 0 and current_value >= 0) or (previous_value < 0 and current_value < 0)
            close_in_height = abs(current_value - previous_value) <= close_threshold

            if same_sign and close_in_height:
                current_group.append(index)
            else:
                groups.append(current_group)
                current_group = [index]

        groups.append(current_group)
        return groups

    def cyclic_lon_lat_data(self, field_2d: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data, lon = add_cyclic_point(field_2d.values, coord=field_2d["lon"].values)
        lat = field_2d["lat"].values
        return lon, lat, data

    def add_circular_boundary(self, ax: plt.Axes) -> None:
        theta = np.linspace(0, 2 * np.pi, 256)
        center = np.array([0.5, 0.5])
        radius = 0.5
        vertices = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(vertices * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    def dedup_ticks(self, ticks: list[float]) -> list[float]:
        deduped_ticks: list[float] = []
        for tick in ticks:
            if not any(np.isclose(tick, existing) for existing in deduped_ticks):
                deduped_ticks.append(tick)
        return deduped_ticks

    def display_power(self, arrays: list[np.ndarray]) -> int:
        finite_values: list[np.ndarray] = []
        for array in arrays:
            values = np.asarray(array, dtype=float)
            finite = np.abs(values[np.isfinite(values)])
            if finite.size:
                finite_values.append(finite)

        if not finite_values:
            return 0

        max_abs = float(np.max(np.concatenate(finite_values)))
        if np.isclose(max_abs, 0.0):
            return 0
        return int(np.floor(np.log10(max_abs)))

    def format_multiplier(self, power: int) -> str:
        return "" if power == 0 else rf"$10^{{{power}}}$"

    def format_scaled_value(self, value: float, power: int) -> str:
        scale = 10.0**power
        scaled_value = float(value) / scale
        if np.isclose(scaled_value, 0.0):
            scaled_value = 0.0
        return f"{scaled_value:.{SCIENTIFIC_DECIMALS}f}"

    def format_bar_value(self, value: float, power: int) -> str:
        return self.format_scaled_value(value, power)

    def format_scientific_value(self, value: float) -> str:
        if np.isclose(value, 0.0):
            return "0.00"
        exponent = int(np.floor(np.log10(abs(value))))
        if exponent == 0:
            return f"{value:.{SCIENTIFIC_DECIMALS}f}"
        coefficient = value / (10.0**exponent)
        return rf"${coefficient:.{SCIENTIFIC_DECIMALS}f}\times10^{{{exponent}}}$"

    def nice_ceiling(self, value: float) -> float:
        if value <= 0:
            return 1.0
        exponent = np.floor(np.log10(value))
        fraction = value / (10 ** exponent)
        for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            if fraction <= candidate:
                return candidate * (10 ** exponent)
        return 10 ** (exponent + 1)

    def nice_floor(self, value: float) -> float:
        if value >= 0:
            return 0.0
        return -self.nice_ceiling(abs(value))

    def save_figure(self, fig, output_dir, stem) -> None:
        save_ep_figure(fig, self.output_dir / self.output_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 for Arctic radiative effects.")
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


def main() -> None:
    args = parse_args()
    plotter = Figure2Plotter(args.output_dir, args.dpi)
    plotter.run()


if __name__ == "__main__":
    main()


def render_radiative_figure(component, panel_index, stem):
    directory = Path(__file__).resolve().parents[2] / "figures" / ("main" if stem.startswith("fig4_") else "supplementary")
    plotter = Figure2Plotter(directory, 1000)
    plotter.output_stem = stem
    plotter.load_datasets()
    try:
        plotter.plot_component_panel_figure(component, PANEL_SPECS[panel_index])
    finally:
        plotter.close_datasets()
