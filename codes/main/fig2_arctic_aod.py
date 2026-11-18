"""Figure 2: Total, anthropogenic and natural pan-Arctic AOD.

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

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
import numpy as np
import xarray as xr

plt.rcParams.update({"font.family": "Arial", "text.usetex": False, "figure.dpi": 100})


ROOT_DIR = Path(__file__).resolve().parents[2] / "data" / "aod"
DEFAULT_BASE = ROOT_DIR / "aod_gc_base.nc"
DEFAULT_NATURAL = ROOT_DIR / "aod_gc_noall.nc"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "main"

PM_VAR = "RadAOD550nm_PM"
ARCTIC_LAT_MIN = 50.0
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
ANTHROPOGENIC_SPECIES = ["BC", "OC", "SU", "AM+NI", "PM"]
NATURAL_SPECIES = ["SS", "DU"]
SOURCE_XTICKS = [2015, 2018, 2021]
FIGURE1_STEM_PREFIX = "fig1"

SPECIES_COMPONENTS = {
    "BC": ("RadAOD550nm_BC",),
    "OC": ("RadAOD550nm_OA",),
    "SU": ("RadAOD550nm_SU",),
    "AM+NI": ("RadAOD550nm_AM", "RadAOD550nm_NI"),
    "SS": ("RadAOD550nm_SS",),
    "DU": ("RadAOD550nm_DU",),
    "PM": (PM_VAR,),
}
DISPLAY_SPECIES = {
    "BC": "BC",
    "OC": "OC",
    "SU": "SU",
    "AM+NI": "NI",
    "SS": "SS",
    "DU": "DU",
    "PM": "Total",
}

DATASET_COLORS = {"total": "#9A6B38", "anth": "#C46A4D", "natural": "#6E8B63"}
FONT_SIZES = {
    "Figure1_panel_title": 15.0,
    "Figure1_inline_colorbar_ticks": 12.6,
    "Figure1_inset_bar_xticks": 9.8,
    "Figure1_inset_bar_yticks": 15.0,
    "Figure1_timeseries_ticks": 15.6,
    "Figure1_source_line_legend": 11.4,
    "Figure1_source_group_title": 15.0,
    "Figure1_source_impact_ticks": 15.0,
    "Figure1_source_impact_values": 15.0,
    "Figure1_source_impact_group": 15.2,
}
FIGURE1_MAP_STYLE = {
    "land_facecolor": "#F6F1E8",
    "ocean_facecolor": "#FCFBF8",
    "gridline_color": "#BBB3A8",
    "gridline_alpha": 0.22,
    "latitude_circle_color": "#D3CBC0",
    "coastline_color": "#8A847B",
    "coastline_linewidth": 0.55,
    "spine_color": "#A79E93",
    "spine_linewidth": 0.65,
}

# Figure 1 and Figure 2 both use these zero-out experiment definitions.
FIGURE1_SOURCE_SPECS = [
    {"label": "ANT", "color": "#B24B3A", "group": "ANT", "linewidth": 2.4, "markersize": 3.8, "zorder": 6},
    {"label": "Local", "path": ROOT_DIR / "aod_gc_nonp.nc", "color": "#5E84A7", "group": "Local", "linewidth": 2.0, "markersize": 3.5, "zorder": 5},
    {"label": "N. Europe", "path": ROOT_DIR / "aod_gc_no_north_europe.nc", "color": "#3F678D", "group": "Local", "linewidth": 1.45, "markersize": 3.0, "zorder": 4},
    {"label": "Other Local", "display_label": "Other", "path": ROOT_DIR / "aod_gc_no_north_europe_sup.nc", "color": "#9DBDD9", "group": "Local", "linewidth": 1.45, "markersize": 3.0, "linestyle": "--", "zorder": 3},
    {"label": "Non-Local", "path": ROOT_DIR / "aod_gc_nonp_sup.nc", "color": "#C48A4E", "group": "Non-Local", "linewidth": 2.0, "markersize": 3.5, "zorder": 5},
    {"label": "S. Europe", "path": ROOT_DIR / "aod_gc_no_south_europe.nc", "color": "#7FA46B", "group": "Non-Local", "linewidth": 1.45, "markersize": 3.0, "zorder": 4},
    {"label": "E. Asia", "path": ROOT_DIR / "aod_gc_no_east_asia.nc", "color": "#4C9A95", "group": "Non-Local", "linewidth": 1.45, "markersize": 3.0, "zorder": 4},
    {"label": "N. America", "path": ROOT_DIR / "aod_gc_no_north_america.nc", "color": "#D2AA58", "group": "Non-Local", "linewidth": 1.45, "markersize": 3.0, "zorder": 4},
    {"label": "Other Non-Local", "display_label": "Other", "path": ROOT_DIR / "aod_gc_no_three_regions_sup.nc", "color": "#8E8375", "group": "Non-Local", "linewidth": 1.45, "markersize": 3.0, "linestyle": "--", "zorder": 3},
]
FIGURE1_NATURAL_SOURCE_SPEC = {"label": "NAT", "color": "#6E8B63", "group": "NAT", "linewidth": 2.4, "markersize": 3.8, "linestyle": "-", "zorder": 6}

SOURCE_MARKERS = {"ANT": "D", "NAT": "D", "Local": "o", "Non-Local": "s"}
OVERVIEW_CMAPS = {
    "total": LinearSegmentedColormap.from_list("overview_total", ["#FDFBF7", "#F0E0C6", "#DCBC8C", "#BE8852", "#875326"]),
    "anth": LinearSegmentedColormap.from_list("overview_anth", ["#FEFBFA", "#F4DDD7", "#E5B4A5", "#CB7E68", "#973D35"]),
    "natural": LinearSegmentedColormap.from_list("overview_natural", ["#FCFBF7", "#E9EEE0", "#C8D2B9", "#95A985", "#607857"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2 of Arctic aerosol overview.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Path to aod_gc_base.nc")
    parser.add_argument("--natural", type=Path, default=DEFAULT_NATURAL, help="Path to aod_gc_noall.nc")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory used for exported figures")
    parser.add_argument("--dpi", type=int, default=1000, help="Export DPI for PNG/PDF output")
    parser.add_argument("--species", nargs="+", choices=ANTHROPOGENIC_SPECIES, default=["PM"])
    return parser.parse_args()


# These helpers stay public because Figure 2 imports them.
def nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10 ** exponent)
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= candidate:
            return candidate * (10 ** exponent)
    return 10 ** (exponent + 1)


def scientific_power(values: np.ndarray | list[float]) -> int:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0
    max_abs = float(np.max(np.abs(finite_values)))
    if np.isclose(max_abs, 0.0):
        return 0
    return int(np.floor(np.log10(max_abs)))


def format_scientific_multiplier(power: int) -> str:
    return rf"$\times 10^{{{power}}}$"


def get_species_field(ds: xr.Dataset, species: str) -> xr.DataArray:
    field = ds[SPECIES_COMPONENTS[species][0]]
    for component in SPECIES_COMPONENTS[species][1:]:
        field = field + ds[component]
    return field


def area_weighted_mean(field: xr.DataArray, lat_min: float = ARCTIC_LAT_MIN) -> xr.DataArray:
    subset = field.sel(lat=slice(lat_min, 90))
    weights = xr.DataArray(np.cos(np.deg2rad(subset["lat"].values)), coords={"lat": subset["lat"].values}, dims=("lat",))
    return subset.weighted(weights).mean(dim=("lat", "lon"))


def scalar_weighted_mean(field: xr.DataArray, lat_min: float = ARCTIC_LAT_MIN) -> float:
    return float(area_weighted_mean(field, lat_min=lat_min).mean().values)


def format_title_scientific(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0.00"
    power = scientific_power([value])
    return rf"{value / (10.0**power):.2f}$\times 10^{{{power}}}$"


def make_anthropogenic_dataset(base: xr.Dataset, natural: xr.Dataset) -> xr.Dataset:
    return xr.Dataset(
        {name: base[name] - natural[name] for name in base.data_vars if name in natural.data_vars},
        coords=base.coords,
        attrs={**base.attrs, "title": "Derived anthropogenic aerosol (base - noall)"},
    )


def seasonal_climatology(field: xr.DataArray) -> xr.DataArray:
    return area_weighted_mean(field).groupby("time.season").mean().reindex(season=SEASON_ORDER)


def annual_mean_series(field: xr.DataArray) -> xr.DataArray:
    return area_weighted_mean(field).groupby("time.year").mean().load()


def decadal_mean_map(field: xr.DataArray) -> xr.DataArray:
    return field.mean(dim="time").sel(lat=slice(ARCTIC_LAT_MIN, 90))


def source_impact_field(base_field: xr.DataArray, anthropogenic_field: xr.DataArray, spec: dict[str, object], species: str) -> xr.DataArray:
    if "path" not in spec:
        return anthropogenic_field.load()
    with xr.open_dataset(spec["path"]) as removed:
        return (base_field - get_species_field(removed, species)).load()


def axis_limits(values: np.ndarray, upper_pad_fraction: float = 0.18) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    if np.isclose(span, 0.0):
        span = max(abs(vmax) * 0.08, 1e-12)
    return max(0.0, vmin - span * 0.14), nice_ceiling(vmax + span * upper_pad_fraction)


def draw_map_panel(
    ax: plt.Axes,
    field_map: xr.DataArray,
    norm: Normalize,
    cmap,
    panel_letter: str,
    title: str,
    colorbar_ticks: list[float],
    colorbar_labels: list[str],
    seasonal_values: xr.DataArray,
    seasonal_max: float,
    seasonal_color: str,
) -> None:
    # Map, inline colorbar, and seasonal inset are drawn together to keep layout logic local.
    style = FIGURE1_MAP_STYLE
    ax.set_extent([-180, 180, ARCTIC_LAT_MIN, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=style["land_facecolor"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=style["ocean_facecolor"], edgecolor="none", zorder=0)
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180, 181, 60),
        ylocs=[60, 70, 80],
        linewidth=0.45,
        color=style["gridline_color"],
        alpha=style["gridline_alpha"],
        linestyle="--",
        zorder=1,
    )
    gridlines.n_steps = 100
    lons = np.linspace(-180, 180, 721)
    ax.plot(lons, np.full_like(lons, ARCTIC_LAT_MIN), transform=ccrs.PlateCarree(), color=style["latitude_circle_color"], linewidth=0.8, zorder=2)
    ax.coastlines(resolution="50m", linewidth=style["coastline_linewidth"], color=style["coastline_color"], zorder=3)
    circle = mpath.Path(np.vstack([np.sin(np.linspace(0, 2 * np.pi, 256)), np.cos(np.linspace(0, 2 * np.pi, 256))]).T * 0.5 + np.array([0.5, 0.5]))
    ax.set_boundary(circle, transform=ax.transAxes)
    if ax.spines.get("geo") is not None:
        ax.spines["geo"].set_linewidth(style["spine_linewidth"])
        ax.spines["geo"].set_edgecolor(style["spine_color"])

    data, cyclic_lon = add_cyclic_point(field_map.values, coord=field_map["lon"].values)
    mesh = ax.pcolormesh(cyclic_lon, field_map["lat"].values, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto", rasterized=True, zorder=1.5, alpha=0.97)
    ax.text(0.5, 1.045, f"({panel_letter}) {title}", transform=ax.transAxes, ha="center", va="bottom", fontsize=FONT_SIZES["Figure1_panel_title"], color="#111111")

    cax = ax.inset_axes([0.18, -0.05, 0.64, 0.035])
    cax.set_facecolor((1, 1, 1, 0.78))
    colorbar = plt.colorbar(mesh, cax=cax, orientation="horizontal")
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(colorbar_labels)
    colorbar.outline.set_linewidth(0.55)
    colorbar.outline.set_edgecolor("#796C5C")
    colorbar.ax.tick_params(labelsize=FONT_SIZES["Figure1_inline_colorbar_ticks"], width=0.45, length=1.8, pad=1.2, colors="#3F3A35")

    inset = ax.inset_axes([0.02, 0.17, 0.45, 0.25])
    season_values = [float(seasonal_values.sel(season=season).values) for season in SEASON_ORDER]
    inset.bar(np.arange(len(SEASON_ORDER)), season_values, width=0.74, color=seasonal_color, edgecolor="white", linewidth=0.35, alpha=0.8, zorder=3)
    inset.set_ylim(0, max(seasonal_max, 1e-10))
    inset.set_xticks(np.arange(len(SEASON_ORDER)))
    inset.set_xticklabels(SEASON_ORDER)
    inset.set_yticks([0, seasonal_max])
    inset.set_yticklabels([f"{0.0:.3f}", f"{seasonal_max:.3f}"])
    inset.set_axisbelow(True)
    inset.grid(axis="y", color=(1, 1, 1, 0.72), linestyle="-", linewidth=0.4)
    inset.tick_params(axis="x", labelsize=FONT_SIZES["Figure1_inset_bar_xticks"], length=0.0, pad=0.8)
    inset.tick_params(axis="y", labelsize=FONT_SIZES["Figure1_inset_bar_yticks"], length=1.2, pad=0.8)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.45)
    inset.spines["bottom"].set_linewidth(0.45)
    inset.spines["left"].set_color((0.15, 0.15, 0.15, 0.75))
    inset.spines["bottom"].set_color((0.15, 0.15, 0.15, 0.75))
    inset.set_facecolor((1, 1, 1, 0.62))


def draw_source_variability_panel(container_ax: plt.Axes, source_series: dict[str, xr.DataArray]) -> None:
    specs = {spec["label"]: spec for spec in FIGURE1_SOURCE_SPECS}
    specs["NAT"] = FIGURE1_NATURAL_SOURCE_SPEC
    panels = [
        ("major", ["ANT", "NAT"], [0.01, 0.54, 0.44, 0.48], None, "#FCFBF8", False, False, True),
        ("grouped", ["Local", "Non-Local"], [0.50, 0.54, 0.44, 0.48], None, "#FDFBF8", False, True, True),
        ("local", ["N. Europe", "Other Local"], [0.01, 0.001, 0.44, 0.48], "Local", "#F8FBFE", True, False, False),
        ("non_local", ["S. Europe", "E. Asia", "N. America", "Other Non-Local"], [0.50, 0.001, 0.44, 0.48], "Non-Local", "#FEFBF7", True, True, False),
    ]
    all_values = np.concatenate([source_series[label].values for _, labels, *_ in panels for label in labels])
    scale_power = scientific_power(all_values)
    scale = 10.0**scale_power
    years = source_series["ANT"]["year"].values
    xtick_years = np.asarray([year for year in SOURCE_XTICKS if float(years.min()) <= year <= float(years.max())])

    for index, (_, labels, rect, title, facecolor, show_xlabels, right_ticks, emphasize) in enumerate(panels):
        ax = container_ax.inset_axes(rect)
        if title == "Local":
            ax.axhspan(0, 1, transform=ax.get_yaxis_transform(), color=to_rgba("#5E84A7", 0.06), zorder=0)
        elif title == "Non-Local":
            ax.axhspan(0, 1, transform=ax.get_yaxis_transform(), color=to_rgba("#C48A4E", 0.06), zorder=0)

        handles = []
        for label in labels:
            spec = specs[label]
            linewidth = float(spec.get("linewidth", 1.5))
            line, = ax.plot(
                source_series[label]["year"].values,
                source_series[label].values,
                color=spec["color"],
                linewidth=linewidth,
                linestyle=spec.get("linestyle", "-"),
                marker=SOURCE_MARKERS.get(spec["group"], "s"),
                markersize=spec.get("markersize", 3.0),
                markerfacecolor=spec["color"],
                markeredgecolor="white",
                markeredgewidth=0.55 if emphasize else 0.4,
                alpha=0.98 if emphasize else 0.92,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=spec.get("zorder", 3),
            )
            if emphasize:
                line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.4, foreground=(1, 1, 1, 0.95)), pe.Normal()])
                ax.axhline(float(source_series[label].mean().values), color=spec["color"], linestyle=(0, (2.0, 2.0)), linewidth=0.7, alpha=0.28, zorder=1)
            handles.append(line)

        values = np.concatenate([source_series[label].values for label in labels])
        ax.set_xlim(float(years.min()) - 0.2, float(years.max()) + 0.2)
        ax.set_ylim(*axis_limits(values, upper_pad_fraction=0.20 if title else 0.18))
        ax.set_xticks(xtick_years)
        if show_xlabels:
            ax.set_xticklabels([str(int(year)) for year in xtick_years], rotation=0)
        else:
            ax.tick_params(axis="x", labelbottom=False, length=0.0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / scale:.2f}"))
        # Keep only the upper-left scale multiplier; this removes the duplicate BC label in the lower-left inset.
        if index == 0:
            ax.text(0.02, 1.05, format_scientific_multiplier(scale_power), transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZES["Figure1_timeseries_ticks"], color="#3A3530", zorder=10)

        ax.tick_params(axis="both", labelsize=FONT_SIZES["Figure1_timeseries_ticks"], width=0.55, length=2.2, pad=1.4, colors="#3A3530")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E0D7", linestyle="-", linewidth=0.55)
        ax.grid(axis="x", color="#F1ECE4", linestyle="-", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(not right_ticks)
        ax.spines["right"].set_visible(right_ticks)
        ax.spines["bottom"].set_visible(True)
        for side in ("left", "right", "bottom"):
            ax.spines[side].set_linewidth(0.75)
            ax.spines[side].set_color("#7A7268")
        if right_ticks:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if title:
            ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZES["Figure1_source_group_title"], fontweight="bold", color="#4B443C", zorder=9)
        ax.set_facecolor(facecolor)
        legend_labels = [str(specs[label].get("display_label", label)) for label in labels]
        ax.legend(handles, legend_labels, frameon=False, fontsize=FONT_SIZES["Figure1_source_line_legend"], loc="upper center", bbox_to_anchor=(0.56 if title is None else 0.5, 0.92 if title is None else -0.22), ncol=2, handlelength=1.8, columnspacing=0.8, handletextpad=0.45, borderaxespad=0.0)


def draw_natural_variability_panel(ax: plt.Axes, source_series: dict[str, xr.DataArray]) -> None:
    values = source_series["NAT"].values
    years = source_series["NAT"]["year"].values
    scale_power = scientific_power(values)
    scale = 10.0**scale_power
    xtick_years = np.asarray([year for year in SOURCE_XTICKS if float(years.min()) <= year <= float(years.max())])
    spec = FIGURE1_NATURAL_SOURCE_SPEC

    line, = ax.plot(
        years,
        values,
        color=spec["color"],
        linewidth=spec["linewidth"],
        linestyle=spec["linestyle"],
        marker=SOURCE_MARKERS[spec["group"]],
        markersize=spec["markersize"],
        markerfacecolor=spec["color"],
        markeredgecolor="white",
        markeredgewidth=0.55,
        alpha=0.98,
        zorder=spec["zorder"],
    )
    line.set_path_effects([pe.Stroke(linewidth=spec["linewidth"] + 1.4, foreground=(1, 1, 1, 0.95)), pe.Normal()])
    ax.axhline(float(source_series["NAT"].mean().values), color=spec["color"], linestyle=(0, (2.0, 2.0)), linewidth=0.7, alpha=0.28, zorder=1)

    ax.set_xlim(float(years.min()) - 0.2, float(years.max()) + 0.2)
    ax.set_ylim(*axis_limits(values))
    ax.set_xticks(xtick_years)
    ax.set_xticklabels([str(int(year)) for year in xtick_years], rotation=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / scale:.2f}"))
    ax.text(0.02, 1.05, format_scientific_multiplier(scale_power), transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZES["Figure1_timeseries_ticks"], color="#3A3530", zorder=10)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["Figure1_timeseries_ticks"], width=0.55, length=2.2, pad=1.4, colors="#3A3530")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E0D7", linestyle="-", linewidth=0.55)
    ax.grid(axis="x", color="#F1ECE4", linestyle="-", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["left"].set_color("#7A7268")
    ax.spines["bottom"].set_color("#7A7268")
    ax.text(0.02, 0.96, "NAT", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZES["Figure1_source_group_title"], fontweight="bold", color="#4B443C", zorder=9)
    ax.set_facecolor("#FCFBF8")
    ax.legend([line], ["NAT"], frameon=False, fontsize=FONT_SIZES["Figure1_source_line_legend"], loc="upper center", bbox_to_anchor=(0.56, 0.92), ncol=1)


def draw_source_impact_panel(ax: plt.Axes, values: list[float], labels: list[str], colors: list[str], groups: list[str]) -> None:
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.72, color=colors, edgecolor=(1, 1, 1, 0.94), linewidth=0.6, alpha=0.96, zorder=3)
    ymax = nice_ceiling(max(values) * 1.22)
    scale_power = scientific_power(values)
    scale = 10.0**scale_power

    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    for label in ax.get_xticklabels():
        label.set_x(label.get_position()[0] + 0.08)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / scale:.2f}"))
    ax.text(0.02, 1.05, format_scientific_multiplier(scale_power), transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZES["Figure1_timeseries_ticks"], color="#3A3530", zorder=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E0D7", linestyle="-", linewidth=0.55)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["Figure1_source_impact_ticks"], length=1.8, pad=1.2, colors="#3A3530")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["left"].set_color("#7A7268")
    ax.spines["bottom"].set_color("#7A7268")

    group_ranges: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        group_ranges.setdefault(group, []).append(idx)
    for idx, group in enumerate(groups[1:], start=1):
        if group != groups[idx - 1]:
            ax.axvline(idx - 0.5, ymin=0.06, ymax=0.88, color="#D7D0C6", linewidth=2.2, zorder=2)
    for group, indices in group_ranges.items():
        x0 = indices[0] - 0.35
        x1 = indices[-1] + 0.35
        y = ymax * 0.93
        ax.plot([x0, x1], [y, y], color="#B8AB9A", linewidth=0.8, clip_on=False)
        ax.text((x0 + x1) * 0.5, y + ymax * 0.03, group, ha="center", va="bottom", fontsize=FONT_SIZES["Figure1_source_impact_group"], color="#6A6258")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() + ymax * 0.028, f"{value / scale:.2f}", ha="center", va="bottom", fontsize=FONT_SIZES["Figure1_source_impact_values"], color="#4A433B")


def create_component_figure(base: xr.Dataset, natural: xr.Dataset, anth: xr.Dataset, output_dir: Path, dpi: int, species: str) -> Path:
    fields = {"total": get_species_field(base, species), "anth": get_species_field(anth, species), "natural": get_species_field(natural, species)}
    maps = {key: decadal_mean_map(field) for key, field in fields.items()}
    vmax = nice_ceiling(max(float(field.max().values) for field in maps.values()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    colorbar_ticks = [0.0, vmax / 2.0, vmax]
    colorbar_labels = [f"{tick:.3f}" for tick in colorbar_ticks]
    seasonal_series = {key: seasonal_climatology(field) for key, field in fields.items()}
    seasonal_max = nice_ceiling(max(float(values.max().values) for values in seasonal_series.values()) * 1.12)

    source_series: dict[str, xr.DataArray] = {}
    source_impacts = [float(annual_mean_series(fields["natural"]).mean().values)]
    source_labels = ["NAT"]
    source_colors = [FIGURE1_NATURAL_SOURCE_SPEC["color"]]
    source_groups = ["NAT"]
    for spec in FIGURE1_SOURCE_SPECS:
        series = annual_mean_series(source_impact_field(fields["total"], fields["anth"], spec, species))
        source_series[spec["label"]] = series
        source_impacts.append(float(series.mean().values))
        source_labels.append(str(spec.get("display_label", spec["label"])))
        source_colors.append(spec["color"])
        source_groups.append(spec["group"])
    source_series["NAT"] = annual_mean_series(fields["natural"])

    fig = plt.figure(figsize=(12.1, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.93, bottom=0.105, height_ratios=[1.0, 0.98], hspace=0.34, wspace=0.30)
    proj = ccrs.NorthPolarStereo(central_longitude=20)
    axes = {"total": fig.add_subplot(gs[0, 0], projection=proj), "anth": fig.add_subplot(gs[0, 1], projection=proj), "natural": fig.add_subplot(gs[0, 2], projection=proj)}

    dataset_names = {"total": "Total", "anth": "ANT", "natural": "NAT"}
    for panel_letter, key in zip(["a", "b", "c"], ["total", "anth", "natural"]):
        display_name = "AOD" if species == "PM" else f"{DISPLAY_SPECIES[species]} AOD"
        title = f"{dataset_names[key]} {display_name} ({format_title_scientific(scalar_weighted_mean(fields[key]))})"
        draw_map_panel(axes[key], maps[key], norm, OVERVIEW_CMAPS[key], panel_letter, title, colorbar_ticks, colorbar_labels, seasonal_series[key], seasonal_max, DATASET_COLORS[key])

    bottom_gs = gs[1, :].subgridspec(1, 2, wspace=0.17)
    ax_d = fig.add_subplot(bottom_gs[0, 0])
    ax_e = fig.add_subplot(bottom_gs[0, 1])
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_facecolor("white")

    ax_d.text(0.5, 1.045, f"(d) {DISPLAY_SPECIES[species]} Source Variability", transform=ax_d.transAxes, ha="center", va="bottom", fontsize=FONT_SIZES["Figure1_panel_title"], color="#111111")
    draw_source_variability_panel(ax_d, source_series)
    ax_e.text(0.5, 1.045, f"(e) {DISPLAY_SPECIES[species]} Source Impacts", transform=ax_e.transAxes, ha="center", va="bottom", fontsize=FONT_SIZES["Figure1_panel_title"], color="#111111")
    draw_source_impact_panel(ax_e, source_impacts, source_labels, source_colors, source_groups)

    stems = {"PM": "fig2_arctic_aod", "AM+NI": "figs8_nitrate_aod", "BC": "figs9_black_carbon_aod", "OC": "figs10_organic_carbon_aod", "SU": "figs11_sulfate_aod"}
    png_path, _ = save_ep_figure(fig, output_dir / stems[species])
    plt.close(fig)
    return png_path


def print_summary(base: xr.Dataset, natural: xr.Dataset, anth: xr.Dataset, species_list: list[str]) -> None:
    print(r"Arctic (>50 deg N) area-weighted ten-year mean AOD")
    for species in species_list:
        total_mean = scalar_weighted_mean(get_species_field(base, species))
        natural_mean = scalar_weighted_mean(get_species_field(natural, species))
        anth_mean = scalar_weighted_mean(get_species_field(anth, species))
        print(f"  {DISPLAY_SPECIES[species]:>5}: total={total_mean:.5f}, anth={anth_mean:.5f}, natural={natural_mean:.5f}")


def main() -> None:
    args = parse_args()
    base = xr.open_dataset(args.base)
    natural = xr.open_dataset(args.natural)
    anth = make_anthropogenic_dataset(base, natural)
    try:
        print_summary(base, natural, anth, args.species)
        for species in args.species:
            print(f"Saved Figure 2 to {create_component_figure(base, natural, anth, args.output_dir, args.dpi, species)}")
    finally:
        base.close()
        natural.close()
        anth.close()


if __name__ == "__main__":
    main()


def render_aod(species, output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "figures" / ("main" if species == "PM" else "supplementary")
    with xr.open_dataset(DEFAULT_BASE) as base, xr.open_dataset(DEFAULT_NATURAL) as natural:
        anth = make_anthropogenic_dataset(base, natural)
        try:
            print_summary(base, natural, anth, [species])
            return create_component_figure(base, natural, anth, output_dir, 1000, species)
        finally:
            anth.close()
