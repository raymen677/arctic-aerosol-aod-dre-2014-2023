"""Figure S3: MERRA-2 total and component AOD scatter evaluation.

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
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ══════════════════════════════════════════════════════════════
# Global matplotlib style — Nature journal conventions
# ══════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    7,
    "axes.titlesize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   5.5,
    "figure.dpi":        100,
    "savefig.dpi":       1000,
    "savefig.bbox":      "tight",
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,     # editable text in PDF
    "ps.fonttype":       42,
})

# ══════════════════════════════════════════════════════════════
# Wong (2011) colorblind-safe palette — Nature Methods
# ══════════════════════════════════════════════════════════════
WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
}

# Component metadata — Wong palette mapping
COMP_META = {
    "TOT(PM)": dict(gc="RadAOD550nm_PM", m2="TOTEXTTAU",
                     label="Total AOD",      color=WONG["black"]),
    "DU":      dict(gc="RadAOD550nm_DU", m2="DUEXTTAU",
                     label="Dust",           color=WONG["vermillion"]),
    "SS":      dict(gc="RadAOD550nm_SS", m2="SSEXTTAU",
                     label="Sea Salt",       color=WONG["blue"]),
    "OA":      dict(gc="RadAOD550nm_OA", m2="OCEXTTAU",
                     label="Organic Carbon", color=WONG["green"]),
    "BC":      dict(gc="RadAOD550nm_BC", m2="BCEXTTAU",
                     label="Black Carbon",   color=WONG["orange"]),
    "SU":      dict(gc="RadAOD550nm_SU", m2="SUEXTTAU",
                     label="Sulfate",        color=WONG["pink"]),
}

PANEL_LABELS = list("abcdef")

# ══════════════════════════════════════════════════════════════
# Data paths
# ══════════════════════════════════════════════════════════════
GC_FILE = Path(__file__).resolve().parents[2] / "data" / "aod" / "aod_gc_base.nc"
M2_FILE = Path(__file__).resolve().parents[2] / "data" / "observations" / "aod_merra2_components.nc"

# ══════════════════════════════════════════════════════════════
# 1. Load & Regrid
# ══════════════════════════════════════════════════════════════
lat_min, lat_max = 50, 90
lon_min, lon_max = -180, 180

ds_gc = xr.open_dataset(GC_FILE).sel(lon=slice(lon_min, lon_max),
                                      lat=slice(lat_min, lat_max))
ds_m2 = xr.open_dataset(M2_FILE).sel(lon=slice(lon_min, lon_max),
                                      lat=slice(lat_min, lat_max))

gc_lat = ds_gc["lat"].values
gc_lon = ds_gc["lon"].values
ds_m2_rg = ds_m2.interp(lat=gc_lat, lon=gc_lon, method="linear")

print(f"  GC grid : {len(gc_lat)} lat × {len(gc_lon)} lon")
print(f"  Time    : {ds_gc.dims['time']} months")


# ══════════════════════════════════════════════════════════════
# 2. Statistics
# ══════════════════════════════════════════════════════════════
def calc_stats(obs: np.ndarray, mod: np.ndarray,
               weights: np.ndarray) -> dict:
    """Calculate latitude-area-weighted spatial statistics."""
    mask = (np.isfinite(obs) & np.isfinite(mod) & np.isfinite(weights)
            & (obs >= 0) & (mod >= 0) & (weights > 0))
    o = obs[mask].ravel()
    m = mod[mask].ravel()
    w = weights[mask].ravel()
    N = len(o)
    if N < 2:
        return {k: np.nan for k in
                ["R", "R2", "RMSE", "MBE", "NMB", "slope", "N"]}

    weight_sum = np.sum(w)
    obs_mean = np.sum(w * o) / weight_sum
    mod_mean = np.sum(w * m) / weight_sum
    obs_anom = o - obs_mean
    mod_anom = m - mod_mean

    covariance = np.sum(w * obs_anom * mod_anom)
    variance_product = (np.sum(w * obs_anom ** 2)
                        * np.sum(w * mod_anom ** 2))
    r = covariance / np.sqrt(variance_product) if variance_product > 0 else np.nan
    rmse = np.sqrt(np.sum(w * (m - o) ** 2) / weight_sum)
    mbe = np.sum(w * (m - o)) / weight_sum
    nmb = mbe / obs_mean * 100.0 if obs_mean != 0 else np.nan
    slope_denominator = np.sum(w * o ** 2)
    slope = (np.sum(w * o * m) / slope_denominator
             if slope_denominator > 0 else np.nan)
    return dict(R=r, R2=r**2, RMSE=rmse, MBE=mbe,
                NMB=nmb, slope=slope, N=N)


def collect_component_stats(time_mean: bool = False) -> dict:
    stats = {}
    for key, meta in COMP_META.items():
        gc_var = ds_gc[meta["gc"]]
        m2_var = ds_m2_rg[meta["m2"]]
        if time_mean:
            gc_var = gc_var.mean(dim="time")
            m2_var = m2_var.mean(dim="time")
        # Grid-cell area on a regular lon-lat grid is proportional to cos(lat).
        lat_weights = np.cos(np.deg2rad(gc_var["lat"]))
        area_weights = lat_weights.broadcast_like(gc_var)
        s = calc_stats(m2_var.values, gc_var.values, area_weights.values)
        s["component"] = key
        stats[key] = s
    return stats


annual_stats_all = collect_component_stats(time_mean=True)

for key, meta in COMP_META.items():
    s = annual_stats_all[key]
    print(f"  [{meta['label']:16s}]  area-weighted annual:  R={s['R']:.3f}  "
          f"RMSE={s['RMSE']:.4f}  NMB={s['NMB']:+.1f}%  slope={s['slope']:.3f}")

def format_stats_scatter(stats: dict, rmse_digits: int = 4) -> str:
    return (
        f"N = {stats['N']:,}\n"
        f"R = {stats['R']:.2f}\n"
        f"NMB = {stats['NMB']:+.1f}%\n"
        f"RMSE = {stats['RMSE']:.{rmse_digits}f}"
    )


def format_stats_spatial(stats: dict, rmse_digits: int = 3) -> str:
    return (
        f"N={stats['N']:,}   R={stats['R']:.2f} \n "
        f"NMB={stats['NMB']:+.1f}%   RMSE={stats['RMSE']:.{rmse_digits}f}"
    )


# ══════════════════════════════════════════════════════════════
# Custom density colormap  (white → steel blue → deep navy)
# Perceptually uniform, print-friendly, Nature-style
# ══════════════════════════════════════════════════════════════
_density_colors = [
    "#FFFFFF",  # white  (empty / low)
    "#EAF2FB",  # very light blue
    "#D3E4F3",  # light blue
    "#A8CCE8",  # medium light blue
    "#74ADD1",  # mid blue
    "#3C78B4",  # strong blue
    "#1F4E8C",  # deep blue
    "#0F2F5F",  # navy
]
DENSITY_CMAP = LinearSegmentedColormap.from_list("nature_density", _density_colors, N=256)


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Scatter density plots
# ══════════════════════════════════════════════════════════════
def plot_scatter(save_path="fig_scatter.png"):
    print("\n→ Drawing Figure S3: scatter density plots …")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55))  # Nature double-column width
    axes = axes.ravel()

    for idx, (key, meta) in enumerate(COMP_META.items()):
        ax = axes[idx]
        ax.set_facecolor("white")
        gc_var = ds_gc[meta["gc"]].mean(dim="time").values.ravel()
        m2_var = ds_m2_rg[meta["m2"]].mean(dim="time").values.ravel()

        mask = (np.isfinite(gc_var) & np.isfinite(m2_var)
                & (gc_var >= 0) & (m2_var >= 0))
        o = m2_var[mask]   # observation (x)
        m = gc_var[mask]   # model       (y)

        s = annual_stats_all[key]

        # Axis limits
        vmax = max(np.percentile(o, 99), np.percentile(m, 99))
        vmax = max(vmax, 0.01)
        bins = np.linspace(0, vmax, 36)

        # 2-D histogram
        h, xe, ye = np.histogram2d(o, m, bins=bins)
        h[h == 0] = np.nan

        im = ax.pcolormesh(xe, ye, h.T, cmap=DENSITY_CMAP,
                           norm=LogNorm(vmin=1, vmax=np.nanmax(h)),
                           shading="auto", rasterized=True)

        # 1:1 line
        ax.plot([0, vmax], [0, vmax], color="#8A8A8A", ls="--", lw=0.75)

        # Regression line (forced-zero-intercept)
        xfit = np.array([0, vmax])
        ax.plot(xfit, s["slope"] * xfit, color=meta["color"], lw=0.95,
                zorder=4)

        # Axis formatting
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in", which="both", width=0.5,
                       length=2.5, pad=2, color="#222222")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.minorticks_off()
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

        if idx >= 3:
            ax.set_xlabel("MERRA-2 AOD$_{550}$", labelpad=3)
        else:
            ax.set_xticklabels([])
        if idx % 3 == 0:
            ax.set_ylabel("GC-RT AOD$_{550}$", labelpad=3)
        else:
            ax.set_yticklabels([])

        # Component title
        ax.set_title(f"({PANEL_LABELS[idx]}) {meta['label']}", fontsize=10, fontweight="normal", pad=8)

        # Regression equation annotation
        ax.text(0.7, 0.30, f"y = {s['slope']:.2f}x",
                 transform=ax.transAxes,
                 fontsize=10, color=meta["color"],)

        # Statistics annotation box
        txt = format_stats_scatter(s, rmse_digits=3)
        ax.text(0.03, 0.99, txt, transform=ax.transAxes, fontsize=7, color="#222222",
                ha="left", va="top",
                linespacing=2.0,
                bbox=dict(boxstyle="square,pad=0.20", fc="white",
                          ec="none", alpha=0.90))

    # Shared colour bar at bottom
    cbar_ax = fig.add_axes([0.18, 0.01, 0.64, 0.016])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        extend="max")
    cbar.set_label("Grid-cell counts", fontsize=6.6, labelpad=2)
    cbar.ax.tick_params(labelsize=10, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1,
                        hspace=0.24, wspace=0.20)
    save_ep_figure(fig, Path(__file__).resolve().parents[2] / "figures" / "supplementary" / Path(save_path).stem)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Spatial bias maps (North Polar Stereographic)
# ══════════════════════════════════════════════════════════════
# Nature-style diverging colormap: blue – white – red  (balanced, elegant)
_bias_colors = [
    "#1B3A6B",   # deep navy
    "#3272A8",   # blue
    "#6AAED6",   # sky blue
    "#B9D9EB",   # light blue
    "#FFFFFF",   # white (zero bias)
    "#FADCB8",   # light peach
    "#E8945A",   # salmon/orange
    "#C44E32",   # brick red
    "#7B1A1A",   # deep red
]
BIAS_CMAP = LinearSegmentedColormap.from_list("nature_bias", _bias_colors, N=256)


def setup_polar_axis_nature(ax, lat_min_extent=50):
    """North Polar Stereographic axis with Nature-quality formatting.

    Lat labels at 50, 60, 70, 80; lon labels every 30° around the perimeter.
    """
    ax.set_extent([-180, 180, lat_min_extent, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.35, color="#B8B8B8", alpha=0.7,
                      linestyle=":")
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator([50, 60, 70, 80])

    ax.coastlines(resolution="110m", linewidth=0.45, color="#4A4A4A")
    ax.add_feature(cfeature.LAND, facecolor="#F7F7F7", edgecolor="none",
                   zorder=0)

    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def plot_spatial(save_path="fig_spatial.png"):
    print("\n→ Drawing Figure S4: spatial bias maps …")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.98, top=0.95, bottom=0.16,
                           hspace=0.48, wspace=0.12)

    bias_vmin, bias_vmax = -0.10, 0.10
    bias_norm = mcolors.TwoSlopeNorm(vmin=bias_vmin, vcenter=0.0,
                                     vmax=bias_vmax)
    im = None

    for idx, (key, meta) in enumerate(COMP_META.items()):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col],
                             projection=ccrs.NorthPolarStereo())

        gc_mean  = ds_gc[meta["gc"]].mean(dim="time")
        m2_mean  = ds_m2_rg[meta["m2"]].mean(dim="time")
        bias     = gc_mean - m2_mean

        # Spatial statistics based on annual-mean fields
        spatial_stats = annual_stats_all[key]

        im = ax.pcolormesh(gc_lon, gc_lat, bias.values,
                           cmap=BIAS_CMAP, norm=bias_norm,
                           transform=ccrs.PlateCarree(),
                           rasterized=True)

        setup_polar_axis_nature(ax, lat_min_extent=50)

        # Title
        ax.set_title(f"({PANEL_LABELS[idx]}) {meta['label']}", fontsize=10, fontweight="normal", pad=4)

        # Compact spatial statistics annotation below each map
        ax.text(0.50, -0.025,
                format_stats_spatial(spatial_stats, rmse_digits=3),
                transform=ax.transAxes, fontsize=10.0,
                ha="center", va="top", color="#222222",
                linespacing=1.15)

    # Shared colour bar
    cbar_ax = fig.add_axes([0.20, 0.06, 0.60, 0.018])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        extend="both")
    cbar.set_label("AOD bias (GC-RT - MERRA-2)", fontsize=7)
    cbar.set_ticks(np.linspace(bias_vmin, bias_vmax, 5))
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.4)


    save_ep_figure(fig, Path(__file__).resolve().parents[2] / "figures" / "supplementary" / Path(save_path).stem)
    # fig.savefig(save_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


if __name__ == "__main__":
    plot_scatter("figs3_merra2_aod_scatter.png")
    ds_gc.close()
    ds_m2.close()
