"""Figure S5: CERES surface and TOA shortwave/longwave radiation evaluation.

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

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────── Nature style defaults ────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "figure.dpi":        100,
    "savefig.dpi":       1000,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "xtick.minor.size":  1.5,
    "ytick.minor.size":  1.5,
    "lines.linewidth":   0.8,
    "patch.linewidth":   0.5,
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

# ──────────────────────────── Data paths ───────────────────────────────────────
CERES_FILE = Path(__file__).resolve().parents[2] / "data" / "observations" / "rad_ceres_monthly.nc"
GC_FILE = Path(__file__).resolve().parents[2] / "data" / "radiation" / "rad_gc_L0_base.nc"

LAT_MIN = 50   # PanArctic: >50°N

# ──────────────────────────── Variable pairs ───────────────────────────────────
# (CERES var, GC var, abs_gc, label, is_sw)
VAR_PAIRS = [
    ("AllSkyLWTOA",   "RadAllSkyLWTOA_BASE",  True,  "All-sky LW TOA",        False),
    ("AllSkySWTOA",   "RadAllSkySWTOA_BASE",  True,  "All-sky SW TOA",        True),
    ("AllSkyLWSurf",  "RadAllSkyLWSurf_BASE", False, "All-sky LW Surf",       False),
    ("AllSkySWSurf",  "RadAllSkySWSurf_BASE", False, "All-sky SW Surf",       True),
    ("ClrSkyLWTOA",   "RadClrSkyLWTOA_BASE",  True,  "Clear-sky LW TOA",      False),
    ("ClrSkySWTOA",   "RadClrSkySWTOA_BASE",  True,  "Clear-sky SW TOA",      True),
    ("ClrSkyLWSurf",  "RadClrSkyLWSurf_BASE", False, "Clear-sky LW Surf",     False),
    ("ClrSkySWSurf",  "RadClrSkySWSurf_BASE", False, "Clear-sky SW Surf",     True),
]

# Spatial variable groups (4-element tuples)
ALLSKY_VARS = [
    ("AllSkyLWTOA",  "RadAllSkyLWTOA_BASE",  True,  "LW TOA"),
    ("AllSkySWTOA",  "RadAllSkySWTOA_BASE",  True,  "SW TOA"),
    ("AllSkyLWSurf", "RadAllSkyLWSurf_BASE", False, "LW Surf"),
    ("AllSkySWSurf", "RadAllSkySWSurf_BASE", False, "SW Surf"),
]
CLRSKY_VARS = [
    ("ClrSkyLWTOA",  "RadClrSkyLWTOA_BASE",  True,  "LW TOA"),
    ("ClrSkySWTOA",  "RadClrSkySWTOA_BASE",  True,  "SW TOA"),
    ("ClrSkyLWSurf", "RadClrSkyLWSurf_BASE", False, "LW Surf"),
    ("ClrSkySWSurf", "RadClrSkySWSurf_BASE", False, "SW Surf"),
]

# ──────────────────────────── Nature palettes ──────────────────────────────────
# Month colormap: a perceptually-uniform cyclic palette (twilight-inspired)
MONTH_CMAP = mpl.colormaps["twilight_shifted"]

# Spatial absolute values: a perceptually-uniform sequential map
SPATIAL_CMAP = mpl.colormaps["YlOrRd"]          # warm sequential for fluxes
SPATIAL_DIFF_CMAP = mpl.colormaps["RdBu_r"]     # diverging for bias
SPATIAL_RATIO_CMAP = mpl.colormaps["PuOr_r"]    # diverging for ratio

# Scatter fit-line & 1:1 line colours (Nature-friendly)
COLOR_FIT  = "#D62728"   # muted red
COLOR_11   = "#2C2C2C"   # near-black dashed
COLOR_GRID = "#E0E0E0"

# Panel label style
PANEL_LABELS = [chr(ord("a") + i) for i in range(26)]


def save_figure(fig, stem):
    names = {"fig_scatter": "figs5_ceres_radiation_scatter", "fig_spatial_allsky": "figs6_ceres_all_sky_spatial", "fig_spatial_clearsky": "figs7_ceres_clear_sky_spatial"}
    save_ep_figure(fig, Path(__file__).resolve().parents[2] / "figures" / "supplementary" / names[stem])


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: load and align data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    ceres = xr.open_dataset(CERES_FILE, chunks=None)
    gc    = xr.open_dataset(GC_FILE, chunks=None)

    cp = pd.to_datetime(ceres.time.values).to_period("M")
    gp = pd.to_datetime(gc.time.values).to_period("M")
    common = sorted(set(cp) & set(gp))
    cidx = [np.where(cp == p)[0][0] for p in common]
    gidx = [np.where(gp == p)[0][0] for p in common]
    print(f"Common months: {len(common)} ({common[0]} – {common[-1]})")
    return ceres, gc, common, cidx, gidx


def forced_origin_fit(x, y):
    """OLS through origin: y = kx.  Returns k, R², RMSE, NMB(%), N."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xm, ym = x[mask], y[mask]
    if len(xm) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    k    = np.dot(xm, ym) / np.dot(xm, xm)
    r, _ = stats.pearsonr(xm, ym)
    rmse = np.sqrt(np.mean((ym - k * xm) ** 2))
    nmb  = (ym.mean() - xm.mean()) / xm.mean() * 100.0
    mb   = ym.mean() - xm.mean()
    return k, r ** 2, rmse, nmb, mb, len(xm)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Scatter validation (PanArctic >50 °N)
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter(ceres, gc, common, cidx, gidx):
    lat_min, lat_max = LAT_MIN, 90
    lat_label = f"{lat_min}°–{lat_max}°N"

    # interpolate CERES → GC grid
    ceres_interp = ceres.isel(time=cidx).interp(
        lat=gc.lat.astype("float64"),
        lon=gc.lon.astype("float64"),
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    gc_sub = gc.isel(time=gidx)

    ceres_reg = ceres_interp.sel(lat=slice(lat_min, lat_max))
    gc_reg    = gc_sub.sel(lat=slice(lat_min, lat_max))

    cos_lat = xr.DataArray(
        np.cos(np.deg2rad(gc_reg.lat.values)),
        dims=["lat"], coords={"lat": gc_reg.lat},
    )
    months = np.array([p.month for p in common], dtype=float)

    # area-weighted monthly means
    plot_data = []
    for cvar, gcvar, abs_gc, label, is_sw in VAR_PAIRS:
        ce_da = ceres_reg[cvar]
        gc_da = gc_reg[gcvar]
        if abs_gc:
            gc_da = np.abs(gc_da)
        ce_gl = ce_da.weighted(cos_lat).mean(["lat", "lon"]).values
        gc_gl = gc_da.weighted(cos_lat).mean(["lat", "lon"]).values
        plot_data.append((label, is_sw, ce_gl, gc_gl))

    # ── draw ──
    fig = plt.figure(figsize=(180 / 25.4, 100 / 25.4))  # ~180×100 mm
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 4, figure=fig, hspace=0.65, wspace=0.50,
        left=0.08, right=0.92, top=0.88, bottom=0.10,
    )

    month_norm = mpl.colors.Normalize(vmin=1, vmax=12)

    for idx, (label, is_sw, ce_gl, gc_gl) in enumerate(plot_data):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.set_facecolor("white")

        cx, gy = ce_gl.copy(), gc_gl.copy()
        k, r2, rmse, nmb, mb, n_pts = forced_origin_fit(cx, gy)

        # axis range
        valid = np.isfinite(cx) & np.isfinite(gy) & (cx > 0) & (gy > 0)
        if valid.any():
            vmin = min(cx[valid].min(), gy[valid].min()) * 0.95
            vmax = max(cx[valid].max(), gy[valid].max()) * 1.05
        else:
            vmin, vmax = 0, 100

        # scatter
        sc = ax.scatter(
            cx, gy, c=months, cmap=MONTH_CMAP,
            norm=month_norm, s=14, alpha=0.75,
            edgecolors="none", zorder=4, rasterized=True,
        )

        # 1:1 line
        ax.plot([vmin, vmax], [vmin, vmax], color=COLOR_11,
                lw=0.6, ls="--", zorder=2, label="1 : 1")

        # fit line
        if np.isfinite(k):
            xf = np.array([vmin, vmax])
            ax.plot(xf, k * xf, color=COLOR_FIT, lw=1.0,
                    zorder=5, label=f"$y={k:.3f}x$")

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("CERES (W m$^{-2}$)")
        ax.set_ylabel("GC-RT (W m$^{-2}$)")

        # panel label (bold a–h)
        ax.text(
            -0.18, 1.06, PANEL_LABELS[idx],
            transform=ax.transAxes, fontsize=9, fontweight="bold",
        )
        ax.set_title(label, fontsize=6.5, pad=4)

        # statistics box
        if np.isfinite(k):
            stat_text = (
                f"N = {n_pts}\n"
                f"R = {np.sqrt(r2):.3f}\n"
                f"NMB = {nmb:+.2f}%\n"
                f"RMSE = {rmse:.1f}"
            )
        else:
            stat_text = "N/A"
        ax.text(
            0.04, 0.96, stat_text,
            transform=ax.transAxes, fontsize=5.5, va="top",
            family="monospace", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#BBBBBB", lw=0.4, alpha=0.95),
        )

        # legend
        ax.legend(fontsize=5.5, loc="lower right",
                  frameon=True, framealpha=0.9,
                  edgecolor="#CCCCCC", handlelength=1.5)

        # minor ticks
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.grid(True, which="major", lw=0.25, color=COLOR_GRID, zorder=0)

    # ── month colour-bar ──
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.50])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=month_norm, cmap=MONTH_CMAP),
        cax=cbar_ax,
    )
    cb.set_ticks([1, 3, 6, 9, 12])
    cb.set_ticklabels(["Jan", "Mar", "Jun", "Sep", "Dec"])
    cb.set_label("Month", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)

    save_figure(fig, "fig_scatter")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Spatial maps (PanArctic polar projection)
# ══════════════════════════════════════════════════════════════════════════════
def _polar_boundary():
    """Circular clip path for polar axes."""
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([np.sin(theta), np.cos(theta)])
    return mpath.Path(verts * 0.5 + 0.5)


def _setup_polar_ax(ax, circle):
    """Common polar-axis setup: boundary, coastlines, gridlines."""
    ax.set_extent([-180, 180, LAT_MIN, 90], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(resolution="50m", linewidth=0.3, color="#555555")

    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.25, color="#999999", alpha=0.5, linestyle="--",
    )
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mpl.ticker.FixedLocator([50, 60, 70, 80])


def _add_lat_lon_labels(ax, proj):
    """Manually add latitude / longitude labels on polar plot."""
    # Latitude labels along 0° meridian
    for lat_v in [50, 60, 70, 80]:
        x, y = proj.transform_point(0, lat_v, ccrs.PlateCarree())
        ax.text(
            x, y, f"{lat_v}°",
            fontsize=8.5, ha="center", va="center",
            color="#666666",
            path_effects=[pe.withStroke(linewidth=1.7, foreground="white")],
            zorder=10,
        )
    # Longitude labels at 50°N ring
    for lon_v in np.arange(-180, 180, 30):
        if lon_v == 0:
            continue
        x, y = proj.transform_point(lon_v, LAT_MIN - 2, ccrs.PlateCarree())
        label = f"{abs(lon_v)}°{'E' if lon_v > 0 else 'W' if lon_v < 0 else ''}"
        ax.text(
            x, y, label,
            fontsize=8.8, ha="center", va="center",
            color="#888888",
            path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
            zorder=10,
        )


def _fig_spatial_group(ceres_mean, gc_mean, lat, lon, lat_mask, lat_a, var_group, sky_label, out_stem):
    """Render one 4×3 spatial figure for all-sky or clear-sky diagnostics."""
    proj = ccrs.NorthPolarStereo()
    circle = _polar_boundary()
    n_rows = len(var_group)

    fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        hspace=0.24, wspace=0.08,
        left=0.07, right=0.95, top=0.94, bottom=0.05,
    )
    col_labels = ["GC-RT", "CERES", "GC-RT − CERES"]

    panel_idx = 0
    for i, (cvar, gcvar, abs_gc, row_label) in enumerate(var_group):
        ce = ceres_mean[cvar].values[lat_mask, :].copy()
        gc_d = gc_mean[gcvar].values[lat_mask, :]
        if abs_gc:
            gc_d = np.abs(gc_d)

        # fix lon=-180 NaN column
        nan_col0 = np.isnan(ce[:, 0])
        if nan_col0.any():
            ce[nan_col0, 0] = ce[nan_col0, -1]

        diff = gc_d - ce

        # colour limits
        vmin = min(np.nanpercentile(ce, 2), np.nanpercentile(gc_d, 2))
        vmax = max(np.nanpercentile(ce, 98), np.nanpercentile(gc_d, 98))
        # Scale bias maps by the parent flux magnitude so small residuals
        # are shown in the context of the original GC-RT/CERES fields.
        # dmax = max(abs(vmin), abs(vmax), 1)
        dmax = 50

        panels = [
            (gc_d, SPATIAL_CMAP, vmin, vmax, "W m$^{-2}$"),
            (ce, SPATIAL_CMAP, vmin, vmax, "W m$^{-2}$"),
            (diff, SPATIAL_DIFF_CMAP, -dmax, dmax, "W m$^{-2}$"),
        ]

        for j, (data, cmap, v0, v1, cb_label) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j], projection=proj)
            _setup_polar_ax(ax, circle)

            data_cyc, lon_cyc = add_cyclic_point(data, coord=lon)
            im = ax.pcolormesh(
                lon_cyc, lat_a, data_cyc,
                transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=v0, vmax=v1,
                shading="nearest", rasterized=True,
            )

            cb = fig.colorbar(
                im, ax=ax, orientation="horizontal",
                pad=0.04, shrink=0.84, aspect=24,
            )
            cb.ax.tick_params(labelsize=5.8)
            cb.set_label(cb_label, fontsize=6.2)
            cb.outline.set_linewidth(0.35)

            if i == 0:
                ax.set_title(col_labels[j], fontsize=9, fontweight="bold", pad=8)

            if j == 0:
                ax.text(
                    -0.24, 0.5, f"{sky_label}\n{row_label}",
                    transform=ax.transAxes,
                    fontsize=8.8, fontweight="bold",
                    rotation=90, va="center", ha="center",
                )

            ax.text(
                0.02, 0.96, PANEL_LABELS[panel_idx],
                transform=ax.transAxes,
                fontsize=8.2, fontweight="bold", va="top",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
            )
            panel_idx += 1

            # _add_lat_lon_labels(ax, proj)
    save_figure(fig, out_stem)
    plt.close(fig)


def fig_spatial(ceres, gc, common, cidx, gidx):
    lat = gc.lat.values.astype(float)
    lon = gc.lon.values.astype(float)

    # time-mean interpolated fields
    ceres_mean = ceres.isel(time=cidx).interp(
        lat=gc.lat.astype("float64"),
        lon=gc.lon.astype("float64"),
        method="linear",
        kwargs={"fill_value": np.nan},
    ).mean(dim="time")
    gc_mean = gc.isel(time=gidx).mean(dim="time")

    lat_mask = lat >= LAT_MIN
    lat_a = lat[lat_mask]

    _fig_spatial_group(
        ceres_mean, gc_mean, lat, lon, lat_mask, lat_a,
        ALLSKY_VARS, "All-sky", "fig_spatial_allsky",
    )
    _fig_spatial_group(
        ceres_mean, gc_mean, lat, lon, lat_mask, lat_a,
        CLRSKY_VARS, "Clear-sky", "fig_spatial_clearsky",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def render_ceres_spatial(clear_sky=False):
    ceres, gc, common, cidx, gidx = load_data()
    try:
        lat = gc.lat.values.astype(float)
        lon = gc.lon.values.astype(float)
        ceres_mean = ceres.isel(time=cidx).interp(lat=gc.lat.astype("float64"), lon=gc.lon.astype("float64"), method="linear", kwargs={"fill_value": np.nan}).mean(dim="time")
        gc_mean = gc.isel(time=gidx).mean(dim="time")
        lat_mask = lat >= LAT_MIN
        _fig_spatial_group(ceres_mean, gc_mean, lat, lon, lat_mask, lat[lat_mask], CLRSKY_VARS if clear_sky else ALLSKY_VARS, "Clear-sky" if clear_sky else "All-sky", "fig_spatial_clearsky" if clear_sky else "fig_spatial_allsky")
    finally:
        ceres.close()
        gc.close()

if __name__ == "__main__":
    ceres, gc, common, cidx, gidx = load_data()
    try:
        fig_scatter(ceres, gc, common, cidx, gidx)
    finally:
        ceres.close()
        gc.close()
