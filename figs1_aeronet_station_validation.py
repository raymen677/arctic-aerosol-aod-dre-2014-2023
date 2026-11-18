"""Figure S1: AERONET eight-station AOD evaluation and station map.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
from fig1_region import save_ep_figure, install_offline_geography
install_offline_geography()
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
import pandas as pd
from scipy.stats import pearsonr

# ============================================================
# Centralized figure appearance settings
# ============================================================
FIGURE_SETTINGS = {
    'figure': {
        'figsize': (7.2, 8.2),
        'map_scale': 1.42,
        'wspace': 0.42,
        'hspace': 0.56,
        'left': 0.075,
        'right': 0.975,
        'top': 0.965,
        'bottom': 0.075,
    },
    'rc': {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.5,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9.0,
        'axes.titlesize': 9.0,
        'xtick.labelsize': 8.2,
        'ytick.labelsize': 8.2,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 2.8,
        'ytick.major.size': 2.8,
        'legend.fontsize': 8.8,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '1.0',
        'figure.dpi': 100,
        'savefig.dpi': 1000,
        'savefig.bbox': 'tight',
        'mathtext.default': 'regular',
        'axes.unicode_minus': False,
    },
    'font_sizes': {
        'map_lon_labels': 7.5,
        'map_lat_labels': 7.5,
        'cbar_label': 9.2,
        'cbar_ticks': 8.0,
        'map_panel_label': 11.0,
        'subplot_panel_label': 10.0,
        'subplot_title': 8.8,
        'subplot_stats': 7.8,
        'subplot_ylabel': 9.2,
        'subplot_xlabel': 8.8,
        'subplot_xtick_labels': 7.2,
        'subplot_ytick_labels': 8.0,
        'subplot_legend': 8.8,
        'map_station_id': 9.0,
    },
    'legend': {
        'fontsize': 8.8,
        'markerscale': 1.0,
        'handlelength': 1.8,
        'handletextpad': 0.4,
        'borderpad': 0.25,
        'labelspacing': 0.25,
        'bbox_to_anchor': (0.02, 0.68),
    },
    'lines': {
        'obs_width': 0.95,
        'mod_width': 0.95,
        'obs_marker_size': 0.6,
        'mod_marker_size': 0.6,
        'map_station_size': 1.1,
        'unselected_station_size': 2.2,
        'connection_width': 1.75,
    },
}

plt.rcParams.update(FIGURE_SETTINGS['rc'])
FIGURE_LAYOUT = FIGURE_SETTINGS['figure']
FONT_SIZES = FIGURE_SETTINGS['font_sizes']
LEGEND_SETTINGS = FIGURE_SETTINGS['legend']
LINE_SETTINGS = FIGURE_SETTINGS['lines']

# ---- NPG (Nature Publishing Group) color palette ----
# Ref: ggsci::pal_npg / Nature journals standard
NPG_BLUE    = '#3C5488'   # steel blue
NPG_RED     = '#E64B35'   # vermillion
NPG_GREEN   = '#00A087'   # teal green
NPG_YELLOW  = '#F39B7F'   # salmon/peach
NPG_PURPLE  = '#8491B4'   # lavender gray
NPG_CYAN    = '#4DBBD5'   # sky cyan
NPG_DARK    = '#3C5488'   # navy for text/borders

OBS_COLOR = NPG_BLUE       # AERONET: steel blue
MOD_COLOR = NPG_RED        # GC-RT: vermillion red
OBS_MARKER = 'o'
MOD_MARKER = 's'

# ============================================================
# Load data
# ============================================================
ds_gc = xr.open_dataset(Path(__file__).resolve().parent / "data" / "aod" / "aod_gc_base.nc")
ds_aeronet = xr.open_dataset(Path(__file__).resolve().parent / "data" / "observations" / "aod_aeronet_monthly_v3.nc")
ds_mean_gc = ds_gc.mean(dim='time')

# ============================================================
# Filter pan-Arctic stations (lat > 50°N)
# ============================================================
all_lats = ds_aeronet['latitude'].values
all_lons = ds_aeronet['longitude'].values
all_stations = ds_aeronet['station'].values
arctic_mask = all_lats > 50
arctic_stations = all_stations[arctic_mask]
arctic_lats = all_lats[arctic_mask]
arctic_lons = all_lons[arctic_mask]
print(f"Found {len(arctic_stations)} pan-Arctic stations (Lat > 50°N)")

# ============================================================
# Validate stations: |NMB| <= 40%, >= 8 continuous months
# ============================================================
valid_stations_data = []

for stn in arctic_stations:
    try:
        stn_obs = ds_aeronet["AOD_550nm"].sel(station=stn)
        stn_lat = float(ds_aeronet["latitude"].sel(station=stn))
        stn_lon = float(ds_aeronet["longitude"].sel(station=stn))

        obs_series = stn_obs.to_pandas()
        is_valid = obs_series.notna()
        max_continuous = is_valid.rolling(window=12).sum().max()
        if max_continuous < 8:
            continue

        stn_mod = ds_gc["RadAOD550nm_PM"].interp(lat=stn_lat, lon=stn_lon, method="linear")
        common_time = np.intersect1d(stn_obs.time, stn_mod.time)
        stn_obs_sel = stn_obs.sel(time=common_time).values
        stn_mod_sel = stn_mod.sel(time=common_time).values
        mask = np.isfinite(stn_obs_sel) & np.isfinite(stn_mod_sel)

        if np.sum(mask) < 12:
            continue

        nmb = np.sum(stn_mod_sel[mask] - stn_obs_sel[mask]) / np.sum(stn_obs_sel[mask])

        if abs(nmb) > 0.40:
            continue

        r_corr, p_value = pearsonr(stn_obs_sel[mask], stn_mod_sel[mask])
        if r_corr < 0.8 or p_value >= 0.1:
            continue

        proj_x, proj_y = ccrs.NorthPolarStereo().transform_point(
            stn_lon, stn_lat, ccrs.PlateCarree())
        map_angle = np.degrees(np.arctan2(proj_y, proj_x))

        valid_stations_data.append({
            'name': stn, 'lon': stn_lon, 'lat': stn_lat,
            'nmb': nmb, 'r_corr': r_corr, 'p_value': p_value,
            'map_angle': map_angle,
            'stn_obs': stn_obs, 'stn_mod': stn_mod,
            'n_obs': int(np.sum(mask))
        })
    except Exception:
        continue

print(f"Stations with |NMB|<=40%, R>=0.8, p<0.1, and >=12 months: {len(valid_stations_data)}")

# ============================================================
# Define 8 subplot positions around the central map
# ============================================================
# NorthPolarStereo(central_longitude=0): 180° at top, 0° at bottom,
#   90°W to the left, 90°E to the right.
# Layout:  [1](a) [2](b) [3](c)
#          [4](d)  MAP(e) [5](f)
#          [6](g) [7](h) [8](i)

subplot_configs = [
    # id, grid_pos,  xy_sub (connection point),  panel_label, slot angle around map
    {'id': 1, 'grid': (0, 0), 'xy_sub': (1.0, 0.0), 'label': 'a', 'slot_angle': 135},
    {'id': 2, 'grid': (0, 1), 'xy_sub': (0.5, 0.0), 'label': 'b', 'slot_angle': 90},
    {'id': 3, 'grid': (0, 2), 'xy_sub': (0.0, 0.0), 'label': 'c', 'slot_angle': 45},
    {'id': 4, 'grid': (1, 0), 'xy_sub': (1.0, 0.5), 'label': 'd', 'slot_angle': 180},
    {'id': 5, 'grid': (1, 2), 'xy_sub': (0.0, 0.5), 'label': 'f', 'slot_angle': 0},
    {'id': 6, 'grid': (2, 0), 'xy_sub': (1.0, 1.0), 'label': 'g', 'slot_angle': -135},
    {'id': 7, 'grid': (2, 1), 'xy_sub': (0.5, 1.0), 'label': 'h', 'slot_angle': -90},
    {'id': 8, 'grid': (2, 2), 'xy_sub': (0.0, 1.0), 'label': 'i', 'slot_angle': -45},
]

# ============================================================
# Select best stations without longitude-sector constraints
# ============================================================
valid_stations_data.sort(key=lambda x: (abs(x['nmb']), -x['r_corr'], -x['n_obs']))
selected_candidates = valid_stations_data[:len(subplot_configs)]


def segments_intersect(a1, a2, b1, b2):
    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orientation(a1, a2, b1) * orientation(a1, a2, b2) < 0 and \
        orientation(b1, b2, a1) * orientation(b1, b2, a2) < 0


# Create figure: 3x3 equal-size layout
# ============================================================
fig = plt.figure(figsize=FIGURE_LAYOUT['figsize'])

gs = gridspec.GridSpec(3, 3, wspace=FIGURE_LAYOUT['wspace'], hspace=FIGURE_LAYOUT['hspace'],
                       width_ratios=[1, 1, 1], height_ratios=[1, 1, 1],
                       left=FIGURE_LAYOUT['left'], right=FIGURE_LAYOUT['right'],
                       top=FIGURE_LAYOUT['top'], bottom=FIGURE_LAYOUT['bottom'])

# --- Central map (North Polar Stereographic, CIRCULAR) ---
# Make the map a perfect square with diameter = subplot b's width
_tmp_b = fig.add_subplot(gs[0, 1])
_pos_b = _tmp_b.get_position()
fig.delaxes(_tmp_b)
_tmp_c = fig.add_subplot(gs[1, 1])
_pos_c = _tmp_c.get_position()
fig.delaxes(_tmp_c)
_cell_w = (_pos_b.x1 - _pos_b.x0) * FIGURE_LAYOUT['map_scale']
_fig_w, _fig_h = fig.get_size_inches()
_cell_h = _cell_w * _fig_w / _fig_h
_cx = (_pos_c.x0 + _pos_c.x1) / 2
_cy = (_pos_c.y0 + _pos_c.y1) / 2
ax_map = fig.add_axes([_cx - _cell_w / 2, _cy - _cell_h / 2,
                        _cell_w, _cell_h],
                      projection=ccrs.NorthPolarStereo())
ax_map.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
fig.canvas.draw()

slot_endpoints = []
for cfg in subplot_configs:
    _tmp_ax = fig.add_subplot(gs[cfg['grid'][0], cfg['grid'][1]])
    endpoint = fig.transFigure.inverted().transform(
        _tmp_ax.transAxes.transform(cfg['xy_sub']))
    slot_endpoints.append((cfg, tuple(endpoint)))
    fig.delaxes(_tmp_ax)

station_endpoints = []
for station in selected_candidates:
    proj_x, proj_y = ax_map.projection.transform_point(
        station['lon'], station['lat'], ccrs.PlateCarree())
    endpoint = fig.transFigure.inverted().transform(
        ax_map.transData.transform((proj_x, proj_y)))
    station_endpoints.append((station, tuple(endpoint)))

best_score = None
best_assignment = None
for permutation in itertools.permutations(range(len(selected_candidates))):
    segments = []
    length_sum = 0.0
    for slot_idx, station_idx in enumerate(permutation):
        _, slot_point = slot_endpoints[slot_idx]
        _, station_point = station_endpoints[station_idx]
        segments.append((slot_point, station_point))
        length_sum += np.hypot(slot_point[0] - station_point[0], slot_point[1] - station_point[1])

    crossings = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                crossings += 1

    score = (crossings, length_sum)
    if best_score is None or score < best_score:
        best_score = score
        best_assignment = permutation
        if crossings == 0:
            break

selected_stations = []
for slot_idx, station_idx in enumerate(best_assignment):
    cfg = slot_endpoints[slot_idx][0]
    chosen = selected_candidates[station_idx]
    selected_stations.append({
        **chosen,
        'subplot_id': cfg['id'],
        'grid_pos': cfg['grid'],
        'xy_sub': cfg['xy_sub'],
        'panel_label': cfg['label'],
    })

selected_stations.sort(key=lambda x: x['subplot_id'])
selected_names = {s['name'] for s in selected_stations}
print(f"Connection-line crossings after subplot assignment: {best_score[0]}")

print(f"\nSelected {len(selected_stations)} stations:")
for s in selected_stations:
    r_corr = s['r_corr']
    p_value = s['p_value']
    is_significant = p_value < 0.05
    s['is_significant'] = is_significant
    print(f"  ({s['panel_label']}) [{s['subplot_id']}] {s['name']:30s} "
          f"lat={s['lat']:.1f} lon={s['lon']:7.1f}  NMB={s['nmb']:+.1%}  "
          f"R={r_corr:.2f}  p={p_value:.3g}  p<0.05={is_significant}  N={s['n_obs']}")

# Make the map circular (standard polar-projection technique)
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax_map.set_boundary(circle, transform=ax_map.transAxes)

# Map features — Nature-style with clear boundaries
ax_map.add_feature(cfeature.OCEAN, facecolor='#F0F4F8', zorder=0)
ax_map.add_feature(cfeature.LAND, facecolor='#E8E8E8', edgecolor='#777777',
                   linewidth=0.12, zorder=1)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.18, color='#555555', zorder=6)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.10, linestyle='-', color='#888888', zorder=5)

# Gridlines: lon every 30°, lat at 50/60/70/80°N
import matplotlib.ticker as mticker
gl = ax_map.gridlines(draw_labels=False, linewidth=0.3, color='#888888',
                      alpha=0.45, linestyle='--', zorder=3)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
gl.ylocator = mticker.FixedLocator([50, 60, 70, 80])


# ---- Nature/Cell-style colormap: custom warm sequential ----
# Inspired by Nature Climate Change / Cell Reports palettes
# Smooth white → light gold → warm orange → deep crimson
nature_colors = ['#FFFFFF', '#FFF5E6', '#FFE0B2', '#FFAB66',
                 '#F4793B', '#E64B35', '#B71C1C']
cmap_map = mcolors.LinearSegmentedColormap.from_list(
    'nature_aod', nature_colors, N=256)
mesh = ax_map.pcolormesh(
    ds_mean_gc['lon'], ds_mean_gc['lat'], ds_mean_gc["RadAOD550nm_PM"],
    transform=ccrs.PlateCarree(), cmap=cmap_map, vmin=0, vmax=0.5,
    rasterized=True, zorder=4)

cbar = plt.colorbar(mesh, ax=ax_map, orientation='horizontal',
                    pad=0.06, aspect=25, shrink=0.70, extend='max')
cbar.set_label('GC-RT AOD$_{550}$', fontsize=FONT_SIZES['cbar_label'], fontweight='bold')
cbar.ax.tick_params(labelsize=FONT_SIZES['cbar_ticks'], width=0.5, length=3)
cbar.outline.set_linewidth(0.5)

# Map panel label
ax_map.text(0.02, 0.98, '(e)', transform=ax_map.transAxes, fontsize=FONT_SIZES['map_panel_label'],
            fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))

# --- Plot all unselected Arctic stations as triangles ---
for i in range(len(arctic_lons)):
    stn_name = arctic_stations[i]
    if stn_name not in selected_names:
        ax_map.plot(arctic_lons[i], arctic_lats[i], marker='^',
                    markeredgecolor='#555555', markerfacecolor='#B0B0B0',
                    markersize=LINE_SETTINGS['unselected_station_size'], alpha=0.55, markeredgewidth=0.35,
                    transform=ccrs.PlateCarree(), linestyle='None', zorder=30)

# ============================================================
# Color scale for selected station markers on the map
# ============================================================
cmap_pts = cmap_map  # reuse the Nature-style colormap
norm_pts = mcolors.Normalize(vmin=0, vmax=0.5)

# ============================================================
# Plot 8 time-series subplots and connection lines
# ============================================================

# --- Legend (only add once) ---
legend_drawn = False

for item in selected_stations:
    stn = item['name']
    stn_lat = item['lat']
    stn_lon = item['lon']
    stn_obs = item['stn_obs']
    stn_mod = item['stn_mod']
    nmb = item['nmb']
    subplot_id = item['subplot_id']
    grid_slot = gs[item['grid_pos'][0], item['grid_pos'][1]]
    xy_sub = item['xy_sub']
    panel_label = item['panel_label']

    # --- Compute statistics ---
    common_time = np.intersect1d(stn_obs.time, stn_mod.time)
    stn_obs_sel = stn_obs.sel(time=common_time).values
    stn_mod_sel = stn_mod.sel(time=common_time).values
    mask = np.isfinite(stn_obs_sel) & np.isfinite(stn_mod_sel)

    r_corr = item['r_corr']
    p_value = item['p_value']
    is_significant = item['is_significant']
    rmse = np.sqrt(np.mean((stn_mod_sel[mask] - stn_obs_sel[mask])**2))
    obs_mean_val = np.nanmean(stn_obs_sel[mask])
    n_pts = int(np.sum(mask))
    point_color = cmap_pts(norm_pts(obs_mean_val))

    # --- Create subplot ---
    ax_sub = fig.add_subplot(grid_slot)

    # Time series lines — NPG palette
    ax_sub.fill_between(stn_obs.time.values, 0, stn_obs.values,
                        alpha=0.06, color=OBS_COLOR, zorder=1)
    ax_sub.plot(stn_obs.time, stn_obs, '-', color=OBS_COLOR, linewidth=LINE_SETTINGS['obs_width'],
                alpha=0.9, label='AERONET', zorder=3)
    ax_sub.plot(stn_obs.time, stn_obs, OBS_MARKER, color=OBS_COLOR,
                markersize=LINE_SETTINGS['obs_marker_size'], alpha=0.65, markeredgewidth=0, zorder=4)
    ax_sub.plot(stn_mod.time, stn_mod, '-', color=MOD_COLOR, linewidth=LINE_SETTINGS['mod_width'],
                alpha=0.9, label='GC-RT', zorder=3)
    ax_sub.plot(stn_mod.time, stn_mod, MOD_MARKER, color=MOD_COLOR,
                markersize=LINE_SETTINGS['mod_marker_size'], alpha=0.65, markeredgewidth=0, zorder=4)

    # Panel label
    ax_sub.text(0.02, 0.96, f'({panel_label})', transform=ax_sub.transAxes,
                fontsize=FONT_SIZES['subplot_panel_label'], fontweight='bold', va='top', ha='left')

    # Station title
    ax_sub.set_title(f'{stn}\n({stn_lat:.1f}°N, {stn_lon:.1f}°E)',
                     fontsize=FONT_SIZES['subplot_title'], fontweight='bold', pad=5)

    # Statistics annotation box — NPG-styled
    stats_text = (
                  f'N = {n_pts}\n'
                  f'R = {r_corr:.2f}\n'
                  f'NMB = {nmb:+.1%}\n'
                  f'RMSE = {rmse:.3f}'
                  )
    ax_sub.text(0.97, 0.95, stats_text, transform=ax_sub.transAxes,
                fontsize=FONT_SIZES['subplot_stats'], va='top', ha='right',
                bbox=dict(facecolor='#F8F9FA', alpha=0.92, edgecolor=NPG_PURPLE, 
                          boxstyle='round,pad=0.3', linewidth=0.5))

    # Axis styling — clean Nature aesthetic
    ax_sub.set_ylim(0, 0.6)
    ax_sub.set_ylabel('AOD$_{550}$', fontsize=FONT_SIZES['subplot_ylabel'], labelpad=1)
    ax_sub.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_sub.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_sub.tick_params(axis='x', rotation=0, labelsize=FONT_SIZES['subplot_xtick_labels'], pad=1)
    ax_sub.tick_params(axis='y', labelsize=FONT_SIZES['subplot_ytick_labels'], pad=1)
    ax_sub.grid(True, linestyle=':', linewidth=0.25, alpha=0.35, color='#CCCCCC')
    ax_sub.spines['top'].set_visible(False)
    ax_sub.spines['right'].set_visible(False)
    ax_sub.spines['left'].set_color('#555555')
    ax_sub.spines['bottom'].set_color('#555555')

    # --- Legend (only add once) ---
    if not legend_drawn:
        leg = ax_sub.legend(
            loc='upper left',
            fontsize=LEGEND_SETTINGS['fontsize'],
            frameon=True,
            bbox_to_anchor=LEGEND_SETTINGS['bbox_to_anchor'],
            handlelength=LEGEND_SETTINGS['handlelength'],
            markerscale=LEGEND_SETTINGS['markerscale'],
            handletextpad=LEGEND_SETTINGS['handletextpad'],
            borderpad=LEGEND_SETTINGS['borderpad'],
            labelspacing=LEGEND_SETTINGS['labelspacing'],
        )
        for text in leg.get_texts():
            text.set_fontfamily('Arial')
    legend_drawn = True

    # --- Mark selected station on map (star) ---
    ax_map.plot(stn_lon, stn_lat, marker='*', markeredgecolor='black',
                markerfacecolor=point_color, markersize=LINE_SETTINGS['map_station_size'], markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=55)
    proj_x, proj_y = ax_map.projection.transform_point(
        stn_lon, stn_lat, ccrs.PlateCarree())
    con = ConnectionPatch(
        xyA=xy_sub, coordsA="axes fraction", axesA=ax_sub,
        xyB=(proj_x, proj_y), coordsB="data", axesB=ax_map,
        linestyle="--", linewidth=LINE_SETTINGS['connection_width'], alpha=0.6, color=NPG_PURPLE)
    fig.add_artist(con)


save_ep_figure(fig, Path(__file__).resolve().parent / "figures" / "supplementary" / "figs1_aeronet_station_validation")
plt.close(fig)
ds_gc.close()
ds_aeronet.close()
