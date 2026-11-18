"""Figure 1: Anthropogenic emission regions and local/nonlocal masks.

Inputs are resolved relative to this script.
Outputs: matching PNG (1000 dpi) and PDF (190 mm wide) in figures/.
Shared exports and offline map helpers: fig1_region.py.
"""

import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union


ROOT_DIR = Path(__file__).resolve().parent

# Common EP export and offline map functions are kept with Figure 1 so every
# Python file is named after a manuscript or supplementary figure.
from functools import lru_cache
from io import BytesIO
import os
import sys
import xarray as xr
from shapely import wkb

PACKAGE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PACKAGE_DIR / "figures"
EP_WIDTH_MM = 190.0
EP_PNG_DPI = 1000

@lru_cache(maxsize=None)
def offline_geometries(category, name, scale):
    path = PACKAGE_DIR / "data" / "geography" / "natural_earth_geometry.nc"
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        key = f"{scale}_{name}"
        if key not in nc.groups:
            key = f"110m_{name}"
        if key not in nc.groups:
            raise FileNotFoundError(f"Offline Natural Earth geometry missing: {key}")
        values = nc.groups[key].variables["wkb_hex"][:]
        return tuple(wkb.loads(str(value), hex=True) for value in values)

def install_offline_geography():
    import cartopy.feature as feature
    if getattr(feature.NaturalEarthFeature, "_ep_offline", False):
        return
    def geometries(self):
        return iter(offline_geometries(self.category, self.name, self.scale))
    feature.NaturalEarthFeature.geometries = geometries
    feature.NaturalEarthFeature._ep_offline = True
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

def save_ep_figure(fig, stem, *, bbox_inches="tight", pad_inches=0.04):
    """Render original artists at final resolution; never resize a raster.

    PNG: 1000 dpi at approximately 190 mm width, >=7480 pixels.
    PDF: 190 mm width, proportional height, embedded TrueType fonts.
    Set EP_EXPORT_DPI to a smaller value only for a local draft.
    """
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, RectangleObject
    dpi = int(os.environ.get("EP_EXPORT_DPI", EP_PNG_DPI))
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Avoid allocating an oversized preview canvas in legacy source layouts.
    fig.set_dpi(100)
    fig.canvas.draw()
    if bbox_inches == "tight":
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        width = bounds.width + 2 * pad_inches
        # Use a fixed crop in inches for both backends. Recomputing "tight"
        # at the export DPI can change font extents and fall below 7480 px.
        export_bounds = bounds.padded(pad_inches)
    else:
        width = float(fig.get_size_inches()[0])
        export_bounds = bbox_inches
    target_inches = EP_WIDTH_MM / 25.4
    render_dpi = (np.ceil(target_inches * dpi) + 1) / width
    settings = dict(dpi=render_dpi, bbox_inches=export_bounds,
                    pad_inches=0, facecolor="white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", **settings)
    buffer.seek(0)
    reader = PdfReader(buffer)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[0])
    scale = target_inches * 72.0 / float(page.mediabox.width)
    width_points = float(page.mediabox.width) * scale
    height_points = float(page.mediabox.height) * scale
    # A single PDF graphics transform preserves all vector paths and fonts.
    # Wrapping the content avoids parsing every coastline vertex again.
    content = page.get_contents().get_data()
    scaled_content = DecodedStreamObject()
    scaled_content.set_data(
        f"q\n{scale:.12f} 0 0 {scale:.12f} 0 0 cm\n".encode("ascii")
        + content + b"\nQ\n"
    )
    page.replace_contents(scaled_content)
    page.mediabox = RectangleObject([0, 0, width_points, height_points])
    page.cropbox = RectangleObject([0, 0, width_points, height_points])
    page.compress_content_streams()
    writer.add_metadata({"/Title": stem.name, "/Subject": "EP figure; 190 mm wide; fonts embedded"})
    with stem.with_suffix(".pdf").open("wb") as stream:
        writer.write(stream)
    fig.savefig(stem.with_suffix(".png"), format="png", **settings)
    with Image.open(stem.with_suffix(".png")) as image:
        # Re-save metadata losslessly; do not interpolate or alter pixel values.
        image.save(stem.with_suffix(".png"), dpi=(dpi, dpi))
        print(f"EXPORT {stem.name}: {image.width} x {image.height} px, {dpi} dpi", flush=True)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


DEFAULT_OUTPUT_DIR = ROOT_DIR / "figures" / "main"
FIGURE_STEM = "anth_aerosol_retained_regions_map"

LOCAL_LAT_MIN = 50.0
MAP_LON_MIN = -179.5
MAP_LON_MAX = 179.5
MAP_LAT_MIN = -60.0
MAP_LAT_MAX = 90.0
MAP_DISPLAY_LAT_MIN = -35.0
MAP_DISPLAY_LAT_MAX = 80.0

REGION_BOUNDS = {
    "N. Europe": (-15.0, 50.0, 50.0, 71.0),
    "S. Europe": (-15.0, 50.0, 34.0, 50.0),
    "E. Asia": (95.0, 145.0, 14.0, 50.0),
    "N. America": (-125.0, -65.0, 22.0, 50.0),
}

REGION_COLORS = {
    "N. Europe": "#3F678D",
    "Other Local": "#9DBDD9",
    "S. Europe": "#7FA46B",
    "E. Asia": "#4C9A95",
    "N. America": "#D2AA58",
    "Other Trans": "#D4CEC5",
}

REGION_NUMBERS = {
    "N. Europe": 1,
    "Other Local": 2,
    "S. Europe": 3,
    "E. Asia": 4,
    "N. America": 5,
    "Other Trans": 6,
}

REGION_DISPLAY_LABELS = {
    "N. Europe": "Local: N. Europe",
    "Other Local": "Local: Other",
    "S. Europe": "Non-Local: S. Europe",
    "E. Asia": "Non-Local: E. Asia",
    "N. America": "Non-Local: N. America",
    "Other Trans": "Non-Local: Other",
}

REGION_LABEL_POSITIONS = {
    "N. Europe": (17.5, 60.0),
    "Other Local": (0.0, 74.0),
    "S. Europe": (17.5, 38.0),
    "E. Asia": (120.0, 32.0),
    "N. America": (-95.0, 30.0),
    "Other Trans": (0.0, 5.0),
}

TEXT_COLOR = "#222222"
OCEAN_COLOR = "#F8FBFD"
LAND_BACKGROUND = "#F3F0EA"
COASTLINE_COLOR = "#78736C"
GRID_COLOR = "#B8B0A6"

FONT_SIZES = {
    "base": 15,
    "region_label": 11.5,
    "region_number": 15,
    "latitude_label": 15,
}

LABEL_LAYOUT = {
    "region_number_y_offset_points": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the retained anthropogenic aerosol source-region map."
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
        help="Export DPI for PNG and embedded rasters in vector output.",
    )
    parser.add_argument(
        "--resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
        help="Natural Earth land-mask resolution.",
    )
    return parser.parse_args()


def rectangular_region(bounds: tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = bounds
    return box(lon_min, lat_min, lon_max, lat_max)


def load_land_geometries(resolution: str):
    return list(offline_geometries("physical", "land", resolution))


def clipped_land_geometries(mask, land_geometries):
    clipped = []
    for geom in land_geometries:
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.intersects(mask):
            continue
        intersection = geom.intersection(mask)
        if not intersection.is_empty:
            clipped.append(intersection)
    return clipped


def add_land_region(ax, geometries, color: str, *, alpha: float = 0.92, zorder: int = 3):
    if not geometries:
        return
    feature = cfeature.ShapelyFeature(
        geometries,
        crs=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_feature(feature, zorder=zorder)


def plot_region_outline(
    ax,
    bounds: tuple[float, float, float, float],
    color: str,
    *,
    linewidth: float = 1.25,
    linestyle: str = "-",
    zorder: int = 7,
):
    lon_min, lon_max, lat_min, lat_max = bounds
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_joinstyle="round",
        zorder=zorder,
    )


def add_region_label(ax, name: str, lon: float, lat: float):
    ax.annotate(
        str(REGION_NUMBERS[name]),
        xy=(lon, lat),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        xytext=(0, LABEL_LAYOUT["region_number_y_offset_points"]),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=FONT_SIZES["region_number"],
        weight="bold",
        color="white",
        bbox={
            "boxstyle": "circle,pad=0.1",
            "facecolor": REGION_COLORS[name],
            "edgecolor": "white",
            "linewidth": 0.9,
        },
        annotation_clip=False,
        zorder=10,
    )
    ax.text(
        lon,
        lat,
        REGION_DISPLAY_LABELS[name],
        transform=ccrs.PlateCarree(),
        ha="center",
        va="center",
        fontsize=FONT_SIZES["region_label"],
        weight="semibold",
        color=TEXT_COLOR,
        path_effects=[pe.withStroke(linewidth=3.5, foreground="white", alpha=0.95)],
        zorder=9,
    )


def create_region_masks():
    named_masks = {name: rectangular_region(bounds) for name, bounds in REGION_BOUNDS.items()}

    local_mask = box(-180.0, LOCAL_LAT_MIN, 180.0, MAP_LAT_MAX)
    trans_mask = box(-180.0, MAP_LAT_MIN, 180.0, LOCAL_LAT_MIN)

    other_local = local_mask.difference(named_masks["N. Europe"])
    named_trans = unary_union(
        [named_masks["S. Europe"], named_masks["E. Asia"], named_masks["N. America"]]
    )
    other_trans = trans_mask.difference(named_trans)

    return {
        "Other Trans": other_trans,
        "Other Local": other_local,
        "N. Europe": named_masks["N. Europe"],
        "S. Europe": named_masks["S. Europe"],
        "E. Asia": named_masks["E. Asia"],
        "N. America": named_masks["N. America"],
    }


def create_figure(output_dir: Path, dpi: int, resolution: str) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_SIZES["base"],
            "axes.linewidth": 0.6,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    projection = ccrs.Mercator(
        central_longitude=0,
        min_latitude=MAP_DISPLAY_LAT_MIN,
        max_latitude=MAP_DISPLAY_LAT_MAX,
    )
    fig = plt.figure(figsize=(11.8, 5.6))
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.92], projection=projection)
    ax.set_extent(
        [MAP_LON_MIN, MAP_LON_MAX, MAP_DISPLAY_LAT_MIN, MAP_DISPLAY_LAT_MAX],
        crs=ccrs.PlateCarree(),
    )
    ax.set_facecolor(OCEAN_COLOR)

    ax.add_feature(
        cfeature.LAND.with_scale(resolution),
        facecolor=LAND_BACKGROUND,
        edgecolor="none",
        zorder=1,
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale(resolution),
        facecolor=OCEAN_COLOR,
        edgecolor="none",
        zorder=0,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        color=GRID_COLOR,
        alpha=0.45,
        linestyle="--",
        zorder=2,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-30, 81, 20))

    masks = create_region_masks()
    land_geometries = load_land_geometries(resolution)

    draw_order = ["Other Trans", "Other Local", "S. Europe", "E. Asia", "N. America", "N. Europe"]
    for name in draw_order:
        clipped = clipped_land_geometries(masks[name], land_geometries)
        add_land_region(
            ax,
            clipped,
            REGION_COLORS[name],
            alpha=0.88 if name.startswith("Other") else 0.96,
            zorder=3 if name.startswith("Other") else 4,
        )

    ax.coastlines(resolution=resolution, linewidth=0.55, color=COASTLINE_COLOR, zorder=6)

    boundary_lons = np.linspace(-180, 180, 361)
    ax.plot(
        boundary_lons,
        np.full_like(boundary_lons, LOCAL_LAT_MIN),
        transform=ccrs.PlateCarree(),
        color="#49433D",
        linewidth=1.05,
        linestyle=(0, (5, 3)),
        zorder=7,
    )
    ax.text(
        -163,
        LOCAL_LAT_MIN + 2.5,
        r"50$^\circ$N",
        transform=ccrs.PlateCarree(),
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["latitude_label"],
        weight="semibold",
        color="#49433D",
        path_effects=[pe.withStroke(linewidth=3.2, foreground="white", alpha=0.95)],
        zorder=9,
    )

    for name, bounds in REGION_BOUNDS.items():
        plot_region_outline(ax, bounds, REGION_COLORS[name])

    for name, (lon, lat) in REGION_LABEL_POSITIONS.items():
        add_region_label(ax, name, lon, lat)

    png_path, svg_path = save_ep_figure(fig, output_dir / "fig1_region")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    install_offline_geography()
    args = parse_args()
    png_path, svg_path = create_figure(args.output_dir, args.dpi, args.resolution)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
