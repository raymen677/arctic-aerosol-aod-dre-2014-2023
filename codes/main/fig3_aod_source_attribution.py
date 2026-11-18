"""Figure 3: Hierarchical AOD attribution and component source fractions.

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MPath


ROOT_DIR = Path(__file__).resolve().parents[2] / "data" / "aod"
FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "main"
VAR = "RadAOD550nm_PM"
ARCTIC_LAT_MIN = 50.0

# EP line artwork is exported at 1000 dpi, with a matching vector PDF.
EP_RASTER_DPI = 1000

FONT_STYLE = {
    # EP uses one sans-serif family throughout each figure. Arial is
    # used consistently in the raster and vector exports.
    "family": "Arial",
    "sans_serif": ["Arial"],
    # Keep every visible label at one journal-ready size. Centralizing the
    # setting here makes later global font changes a one-line edit.
    "size_base": 16.4,
    "size_stage": 16.4,
    "size_node": 16.4,
    "size_row": 16.4,
    "size_value": 16.4,
    "weight_emphasis": "semibold",
}

plt.rcParams.update(
    {
        "font.family": FONT_STYLE["family"],
        "font.sans-serif": FONT_STYLE["sans_serif"],
        "font.size": FONT_STYLE["size_base"],
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# Muted source colors and stronger component colors keep the hierarchy legible
# in print while remaining distinguishable for common color-vision deficiencies.
COLORS = {
    "anth": "#B86F7E",
    "natural": "#B9BCB6",
    "local": "#5B86AD",
    "non_local": "#D7A15F",
    "n_europe": "#4E79A7",
    "other_local": "#A9C6DC",
    "s_europe": "#78A66A",
    "e_asia": "#70AAA5",
    "n_america": "#D8B365",
    "other_non_local": "#AAA39A",
    "BC": "#3F3F3F",
    "OC": "#4F8F6B",
    "SU": "#C85C5C",
    "NI": "#D6AA36",
    "DU": "#B8875A",
    "SS": "#5B9BC3",
    "text": "#222222",
    "muted_text": "#666666",
    "guide": "#77736D",
}

SPECIES_FIELDS = {
    "SU": ("RadAOD550nm_SU",),
    "NI": ("RadAOD550nm_NI", "RadAOD550nm_AM"),
    "BC": ("RadAOD550nm_BC",),
    "OC": ("RadAOD550nm_OA",),
    "DU": ("RadAOD550nm_DU",),
    "SS": ("RadAOD550nm_SS",),
}

EXPERIMENT_FILES = {
    "base": "aod_gc_base.nc",
    "noall": "aod_gc_noall.nc",
    "nonp": "aod_gc_nonp.nc",
    "nonp_sup": "aod_gc_nonp_sup.nc",
    "no_neur": "aod_gc_no_north_europe.nc",
    "no_neur_sup": "aod_gc_no_north_europe_sup.nc",
    "no_seur": "aod_gc_no_south_europe.nc",
    "no_nam": "aod_gc_no_north_america.nc",
    "no_eas": "aod_gc_no_east_asia.nc",
    "no_three_sup": "aod_gc_no_three_regions_sup.nc",
}

ANTHROPIC_COMPONENTS = ("BC", "OC", "SU", "NI")
NATURAL_COMPONENTS = ANTHROPIC_COMPONENTS + ("DU", "SS")
PARENT_KEYS = ("local", "non_local")
REGION_PARENTS = {
    "local": ("n_europe", "other_local"),
    "non_local": ("s_europe", "e_asia", "n_america", "other_non_local"),
}
REGION_ORDER = REGION_PARENTS["local"] + REGION_PARENTS["non_local"]
SOURCE_LABELS = {
    "local": "Local",
    "non_local": "Non-Local",
    "n_europe": "N. Europe",
    "other_local": "Other Local",
    "s_europe": "S. Europe",
    "e_asia": "E. Asia",
    "n_america": "N. America",
    "other_non_local": "Other Non-Local",
}
NODE_REGION_LABELS = {
    "n_europe": "N. Eur.",
    "other_local": "Other Local",
    "s_europe": "S. Eur.",
    "e_asia": "E. Asia",
    "n_america": "N. Am.",
    "other_non_local": "Other N-L",
}
BRANCH_LABELS = {
    "anth": "Anthropogenic",
    "natural": "Natural",
    "local": "Local",
    "non_local": "Non-Local",
}
BRANCH_COMPONENTS = {
    "anth": ANTHROPIC_COMPONENTS,
    "natural": NATURAL_COMPONENTS,
    "local": ANTHROPIC_COMPONENTS,
    "non_local": ANTHROPIC_COMPONENTS,
}

# Subfigure b follows the requested reading order: SU and NI on the first row,
# BC and OC on the second row. Each mini-panel contains a source-domain bar
# above a source-region bar.
COMPONENT_PANEL_ORDER = (("SU", "NI"), ("BC", "OC"))

# Subfigure b uses the requested right-to-left source hierarchy. These display
# orders are separate from REGION_ORDER, so the cascade in subfigure a remains
# unchanged while all source colors and numerical shares keep their identities.
COMPONENT_SOURCE_PARENT_ORDER = ("non_local", "local")
COMPONENT_SOURCE_REGION_GROUPS = {
    "non_local": ("other_non_local", "n_america", "e_asia", "s_europe"),
    "local": ("other_local", "n_europe"),
}
COMPONENT_SOURCE_REGION_ORDER = tuple(
    region
    for parent_key in COMPONENT_SOURCE_PARENT_ORDER
    for region in COMPONENT_SOURCE_REGION_GROUPS[parent_key]
)


def species_field(ds: xr.Dataset, species: str) -> xr.DataArray:
    field = ds[SPECIES_FIELDS[species][0]]
    for part in SPECIES_FIELDS[species][1:]:
        field = field + ds[part]
    return field


def arctic_weighted_mean(field: xr.DataArray) -> float:
    subset = field.sel(lat=slice(ARCTIC_LAT_MIN, 90))
    weights = np.cos(np.deg2rad(subset["lat"]))
    weights.name = "weights"
    return float(subset.weighted(weights).mean().values)


def load_experiments() -> dict[str, xr.Dataset]:
    datasets = {}
    for name, filename in EXPERIMENT_FILES.items():
        with xr.open_dataset(ROOT_DIR / filename) as ds:
            datasets[name] = ds.mean(dim="time").load()
    return datasets


def calculate_attribution(datasets: dict[str, xr.Dataset]) -> dict[str, dict[str, float]]:
    """Calculate the parent-normalized total, domain, and regional shares."""
    base = datasets["base"]
    fields = {
        "total": base[VAR],
        "anth": (base - datasets["noall"])[VAR],
        "local": (base - datasets["nonp"])[VAR],
        "non_local": (base - datasets["nonp_sup"])[VAR],
        "n_europe": (base - datasets["no_neur"])[VAR],
        "other_local": (base - datasets["no_neur_sup"])[VAR],
        "s_europe": (base - datasets["no_seur"])[VAR],
        "e_asia": (base - datasets["no_eas"])[VAR],
        "n_america": (base - datasets["no_nam"])[VAR],
        "other_non_local": (base - datasets["no_three_sup"])[VAR],
    }
    weights = {name: arctic_weighted_mean(field) for name, field in fields.items()}

    parent_total = weights["local"] + weights["non_local"]
    local_children_total = weights["n_europe"] + weights["other_local"]
    non_local_children_total = sum(weights[key] for key in REGION_PARENTS["non_local"])
    ratios = {
        "anth_total": weights["anth"] / weights["total"],
        "local": weights["local"] / parent_total,
        "non_local": weights["non_local"] / parent_total,
        "n_europe": weights["n_europe"] / local_children_total,
        "other_local": weights["other_local"] / local_children_total,
        **{
            key: weights[key] / non_local_children_total
            for key in REGION_PARENTS["non_local"]
        },
    }
    return {"weights": weights, "ratios": ratios}


def calculate_component_region_flows(datasets: dict[str, xr.Dataset]) -> dict[str, dict]:
    """Route component AOD through parent-normalized source-region branches."""
    base = datasets["base"]
    removed = {
        "local": datasets["nonp"],
        "non_local": datasets["nonp_sup"],
        "n_europe": datasets["no_neur"],
        "other_local": datasets["no_neur_sup"],
        "s_europe": datasets["no_seur"],
        "e_asia": datasets["no_eas"],
        "n_america": datasets["no_nam"],
        "other_non_local": datasets["no_three_sup"],
    }
    impacts: dict[str, dict[str, float]] = {}
    for component in ANTHROPIC_COMPONENTS:
        base_field = species_field(base, component)
        impacts[component] = {
            source: max(
                arctic_weighted_mean(base_field - species_field(ds, component)),
                0.0,
            )
            for source, ds in removed.items()
        }

    component_totals = {
        component: impacts[component]["local"] + impacts[component]["non_local"]
        for component in ANTHROPIC_COMPONENTS
    }
    flows: dict[str, dict[str, float]] = {
        component: {} for component in ANTHROPIC_COMPONENTS
    }
    for component in ANTHROPIC_COMPONENTS:
        for parent, children in REGION_PARENTS.items():
            child_total = sum(impacts[component][child] for child in children)
            if child_total <= 0:
                raise ValueError(f"{component} {parent} regional impact must be positive.")
            for child in children:
                flows[component][child] = (
                    impacts[component][parent]
                    * impacts[component][child]
                    / child_total
                )

    region_totals = {
        region: sum(flows[component][region] for component in ANTHROPIC_COMPONENTS)
        for region in REGION_ORDER
    }
    compositions = {
        region: {
            component: flows[component][region] / region_totals[region]
            for component in ANTHROPIC_COMPONENTS
        }
        for region in REGION_ORDER
    }
    return {
        "component_totals": component_totals,
        "region_totals": region_totals,
        "flows": flows,
        "compositions": compositions,
    }


def calculate_branch_compositions(
    datasets: dict[str, xr.Dataset],
) -> dict[str, dict[str, float]]:
    """Return 100% component profiles for the four major attribution branches."""
    base = datasets["base"]
    getters = {
        "anth": lambda component: species_field(base, component)
        - species_field(datasets["noall"], component),
        "natural": lambda component: species_field(datasets["noall"], component),
        "local": lambda component: species_field(base, component)
        - species_field(datasets["nonp"], component),
        "non_local": lambda component: species_field(base, component)
        - species_field(datasets["nonp_sup"], component),
    }
    compositions: dict[str, dict[str, float]] = {}
    for branch, components in BRANCH_COMPONENTS.items():
        values = {
            component: max(arctic_weighted_mean(getters[branch](component)), 0.0)
            for component in components
        }
        total = sum(values.values())
        if total <= 0:
            raise ValueError(f"{branch} component total must be positive.")
        compositions[branch] = {
            component: value / total for component, value in values.items()
        }
        if not np.isclose(sum(compositions[branch].values()), 1.0):
            raise RuntimeError(f"{branch} component composition must sum to one.")
    return compositions


def calculate_regional_shares(ratios: dict[str, float]) -> dict[str, float]:
    """Return each region's share of anthropogenic AOD on one conserved scale."""
    shares = {
        region: ratios[parent] * ratios[region]
        for parent, children in REGION_PARENTS.items()
        for region in children
    }
    if not np.isclose(sum(shares.values()), 1.0):
        raise RuntimeError("Regional anthropogenic shares must sum to one.")
    return shares


def print_summary(weights: dict[str, float], ratios: dict[str, float]) -> None:
    regional_shares = calculate_regional_shares(ratios)
    print("=" * 74)
    print("  Arctic (>50 deg N) parent-normalized AOD source attribution")
    print("=" * 74)
    print(f"  Total AOD:                         {weights['total']:.5f}")
    print(f"  Anthropogenic / total AOD:         {ratios['anth_total'] * 100:.1f} %")
    print(f"  Local / anthropogenic AOD:         {ratios['local'] * 100:.1f} %")
    print(f"  Non-Local / anthropogenic AOD:     {ratios['non_local'] * 100:.1f} %")
    print("  Regional shares of anthropogenic AOD:")
    for region in REGION_ORDER:
        print(f"    {SOURCE_LABELS[region]:<20}{regional_shares[region] * 100:6.1f} %")
    print("=" * 74)


def print_regional_component_summary(
    regional_shares: dict[str, float],
    compositions: dict[str, dict[str, float]],
) -> None:
    print("=" * 74)
    print("  Component composition within each anthropogenic source region (%)")
    print("=" * 74)
    print(
        "region".ljust(22)
        + "anth share".rjust(12)
        + "".join(component.rjust(10) for component in ANTHROPIC_COMPONENTS)
    )
    for region in REGION_ORDER:
        print(
            SOURCE_LABELS[region].ljust(22)
            + f"{regional_shares[region] * 100:11.2f} %"
            + "".join(
                f"{compositions[region][component] * 100:10.2f}"
                for component in ANTHROPIC_COMPONENTS
            )
        )
    print("=" * 74)


def print_branch_component_summary(
    compositions: dict[str, dict[str, float]],
) -> None:
    print("=" * 74)
    print("  Component composition within major AOD branches (%)")
    print("=" * 74)
    for branch in ("anth", "natural", "non_local", "local"):
        values = "  ".join(
            f"{component}={compositions[branch][component] * 100:.2f}"
            for component in BRANCH_COMPONENTS[branch]
        )
        print(f"  {BRANCH_LABELS[branch]:<16}{values}")
    print("=" * 74)


def mix_with_white(color: str, amount: float = 0.70) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones(3))


def text_color_for_fill(color: str) -> str:
    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.53 else COLORS["text"]


def add_ribbon(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    left_bottom: float,
    left_top: float,
    right_bottom: float,
    right_top: float,
    color: str,
    *,
    whiten: float,
    alpha: float = 1.0,
) -> None:
    """Draw a smooth, non-crossing alluvial ribbon between two vertical spans."""
    control_1 = x_left + (x_right - x_left) * 0.42
    control_2 = x_left + (x_right - x_left) * 0.58
    path = MPath(
        [
            (x_left, left_top),
            (control_1, left_top),
            (control_2, right_top),
            (x_right, right_top),
            (x_right, right_bottom),
            (control_2, right_bottom),
            (control_1, left_bottom),
            (x_left, left_bottom),
            (x_left, left_top),
        ],
        [
            MPath.MOVETO,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.LINETO,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.CLOSEPOLY,
        ],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor=mix_with_white(color, whiten),
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def add_vertical_ribbon(
    ax: plt.Axes,
    top_left: float,
    top_right: float,
    top_y: float,
    bottom_left: float,
    bottom_right: float,
    bottom_y: float,
    color: str,
    *,
    whiten: float = 0.76,
    alpha: float = 0.90,
) -> None:
    """Connect a parent interval to one child interval in subfigure b.

    The upper interval belongs to the Local/Non-Local bar and the lower one to
    the six-region bar. Curved sides make the parent-child grouping readable
    without using a pie chart.
    """
    vertical_distance = top_y - bottom_y
    control_top = top_y - vertical_distance * 0.42
    control_bottom = bottom_y + vertical_distance * 0.42
    path = MPath(
        [
            (top_left, top_y),
            (top_right, top_y),
            (top_right, control_top),
            (bottom_right, control_bottom),
            (bottom_right, bottom_y),
            (bottom_left, bottom_y),
            (bottom_left, control_bottom),
            (top_left, control_top),
            (top_left, top_y),
        ],
        [
            MPath.MOVETO,
            MPath.LINETO,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.LINETO,
            MPath.CURVE4,
            MPath.CURVE4,
            MPath.CURVE4,
        ],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor=mix_with_white(color, whiten),
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def draw_node(
    ax: plt.Axes,
    x_center: float,
    y_bottom: float,
    y_top: float,
    color: str,
    label: str,
    *,
    width: float,
    font_size: float | None = None,
) -> None:
    ax.add_patch(
        Rectangle(
            (x_center - width / 2, y_bottom),
            width,
            y_top - y_bottom,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(
        x_center,
        (y_bottom + y_top) / 2,
        label,
        ha="center",
        va="center",
        fontsize=font_size or FONT_STYLE["size_node"],
        fontweight=FONT_STYLE["weight_emphasis"],
        color=text_color_for_fill(color),
        linespacing=1.08,
        zorder=5,
    )


def make_spans(
    shares: dict[str, float], order: tuple[str, ...], *, height: float = 100.0
) -> dict[str, tuple[float, float]]:
    spans: dict[str, tuple[float, float]] = {}
    y = 0.0
    for key in order:
        segment_height = shares[key] * height
        spans[key] = (y, y + segment_height)
        y += segment_height
    return spans


def draw_composition_bar(
    ax: plt.Axes,
    profile_label: str,
    context_label: str,
    composition: dict[str, float],
    components: tuple[str, ...],
    *,
    x_left: float,
    y_center: float,
    width: float,
    height: float,
    label_position: str = "top",
    small_label_side: str = "right",
) -> None:
    """Draw one directly labeled, 100% stacked component-composition bar."""
    if label_position not in {"top", "bottom"}:
        raise ValueError(f"Unsupported label position: {label_position}")
    label_y = (
        y_center + height / 2 + 0.72
        if label_position == "top"
        else y_center - height / 2 - 0.72
    )
    ax.text(
        x_left,
        label_y,
        f"{profile_label}  |  {context_label}",
        ha="left",
        va="bottom" if label_position == "top" else "top",
        fontsize=FONT_STYLE["size_row"],
        fontweight=FONT_STYLE["weight_emphasis"],
        color=COLORS["text"],
        zorder=6,
    )

    x = x_left
    small_components: list[str] = []
    for component in components:
        fraction = composition[component]
        segment_width = width * fraction
        ax.add_patch(
            Rectangle(
                (x, y_center - height / 2),
                segment_width,
                height,
                facecolor=COLORS[component],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        )
        if fraction >= 0.115:
            label = f"{component}\n{fraction * 100:.1f} %"
        elif fraction >= 0.070:
            label = f"{fraction * 100:.1f} %"
        else:
            label = ""
            small_components.append(component)
        if label:
            ax.text(
                x + segment_width / 2,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=FONT_STYLE["size_value"],
                fontweight=FONT_STYLE["weight_emphasis"],
                color=text_color_for_fill(COLORS[component]),
                linespacing=0.95,
                zorder=6,
            )
        x += segment_width

    # Tiny slices retain their true width and receive one compact external label.
    if small_components:
        small_label = ", ".join(
            f"{component} {composition[component] * 100:.1f} %"
            for component in small_components
        )
        small_x = x_left + width + 0.16 if small_label_side == "right" else x_left - 0.16
        ax.text(
            small_x,
            y_center,
            small_label,
            ha="left" if small_label_side == "right" else "right",
            va="center",
            fontsize=FONT_STYLE["size_value"],
            color=COLORS["text"],
            zorder=7,
        )


def draw_vertical_composition_profile(
    ax: plt.Axes,
    composition: dict[str, float],
    components: tuple[str, ...],
    *,
    x_left: float,
    y_bottom: float,
    width: float,
    height: float,
) -> None:
    """Draw a branch-normalized component profile beside its hierarchy node."""
    y = y_bottom
    small_segments: list[tuple[str, float, float]] = []
    for component in components:
        fraction = composition[component]
        segment_height = height * fraction
        ax.add_patch(
            Rectangle(
                (x_left, y),
                width,
                segment_height,
                facecolor=COLORS[component],
                edgecolor="white",
                linewidth=0.75,
                zorder=4,
            )
        )
        y_center = y + segment_height / 2
        if segment_height >= 5.2:
            ax.text(
                x_left + width / 2,
                y_center,
                f"{component}\n{fraction * 100:.1f} %",
                ha="center",
                va="center",
                fontsize=FONT_STYLE["size_value"],
                fontweight=FONT_STYLE["weight_emphasis"],
                color=text_color_for_fill(COLORS[component]),
                linespacing=0.92,
                zorder=6,
            )
        else:
            small_segments.append((component, fraction, y_center))
        y += segment_height

    for component, fraction, y_center in small_segments:
        ax.annotate(
            f"{component} {fraction * 100:.1f} %",
            xy=(x_left, y_center),
            xytext=(x_left - 0.18, y_center),
            ha="right",
            va="center",
            fontsize=FONT_STYLE["size_value"],
            color=COLORS["text"],
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.48,
                "color": COLORS[component],
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=7,
        )


# ---------------------------------------------------------------------------
# Layout controls
# ---------------------------------------------------------------------------
# These two distances control the gaps between the three hierarchy columns.
# Lower either value to make that part of the cascade more compact; increase it
# when more room is needed for labels or ribbons.
TOTAL_X = 0.0
TOTAL_TO_PARENT_DISTANCE = 2.7
PARENT_TO_REGION_DISTANCE = 1.8
PARENT_X = TOTAL_X + TOTAL_TO_PARENT_DISTANCE
REGION_X = PARENT_X + PARENT_TO_REGION_DISTANCE

# Node and component-profile dimensions, in the data coordinates of this axes.
NODE_WIDTH = 1.00
PROFILE_WIDTH = 3.70
REGION_PROFILE_HEIGHT = 8.0

# Distance from the center of column 3 to the start of the horizontal regional
# component bars. Defining it as an offset keeps the layout aligned if REGION_X
# is changed above.
REGION_TO_PROFILE_DISTANCE = 1.0
RIGHT_PROFILE_X = REGION_X + REGION_TO_PROFILE_DISTANCE

# Width of the four vertical composition bars and the short connector between a
# composition bar and its hierarchy node.
VERTICAL_PROFILE_WIDTH = 0.74
VERTICAL_PROFILE_GAP = 0.08

# Visible white space between Natural/Anth and between Local/Non-Local component
# bars. This changes presentation only: every bar is still normalized to 100%.
VERTICAL_PROFILE_GROUP_GAP = 3.0
TOTAL_PROFILE_X = (
    TOTAL_X - NODE_WIDTH / 2 - VERTICAL_PROFILE_GAP - VERTICAL_PROFILE_WIDTH
)
PARENT_PROFILE_X = (
    PARENT_X - NODE_WIDTH / 2 - VERTICAL_PROFILE_GAP - VERTICAL_PROFILE_WIDTH
)
REGION_PROFILE_CENTERS = {
    region: center
    for region, center in zip(REGION_ORDER, np.linspace(7.0, 93.0, len(REGION_ORDER)))
}


def add_gap_between_profile_spans(
    spans: dict[str, tuple[float, float]],
    lower_key: str,
    upper_key: str,
    gap: float,
) -> dict[str, tuple[float, float]]:
    """Inset two adjacent profile spans while leaving the hierarchy nodes intact.

    Half of ``gap`` is removed from each side of the shared boundary. The
    returned spans are used only by the component bars and their connectors, so
    the source fractions encoded by the main hierarchy nodes do not change.
    """
    lower_bottom, lower_top = spans[lower_key]
    upper_bottom, upper_top = spans[upper_key]
    half_gap = gap / 2.0

    # Guard against a layout value large enough to erase either profile.
    if gap < 0.0 or half_gap >= min(
        lower_top - lower_bottom,
        upper_top - upper_bottom,
    ):
        raise ValueError(
            f"Profile group gap {gap} is too large for {lower_key}/{upper_key}."
        )

    adjusted = dict(spans)
    adjusted[lower_key] = (lower_bottom, lower_top - half_gap)
    adjusted[upper_key] = (upper_bottom + half_gap, upper_top)
    return adjusted


def calculate_component_source_shares(
    component_data: dict[str, dict],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return source-domain and source-region shares for each AOD component.

    ``calculate_component_region_flows`` already routes every component through
    the Local/Non-Local hierarchy. Dividing those conserved flows by the total
    component impact gives the two 100% bars used in subfigure b.
    """
    attribution: dict[str, dict[str, dict[str, float]]] = {}
    for component in ANTHROPIC_COMPONENTS:
        component_total = component_data["component_totals"][component]
        if component_total <= 0.0:
            raise ValueError(f"{component} source total must be positive.")

        regional = {
            region: component_data["flows"][component][region] / component_total
            for region in REGION_ORDER
        }
        parent = {
            parent_key: sum(regional[region] for region in children)
            for parent_key, children in REGION_PARENTS.items()
        }
        if not np.isclose(sum(parent.values()), 1.0):
            raise RuntimeError(f"{component} parent source shares must sum to one.")
        if not np.isclose(sum(regional.values()), 1.0):
            raise RuntimeError(f"{component} regional source shares must sum to one.")

        attribution[component] = {"parent": parent, "regional": regional}
    return attribution


def draw_component_source_panel(
    ax: plt.Axes,
    component: str,
    attribution: dict[str, dict[str, float]],
) -> None:
    """Draw one two-level, 100% stacked source-attribution panel.

    The first bar shows Local versus Non-Local. The second bar expands those
    domains into the six source regions on the same 0-100% horizontal scale.
    """
    parent = attribution["parent"]
    regional = attribution["regional"]

    # These layout values are deliberately grouped here so the vertical gap and
    # bar thickness can be adjusted without touching the drawing calculations.
    bar_height = 0.58
    parent_y = 1.78
    region_y = 0.54
    parent_bottom = parent_y - bar_height / 2
    region_top = region_y + bar_height / 2
    component_y = (parent_bottom + region_top) / 2

    # Allocate each child inside its parent interval at the upper edge. Because
    # both rows use the same conserved 0-100% scale, the child intervals also
    # align with their positions in the lower source-region bar.
    region_spans: dict[str, tuple[float, float]] = {}
    parent_spans: dict[str, tuple[float, float]] = {}
    x = 0.0
    for parent_key in COMPONENT_SOURCE_PARENT_ORDER:
        parent_left = x
        for region in COMPONENT_SOURCE_REGION_GROUPS[parent_key]:
            region_left = x
            x += regional[region] * 100.0
            region_spans[region] = (region_left, x)
        parent_spans[parent_key] = (parent_left, x)

    # Pale source-colored bands connect every parent domain to its child region.
    for region in COMPONENT_SOURCE_REGION_ORDER:
        left, right = region_spans[region]
        add_vertical_ribbon(
            ax,
            left,
            right,
            parent_bottom,
            left,
            right,
            region_top,
            COLORS[region],
        )

    # Upper row: Local and Non-Local source-domain shares.
    for parent_key in COMPONENT_SOURCE_PARENT_ORDER:
        left, right = parent_spans[parent_key]
        ax.add_patch(
            Rectangle(
                (left, parent_bottom),
                right - left,
                bar_height,
                facecolor=COLORS[parent_key],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        )
        ax.text(
            (left + right) / 2,
            parent_y,
            f"{SOURCE_LABELS[parent_key]}  {parent[parent_key] * 100:.1f} %",
            ha="center",
            va="center",
            fontsize=FONT_STYLE["size_base"],
            fontweight=FONT_STYLE["weight_emphasis"],
            color=text_color_for_fill(COLORS[parent_key]),
            zorder=5,
        )

    # Lower row: names are carried by the shared legend above all four panels,
    # leaving only quantitative labels in these compact bars. Percentages that
    # cannot fit at 15 pt are moved into their pale connector band; this keeps
    # every value legible without shrinking the type or changing the data.
    narrow_label_threshold = 9.0
    for region in COMPONENT_SOURCE_REGION_ORDER:
        left, right = region_spans[region]
        width = right - left
        ax.add_patch(
            Rectangle(
                (left, region_y - bar_height / 2),
                width,
                bar_height,
                facecolor=COLORS[region],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        )
        label_in_connector = width < narrow_label_threshold
        ax.text(
            (left + right) / 2,
            component_y if label_in_connector else region_y,
            f"{regional[region] * 100:.1f} %",
            ha="center",
            va="center",
            fontsize=FONT_STYLE["size_base"],
            fontweight=FONT_STYLE["weight_emphasis"],
            color=(
                COLORS["text"]
                if label_in_connector
                else text_color_for_fill(COLORS[region])
            ),
            zorder=5,
        )

    ax.text(
        50.0,
        component_y,
        component,
        ha="center",
        va="center",
        fontsize=FONT_STYLE["size_base"],
        fontweight=FONT_STYLE["weight_emphasis"],
        color=COLORS[component],
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.18",
            "alpha": 0.94,
        },
        zorder=7,
    )
    # Domain/Region row labels are intentionally omitted; the two-row hierarchy
    # is already encoded by position and connecting bands. Removing their left
    # gutter lets the four panels use the available width more efficiently.
    ax.set_xlim(-0.5, 100.4)
    # Crop close to the lower bar while retaining a small print-safe margin.
    ax.set_ylim(0.22, 2.10)
    ax.axis("off")


def create_figure(
    ratios: dict[str, float],
    component_data: dict[str, dict],
    branch_compositions: dict[str, dict[str, float]],
) -> tuple[Path, Path, Path]:
    height = 100.0
    anth_bottom = (1.0 - ratios["anth_total"]) * height
    parent_spans = make_spans({key: ratios[key] for key in PARENT_KEYS}, PARENT_KEYS)
    regional_shares = calculate_regional_shares(ratios)
    region_spans = make_spans(regional_shares, REGION_ORDER)
    total_spans = {
        "natural": (0.0, anth_bottom),
        "anth": (anth_bottom, height),
    }

    # Main node spans remain contiguous because they encode the actual shares.
    # Only the neighboring component profiles are shortened to create separation.
    total_profile_spans = add_gap_between_profile_spans(
        total_spans,
        "natural",
        "anth",
        VERTICAL_PROFILE_GROUP_GAP,
    )
    parent_profile_spans = add_gap_between_profile_spans(
        parent_spans,
        "local",
        "non_local",
        VERTICAL_PROFILE_GROUP_GAP,
    )

    # The complete journal figure contains the existing cascade as subfigure a
    # and four component-source panels as subfigure b underneath it.
    fig = plt.figure(figsize=(15.6, 10.8))
    outer_grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(7.2, 2.75),
        left=0.018,
        right=0.995,
        top=0.990,
        bottom=0.025,
        hspace=0.050,
    )
    ax = fig.add_subplot(outer_grid[0])
    ax.set_facecolor("white")

    # Main hierarchy: Anthropogenic -> Local/Non-Local -> source regions.
    source_y = anth_bottom
    for parent in PARENT_KEYS:
        source_height = ratios["anth_total"] * ratios[parent] * height
        add_ribbon(
            ax,
            TOTAL_X + NODE_WIDTH / 2,
            PARENT_PROFILE_X,
            source_y,
            source_y + source_height,
            parent_profile_spans[parent][0],
            parent_profile_spans[parent][1],
            COLORS[parent],
            whiten=0.73,
        )
        source_y += source_height

    for parent, children in REGION_PARENTS.items():
        source_y = parent_spans[parent][0]
        for region in children:
            region_bottom, region_top = region_spans[region]
            add_ribbon(
                ax,
                PARENT_X + NODE_WIDTH / 2,
                REGION_X - NODE_WIDTH / 2,
                source_y,
                source_y + (region_top - region_bottom),
                region_bottom,
                region_top,
                COLORS[region],
                whiten=0.58,
            )
            source_y += region_top - region_bottom

    # Column 3 profiles remain to the right and preserve source-region order.
    for region in REGION_ORDER:
        region_bottom, region_top = region_spans[region]
        row_center = REGION_PROFILE_CENTERS[region]
        add_ribbon(
            ax,
            REGION_X + NODE_WIDTH / 2,
            RIGHT_PROFILE_X,
            region_bottom,
            region_top,
            row_center - REGION_PROFILE_HEIGHT / 2,
            row_center + REGION_PROFILE_HEIGHT / 2,
            COLORS[region],
            whiten=0.80,
            alpha=0.92,
        )

    # Four major branch profiles are vertical and sit immediately to the left
    # of their corresponding hierarchy nodes. Use the display-only profile
    # spans here so adjacent profiles have a clear white separator.
    for branch in ("natural", "anth"):
        branch_bottom, branch_top = total_profile_spans[branch]
        draw_vertical_composition_profile(
            ax,
            branch_compositions[branch],
            BRANCH_COMPONENTS[branch],
            x_left=TOTAL_PROFILE_X,
            y_bottom=branch_bottom,
            width=VERTICAL_PROFILE_WIDTH,
            height=branch_top - branch_bottom,
        )
        ax.add_patch(
            Rectangle(
                (TOTAL_PROFILE_X + VERTICAL_PROFILE_WIDTH, branch_bottom),
                VERTICAL_PROFILE_GAP,
                branch_top - branch_bottom,
                facecolor=mix_with_white(COLORS[branch], 0.72),
                edgecolor="none",
                zorder=2,
            )
        )

    for parent in PARENT_KEYS:
        parent_bottom, parent_top = parent_profile_spans[parent]
        draw_vertical_composition_profile(
            ax,
            branch_compositions[parent],
            BRANCH_COMPONENTS[parent],
            x_left=PARENT_PROFILE_X,
            y_bottom=parent_bottom,
            width=VERTICAL_PROFILE_WIDTH,
            height=parent_top - parent_bottom,
        )
        ax.add_patch(
            Rectangle(
                (PARENT_PROFILE_X + VERTICAL_PROFILE_WIDTH, parent_bottom),
                VERTICAL_PROFILE_GAP,
                parent_top - parent_bottom,
                facecolor=mix_with_white(COLORS[parent], 0.72),
                edgecolor="none",
                zorder=2,
            )
        )

    draw_node(
        ax,
        TOTAL_X,
        *total_spans["natural"],
        COLORS["natural"],
        f"Natural\n{(1.0 - ratios['anth_total']) * 100:.1f} %",
        width=NODE_WIDTH,
    )
    draw_node(
        ax,
        TOTAL_X,
        *total_spans["anth"],
        COLORS["anth"],
        f"Anth.\n{ratios['anth_total'] * 100:.1f} %",
        width=NODE_WIDTH,
    )
    for parent in PARENT_KEYS:
        draw_node(
            ax,
            PARENT_X,
            *parent_spans[parent],
            COLORS[parent],
            f"{SOURCE_LABELS[parent]}\n{ratios[parent] * 100:.1f} %",
            width=NODE_WIDTH,
        )
    for region in REGION_ORDER:
        draw_node(
            ax,
            REGION_X,
            *region_spans[region],
            COLORS[region],
            f"{NODE_REGION_LABELS[region]}\n{regional_shares[region] * 100:.1f} %",
            width=NODE_WIDTH,
            font_size=FONT_STYLE["size_node"],
        )

    for region in REGION_ORDER:
        draw_composition_bar(
            ax,
            SOURCE_LABELS[region],
            f"{regional_shares[region] * 100:.1f} % of anthropogenic AOD",
            component_data["compositions"][region],
            ANTHROPIC_COMPONENTS,
            x_left=RIGHT_PROFILE_X,
            y_center=REGION_PROFILE_CENTERS[region],
            width=PROFILE_WIDTH,
            height=REGION_PROFILE_HEIGHT,
        )

    stage_y = 112.0
    # stage_titles = (
    #     (TOTAL_X, "Total Arctic AOD", "share of total"),
    #     (PARENT_X, "Anthropogenic source", "share of anthropogenic AOD"),
    #     (REGION_X, "Source region", "share of anthropogenic AOD"),
    # )
    # for x_pos, title, subtitle in stage_titles:
    #     ax.text(
    #         x_pos,
    #         stage_y,
    #         title,
    #         ha="center",
    #         va="bottom",
    #         fontsize=FONT_STYLE["size_stage"],
    #         fontweight=FONT_STYLE["weight_emphasis"],
    #         color=COLORS["text"],
    #     )
    #     ax.text(
    #         x_pos,
    #         stage_y - 3.0,
    #         subtitle,
    #         ha="center",
    #         va="bottom",
    #         fontsize=7.6,
    #         color=COLORS["muted_text"],
    #     )

    # legend_handles = [
    #     Patch(facecolor=COLORS[component], edgecolor="none", label=component)
    #     for component in NATURAL_COMPONENTS
    # ]
    # ax.legend(
    #     handles=legend_handles,
    #     loc="lower center",
    #     bbox_to_anchor=(PARENT_X, 103.0),
    #     bbox_transform=ax.transData,
    #     ncol=6,
    #     frameon=False,
    #     handlelength=1.2,
    #     handleheight=0.8,
    #     columnspacing=0.95,
    #     fontsize=7.8,
    #     borderaxespad=0,
    # )

    # Crop the canvas close to the actual marks. The small horizontal margins
    # protect the external BC/NI labels and the left-side tiny-slice labels;
    # increase them only if you later add longer annotations.
    LEFT_CANVAS_MARGIN = 0.50
    RIGHT_CANVAS_MARGIN = 0.35
    BOTTOM_CANVAS_MARGIN = 0.6
    TOP_CANVAS_MARGIN = 0.60
    ax.set_xlim(
        TOTAL_PROFILE_X - LEFT_CANVAS_MARGIN,
        RIGHT_PROFILE_X + PROFILE_WIDTH + RIGHT_CANVAS_MARGIN,
    )
    ax.set_ylim(-BOTTOM_CANVAS_MARGIN, height + TOP_CANVAS_MARGIN)
    ax.axis("off")

    # Subfigure b: SU and NI form the first row; BC and OC form the second.
    # Each component uses two horizontally aligned 100% bars connected through
    # the same parent-normalized source hierarchy as Figure 4.
    component_source_shares = calculate_component_source_shares(component_data)
    component_grid = outer_grid[1].subgridspec(
        3,
        2,
        height_ratios=(0.18, 1.0, 1.0),
        wspace=0.045,
        # Separate SU from BC and NI from OC while keeping each two-bar panel
        # visually compact as one unit.
        hspace=0.14,
    )
    component_title_ax = fig.add_subplot(component_grid[0, :])
    component_title_ax.text(
        0.0,
        0.50,
        "(b) Each component AOD by region",
        transform=component_title_ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_STYLE["size_base"],
        fontweight="bold",
        color=COLORS["text"],
    )

    # One compact legend replaces six repeated region names in every lower bar.
    # Its order exactly follows the left-to-right stacking order in each panel.
    region_legend_handles = [
        Patch(
            facecolor=COLORS[region],
            edgecolor="none",
            label=NODE_REGION_LABELS[region],
        )
        for region in COMPONENT_SOURCE_REGION_ORDER
    ]
    component_title_ax.legend(
        handles=region_legend_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.50),
        ncol=6,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
        fontsize=FONT_STYLE["size_base"],
    )
    component_title_ax.axis("off")

    component_axes: list[plt.Axes] = []
    for row_index, component_row in enumerate(COMPONENT_PANEL_ORDER):
        for column_index, component in enumerate(component_row):
            component_ax = fig.add_subplot(
                component_grid[row_index + 1, column_index]
            )
            draw_component_source_panel(
                component_ax,
                component,
                component_source_shares[component],
            )
            component_axes.append(component_ax)

    # Panel labels are combined with concise, sentence-case panel titles.
    # Both titles sit in the outer margin and do not cover quantitative marks.
    ax.text(
        -0.012,
        1.022,
        "(a) Total AOD by region and each component",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_STYLE["size_base"],
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = save_ep_figure(fig, FIG_DIR / "fig3_aod_source_attribution")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    datasets = load_experiments()
    attribution = calculate_attribution(datasets)
    component_data = calculate_component_region_flows(datasets)
    branch_compositions = calculate_branch_compositions(datasets)
    regional_shares = calculate_regional_shares(attribution["ratios"])
    print_summary(attribution["weights"], attribution["ratios"])
    print_branch_component_summary(branch_compositions)
    print_regional_component_summary(
        regional_shares,
        component_data["compositions"],
    )
    png_path, pdf_path = create_figure(
        attribution["ratios"],
        component_data,
        branch_compositions,
    )
    print(f"Saved Figure 3 to {png_path}")
    print(f"Saved Figure 3 to {pdf_path}")


if __name__ == "__main__":
    main()
