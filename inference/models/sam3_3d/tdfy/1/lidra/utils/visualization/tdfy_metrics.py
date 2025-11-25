import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.path import Path
from dataclasses import dataclass, field

from typing import Literal, Dict, Optional, List, Tuple


@dataclass
class CategoryParams:
    name: str
    pretty_name: str
    axis_min: float
    axis_max: float
    category: str
    better_when_lower: bool = False

    def __post_init__(self):
        pass

    def to_pretty_name(self, name: str):
        better_when_lower = " ↓" if self.better_when_lower else " ↑"
        return f"{self.pretty_name}{better_when_lower}"


@dataclass
class TdfyRadarChartParams:

    metrics: Dict[str, CategoryParams] = field(
        default_factory=lambda: {
            # "chamfer_distance": CategoryParams(
            #     name="chamfer_distance",
            #     pretty_name="Chamfer\nDistance",
            #     axis_min=0.0,
            #     axis_max=0.05,
            #     category="Shape",
            #     better_when_lower=True,
            # ),
            "log10_chamfer_distance": CategoryParams(
                name="log10_chamfer_distance",
                pretty_name="log10(Chamfer)",
                axis_min=-6.5,
                axis_max=-1.0,
                category="Shape",
                better_when_lower=True,
            ),
            "f1": CategoryParams(
                name="f1",
                pretty_name="F1",
                axis_min=0.0,
                axis_max=1.0,
                category="Shape",
                better_when_lower=False,
            ),
            # "precision": CategoryParams(
            #     name="precision",
            #     pretty_name="Precision",
            #     axis_min=0.0,
            #     axis_max=1.0,
            #     category="Shape",
            # ),
            # "recall": CategoryParams(
            #     name="recall",
            #     pretty_name="Recall",
            #     axis_min=0.0,
            #     axis_max=1.0,
            #     category="Shape",
            # ),
            "rot_error_deg": CategoryParams(
                name="rot_error_deg",
                pretty_name="Rotation\n(degrees)",
                axis_min=0.0,
                axis_max=90.0,
                category="Pose",
                better_when_lower=True,
            ),
            "trans_abs_rel_error": CategoryParams(
                name="trans_abs_rel_error",
                pretty_name="Translation\n(Abs Rel Err.)",
                axis_min=0.0,
                axis_max=0.5,
                category="Pose",
                better_when_lower=True,
            ),
            "trans_angle_err_deg": CategoryParams(
                name="trans_angle_err_deg",
                pretty_name="Translation\n(Angle Err.)",
                axis_min=0.0,
                axis_max=90.0,
                category="Pose",
                better_when_lower=True,
            ),
            "scale_abs_rel_error": CategoryParams(
                name="scale_abs_rel_error",
                pretty_name="Scale\n(Abs Rel Err.)",
                axis_min=0.0,
                axis_max=0.75,
                category="Pose",
                better_when_lower=True,
            ),
            "psnr": CategoryParams(
                name="psnr",
                pretty_name="PSNR",
                axis_min=26.0,
                axis_max=38.0,
                category="Texture",
            ),
        }
    )

    category_to_color: Dict[str, str] = field(
        default_factory=lambda: {
            "Shape": "#4c78a8",
            "Pose": "#f58518",
            "Texture": "#72b7b2",
        }
    )


DEFAULT_METRIC_COL_NAME = "metric"
DEFAULT_METRIC_CATEGORY_COL_NAME = "metric_type"
DEFAULT_METHOD_COLORS = [
    "#3a6ea5",
    "#e84855",
    "#f9a03f",
    "#5ab203",
    "#8a2be2",
    "#ff69b4",
    "#1e90ff",
    "#32cd32",
]

FAKE_R3DFY_VALUES = {
    "chamfer_distance": 0.18,
    "f1": 0.82,
    "precision": 0.79,
    "recall": 0.85,
    "rotation": 8.5,
    "translation": 0.045,
    "scale": 0.92,
    "photometric_texture": 0.76,
}
FAKE_Q1_VALUES = {
    "chamfer_distance": 0.12,
    "f1": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "rotation": 5.2,
    "translation": 0.032,
    "scale": 0.95,
    "photometric_texture": 0.88,
}


def create_df_from_data(
    data,
    method_name: str = "Custom Method",
    params: Optional[TdfyRadarChartParams] = None,
):
    """
    Create a DataFrame from either a list of float values or a dictionary of metric values.

    Parameters:
    - data: Either a list of floats (in the same order as params.metrics) or
            a dictionary mapping metric names to float values
    - method_name: Name to use for the method column in the DataFrame
    - params: TdfyRadarChartParams object containing metric configurations

    Returns:
    - data_df: Original DataFrame with actual metric values
    - normalized_df: DataFrame with normalized values
    - metric_names: List of metric names
    - metric_categories: List of category groups
    - better_when_lower: List indicating whether lower values are better
    """
    if params is None:
        params = TdfyRadarChartParams()

    # Extract metrics information from params
    metrics = []
    categories = []
    better_when_lower = []

    for metric_key, metric in params.metrics.items():
        metrics.append(metric_key)
        categories.append(metric.category)
        better_when_lower.append(metric.better_when_lower)

    # Convert input data to dictionary format if it's a list
    if isinstance(data, list):
        # Ensure the list has the right length
        if len(data) != len(metrics):
            raise ValueError(f"Expected {len(metrics)} values, got {len(data)}")

        # Convert list to dictionary
        data_dict = {metric: value for metric, value in zip(metrics, data)}
    elif isinstance(data, dict):
        # Verify all keys are valid metrics
        for key in data.keys():
            if key not in metrics:
                raise ValueError(f"Unknown metric: {key}")

        # Use the provided dictionary
        data_dict = data

        # Fill in missing metrics with 0 or another default value
        for metric in metrics:
            if metric not in data_dict:
                data_dict[metric] = 0.0
    else:
        raise TypeError(
            "Data must be either a list of floats or a dictionary mapping metric names to values"
        )

    # Create the data DataFrame
    data_values = [data_dict.get(metric, 0.0) for metric in metrics]

    data_df = pd.DataFrame(
        {
            DEFAULT_METRIC_COL_NAME: metrics,
            DEFAULT_METRIC_CATEGORY_COL_NAME: categories,
            method_name: data_values,
        }
    )

    # Create the normalized DataFrame
    normalized_df = pd.DataFrame()
    normalized_df[DEFAULT_METRIC_COL_NAME] = data_df[DEFAULT_METRIC_COL_NAME]
    normalized_df[DEFAULT_METRIC_CATEGORY_COL_NAME] = data_df[
        DEFAULT_METRIC_CATEGORY_COL_NAME
    ]

    # Normalize the values
    normalized_values = []
    for i, metric_name in enumerate(metrics):
        metric_param = params.metrics[metric_name]
        value = data_values[i]

        # Normalize using the min-max values from the parameter
        min_val = metric_param.axis_min
        max_val = metric_param.axis_max

        # Clamp the value to the specified range
        clamped_value = max(min(value, max_val), min_val)

        # Normalize to [0, 1] range
        if metric_param.better_when_lower:
            # Invert for metrics where lower is better
            normalized = 1 - ((clamped_value - min_val) / (max_val - min_val))
        else:
            normalized = (clamped_value - min_val) / (max_val - min_val)

        normalized_values.append(normalized)

    normalized_df[method_name] = normalized_values

    return data_df, normalized_df


def plot_radar_chart(
    normalized_data,
    metric_col_name: str = DEFAULT_METRIC_COL_NAME,
    metric_category_col_name: str = DEFAULT_METRIC_CATEGORY_COL_NAME,
    methods=None,
    colors=None,
    legend_fontsize=12,
    params: TdfyRadarChartParams = None,
    use_colormap: bool = False,
    colormap: str = "viridis",
    plot_ytick_metric_values: bool = True,
    category_radius: float = 1.1,
    category_text_radius: float = 1.5,
    category_text_angle_offset: float = 0,
    figsize: Tuple[float, float] = (16, 12),
    title: str = "Condensed performance metrics",
):
    """
    Plot a radar chart with customizable methods/lines.

    Parameters:
    - normalized_data: Normalized DataFrame
    - metric_col_name: Column name containing metric names
    - metric_category_col_name: Column name containing metric categories
    - methods: List of method names to plot (columns in the DataFrame)
                If None, plots all columns except 'metric_col_name' and 'metric_category_col_name'
    - colors: Dictionary mapping method names to colors
                If None, generates colors automatically
    - legend_fontsize: Font size for the legend (default: 12)
    - params: TdfyRadarChartParams object containing metric configurations
    - use_colormap: If True, use a colormap gradient for method colors instead of distinct colors
    - colormap: Name of the matplotlib colormap to use when use_colormap is True
    """
    if params is None:
        params = TdfyRadarChartParams()

    metric_names = normalized_data[metric_col_name].unique().tolist()
    metric_categories = normalized_data[metric_category_col_name].tolist()

    # Create angle values (in radians) for each category
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles = angles.copy()  # Create a copy to avoid modifying the original
    # Make the plot circular by appending the first angle to the end
    angles += angles[:1]

    # Determine which methods to plot
    if methods is None:
        methods = [
            col
            for col in normalized_data.columns
            if col not in [metric_col_name, metric_category_col_name]
        ]

    # Set up automatic colors if not provided
    if colors is None:
        if use_colormap:
            # Use a colormap gradient for colors
            cmap = plt.get_cmap(colormap)
            # Generate evenly spaced colors from the colormap
            colormap_colors = [
                cmap(i / (len(methods) - 1) if len(methods) > 1 else 0.5)
                for i in range(len(methods))
            ]
            colors = {method: colormap_colors[i] for i, method in enumerate(methods)}
        else:
            # Default color palette with distinct colors
            colors = {}
            for i, method in enumerate(methods):
                colors[method] = DEFAULT_METHOD_COLORS[i % len(DEFAULT_METHOD_COLORS)]

    # Create a larger figure to accommodate the pushed-out labels
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Plot each method
    for method in methods:
        # Get values and close the loop by adding the first value at the end
        values = normalized_data[method].tolist()

        # Replace NaN values with axis_min for each metric
        for i, metric_name in enumerate(metric_names):
            if pd.isna(values[i]):
                metric_param = params.metrics[metric_name]
                # For normalized data, axis_min corresponds to 0 in the radar chart
                values[i] = 0

        # Close the loop by adding the first value at the end
        values += values[:1]
        color = colors.get(method, "#333333")  # Default to dark gray if not found

        # Plot the line and fill
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Add a grid and set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)  # Lighter grid

    # Add concentric circles with custom styling
    for ytick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        # Make the outermost circle (y=1.0) darker and thicker
        if ytick == 1.0:
            circle = plt.Circle(
                (0, 0),
                ytick,
                transform=ax.transData._b,
                fill=False,
                edgecolor="black",  # Darker color for the outer circle
                alpha=0.6,  # Higher alpha for more visibility
                linestyle="-",
                linewidth=1.5,  # Thicker line for the outer circle
            )
        else:
            circle = plt.Circle(
                (0, 0),
                ytick,
                transform=ax.transData._b,
                fill=False,
                edgecolor="gray",
                alpha=0.5,
                linestyle="-",
                linewidth=0.75,
            )
        ax.add_patch(circle)

    # Add custom ytick labels for each metric
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    if plot_ytick_metric_values:
        ax.set_yticklabels([])  # Hide default ytick labels
        for i, metric_name in enumerate(metric_names):
            metric_param = params.metrics[metric_name]
            min_val = metric_param.axis_min
            max_val = metric_param.axis_max

            # Position the tick labels along each radial axis
            angle = angles[i]

            # For metrics where lower is better, we need to reverse the mapping
            if metric_param.better_when_lower:
                tick_values = [
                    max_val - ytick * (max_val - min_val) for ytick in yticks
                ]
            else:
                tick_values = [
                    min_val + ytick * (max_val - min_val) for ytick in yticks
                ]

            # Only add labels for 0.4, 0.8 to avoid clutter
            for ytick, value in zip(yticks[1::2], tick_values[1::2]):
                # Format the value based on its magnitude
                if abs(value) < 0.01:
                    formatted_value = f"{value:.3f}"
                elif abs(value) < 1:
                    formatted_value = f"{value:.2f}"
                elif abs(value) < 10:
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = f"{int(value)}"

                ax.text(
                    angle,
                    ytick + 0.03,  # Offset slightly for better visibility
                    formatted_value,
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.1",
                        edgecolor="none",
                    ),
                )

    # Set category labels with pretty names from params if available
    pretty_categories = []
    for i, metric_name in enumerate(metric_names):
        # Try to find the metric in params
        metric_param = params.metrics[metric_name]
        pretty_categories.append(metric_param.to_pretty_name(metric_param.name))

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(pretty_categories, fontweight="bold", fontsize=15)
    # Set each tick label with a different color based on its corresponding metric category
    for i, label in enumerate(ax.get_xticklabels()):
        category = metric_categories[i]
        color = params.category_to_color.get(category, "grey")
        # Set alpha to 0.5 if the metric value is nan
        alpha = 0.4 if is_all_nan(normalized_data, metric_names[i], methods) else 1.0
        label.set_alpha(alpha)
        label.set_color(color)

    # Add section labels for Shape, Pose, and Texture groups
    categories_for_plot = metric_categories + [metric_categories[0]]
    add_category_sections(
        ax,
        angles,
        categories_for_plot,
        params.category_to_color,
        category_radius=category_radius,
        category_text_radius=category_text_radius,
        category_text_angle_offset=category_text_angle_offset,
    )

    # Legend
    fig.legend(
        loc="upper right", bbox_to_anchor=(1.1, 1.1), prop={"size": legend_fontsize}
    )
    fig.suptitle(title, size=16, y=1.15)
    # Use tight_layout with rect parameter to reserve space at the bottom for the note
    # The rect format is [left, bottom, right, top] in normalized figure coordinates
    # Add padding on the left side of the figure with adjusted rect parameter
    fig.tight_layout(rect=[0.05, 0.05, 1, 1])
    note = (
        "Note: All metrics normalized so that outer values represent better performance"
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=10)
    # Remove the outer circular border
    ax.spines["polar"].set_visible(False)

    # Move the axis to the right
    # For polar plots, we need to adjust the position differently
    pos = ax.get_position()
    pos_new = [pos.x0 + 0.15, pos.y0, pos.width, pos.height]  # Shift right by 0.15
    ax.set_position(pos_new)

    return fig, ax


def add_category_sections(
    ax,
    angles,
    metric_categories,
    category_colors=None,
    category_radius: float = 1.3,
    category_text_radius: float = 1.75,
    category_text_angle_offset: float = 0,
):
    # Add section labels for Shape, Pose, and Texture groups

    unique_categories = list(set(metric_categories))
    if category_colors is None:
        colors = sns.color_palette("husl", len(unique_categories)).as_hex()
        category_colors = {cat: color for cat, color in zip(unique_categories, colors)}
    category_bounds = {}

    # Find the angle ranges for each category
    for category in unique_categories:
        category_indices = [
            i for i, cat in enumerate(metric_categories[:-1]) if cat == category
        ]
        category_bounds[category] = (min(category_indices), max(category_indices))

    # Add section backgrounds and labels with extended boundaries
    n_categories = len(category_bounds)
    for i, (category, (start_idx, end_idx)) in enumerate(category_bounds.items()):
        # Adjust the indices to extend halfway to the next category
        if start_idx > 0:
            extended_start_idx = start_idx - 0.5
        else:
            # For the first category, extend backward into the last category
            extended_start_idx = -0.5  # Extend before the first tick

        if (
            end_idx < len(angles) - 2
        ):  # -2 because the last angle is a repeat of the first
            extended_end_idx = end_idx + 0.5
        else:
            # For the last category (Texture), wrap around to the beginning
            extended_end_idx = (
                angles[0] - 0.5
            )  # This ensures it connects with the first category

        # Draw sections
        add_category_section(
            ax,
            angles,
            category,
            extended_start_idx,
            extended_end_idx,
            category_colors[category],
            final_category=(i == n_categories - 1),
            category_radius=category_radius,
            text_radius=category_text_radius,
            category_text_angle_offset=category_text_angle_offset,
        )


def add_category_section(
    ax,
    angles,
    category,
    start_idx,
    end_idx,
    color,
    final_category=False,
    text_end_idx=None,
    category_radius=1.3,
    text_radius=1.75,
    border_linewidth=2,
    border_alpha=0.1,
    fill_alpha=0.2,
    category_text_angle_offset=0,
):
    # Calculate angles, handling fractional indices for smoother transitions
    if isinstance(start_idx, int):
        start_angle = angles[start_idx]
    else:
        # For fractional indices, interpolate between adjacent angles
        idx_floor = int(start_idx)
        idx_ceil = idx_floor + 1
        fraction = start_idx - idx_floor
        start_angle = angles[idx_floor] + fraction * (
            angles[min(idx_ceil, len(angles) - 1)] - angles[idx_floor]
        )

    if isinstance(end_idx, int):
        end_angle = angles[end_idx]
    else:
        # For fractional indices, interpolate between adjacent angles
        idx_floor = int(end_idx)
        idx_ceil = idx_floor + 1
        fraction = end_idx - idx_floor
        end_angle = angles[idx_floor] + fraction * (
            angles[min(idx_ceil, len(angles) - 1)] - angles[idx_floor]
        )

    if end_angle < start_angle:
        end_angle += 2 * np.pi

    # Create patch vertices (at radius 1.3 to extend beyond the plot)
    radius = category_radius
    center_angle = (start_angle + end_angle) / 2

    # Draw colored background for entire section
    theta = np.linspace(start_angle, end_angle, 100)
    r_outer = np.ones_like(theta) * radius

    # Fill background from center to beyond the plot area
    ax.fill_between(theta, 0, r_outer, color=color, alpha=fill_alpha)

    # Add a more prominent border for the section
    border_color = color.replace("ff", "99")  # Darker shade of the same color

    ax.plot(
        theta,
        r_outer,
        color=border_color,
        linewidth=border_linewidth,
        alpha=border_alpha,
    )

    # Add radial lines to mark section boundaries
    ax.plot(
        [start_angle, start_angle],
        [0, radius],
        color=border_color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
    )
    ax.plot(
        [end_angle, end_angle],
        [0, radius],
        color=border_color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
    )

    # Add category label at the outer edge with a more prominent style
    ax.text(
        center_angle + category_text_angle_offset,
        text_radius,
        category,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor=border_color,
            boxstyle="round,pad=1.2",
            linewidth=2,
        ),
    )


def generate_data(params: Optional[TdfyRadarChartParams] = None):
    """
    Generate data for the radar chart.

    Parameters:
    - params: TdfyRadarChartParams object containing metric configurations
    """
    if params is None:
        params = TdfyRadarChartParams()

    # Sample data - in a real implementation, this would come from actual measurements
    fake_datas = {
        "R-3Dfy (2024)": FAKE_R3DFY_VALUES,
        "Q1": FAKE_Q1_VALUES,
    }

    fake_data_dfs = []
    for method_name, values in fake_datas.items():
        fake_data_dfs.append(create_df_from_data(values, method_name, params))

    fake_df, fake_norm_df = [list(v) for v in zip(*fake_data_dfs)]

    data = merge_dataframes(fake_df)
    normalized_data = merge_dataframes(fake_norm_df)
    return data, normalized_data


def merge_dataframes(
    dataframes: List[pd.DataFrame],
    on: List[str] = (DEFAULT_METRIC_COL_NAME, DEFAULT_METRIC_CATEGORY_COL_NAME),
):
    dataframe = dataframes[0]
    for i in range(1, len(dataframes)):
        dataframe = pd.merge(dataframe, dataframes[i], on=on, how="inner")
    return dataframe


def is_all_nan(
    df: pd.DataFrame,
    metric: str,
    methods: List[str],
    metric_col: str = DEFAULT_METRIC_COL_NAME,
):
    metric_is_nan = df[df[metric_col] == metric][methods].isna().all()
    if len(methods) > 1:
        metric_is_nan = metric_is_nan.all()
    return metric_is_nan.item()
