"""Rich table formatter for analysis results."""

from rich.table import Table
from rich.text import Text
from rich import box
import pandas as pd
from typing import List, Optional, Dict, Any, Iterable
from .base import Formatter


class RichTableFormatter(Formatter):
    """Format results as Rich tables with sections and styling.

    This formatter creates visually appealing tables with:
    - Hierarchical metric display (base metrics and thresholds)
    - Direction indicators (↑/↓) for metrics
    - Section separators between metric groups
    - Adaptive column widths
    - Customizable styling
    """

    def __init__(
        self,
        show_section_lines: bool = False,
        show_direction_indicators: bool = True,
        highlight_thresholds: bool = True,
    ):
        """Initialize formatter with display options.

        Args:
            show_section_lines: Whether to show section separator lines
            show_direction_indicators: Whether to show ↑/↓ for metrics
            highlight_thresholds: Whether to use different styling for threshold metrics
        """
        self.show_section_lines = show_section_lines
        self.show_direction_indicators = show_direction_indicators
        self.highlight_thresholds = highlight_thresholds

    def format(self, results: pd.DataFrame, config: Any) -> Table:
        """Format DataFrame as Rich table.

        Args:
            results: DataFrame with metrics as index and statistics as columns
            config: Configuration object with formatting options

        Returns:
            Rich Table object for console rendering
        """
        # Extract display options from config if available
        show_sections = getattr(config, "show_section_lines", self.show_section_lines)
        precision = getattr(config, "format_precision", 6)

        # Sort metrics hierarchically
        sorted_metrics = self._sort_metrics_hierarchically(results.index.tolist())

        # Calculate adaptive column width
        metric_col_width = self._calculate_metric_column_width(sorted_metrics)

        # Create table with styling
        table = self._create_table(results, config, metric_col_width)

        # Populate rows with proper grouping
        self._populate_rows(
            table, results, sorted_metrics, config, precision, show_sections
        )

        return table

    def _create_table(
        self, results: pd.DataFrame, config: Any, metric_col_width: int
    ) -> Table:
        """Create and configure the Rich table.

        Args:
            results: The results DataFrame
            config: Configuration object
            metric_col_width: Calculated width for metric column

        Returns:
            Configured Rich Table
        """
        # Create title based on processor type and configuration
        title = self._create_title(config)

        # Create table with styling
        table = Table(
            title=title,
            title_style="bold white",
            border_style="steel_blue",
            header_style="bold steel_blue",
            # box=box.SIMPLE_HEAD,
            show_lines=False,
            show_edge=True,
        )

        # Add metric column
        table.add_column("Metric", style="bold cyan", width=metric_col_width)

        # Add direction indicator column if enabled
        if self.show_direction_indicators and self._has_metric_directions(config):
            table.add_column("↑/↓", style="bold", width=3, justify="center")

        # Add statistic columns
        for col in results.columns:
            # Capitalize column names for display
            display_name = col.replace("_", " ").title()
            if col.startswith("p") and col[1:].isdigit():
                # Special handling for percentiles (p5, p95, etc)
                display_name = f"P{col[1:]}"
            table.add_column(display_name, justify="right")

        return table

    def _populate_rows(
        self,
        table: Table,
        results: pd.DataFrame,
        sorted_metrics: List[str],
        config: Any,
        precision: int,
        show_sections: bool,
    ) -> None:
        """Populate table with metric rows and proper spacing.

        Args:
            table: The Rich table to populate
            results: Results DataFrame
            sorted_metrics: Hierarchically sorted list of metrics
            config: Configuration object
            precision: Decimal precision for formatting
            show_sections: Whether to add section separators
        """
        previous_base = None

        for metric in sorted_metrics:
            if metric not in results.index:
                continue

            is_threshold = "_at_" in metric
            current_base = metric.rsplit("_at_", 1)[0] if is_threshold else metric

            # Add spacing between different base metrics (like original formatting)
            if previous_base and previous_base != current_base and not is_threshold:
                # Add empty row as separator
                empty_cols = [""] * (len(table.columns) - 1)
                table.add_row("", *empty_cols)

            # Create and add the row
            row = self._create_row(metric, results.loc[metric], config, precision)
            table.add_row(*row)

            previous_base = current_base

    def _create_row(
        self, metric: str, stats: pd.Series, config: Any, precision: int
    ) -> List[Any]:
        """Create a formatted row for a metric.

        Args:
            metric: Metric name
            stats: Series of statistics for this metric
            config: Configuration object
            precision: Decimal precision for formatting

        Returns:
            List of formatted cell values
        """
        is_threshold = "_at_" in metric

        # Format metric name with hierarchy
        display_name = self._format_metric_name(metric)

        # Apply styling based on metric type
        if is_threshold and self.highlight_thresholds:
            metric_text = Text(display_name, style="dim steel_blue")
        else:
            metric_text = Text(display_name, style="cyan")

        # Build row
        row = [metric_text]

        # Add direction indicator if enabled
        if self.show_direction_indicators and self._has_metric_directions(config):
            direction = ""
            if (
                not is_threshold
                and hasattr(config, "metrics")
                and metric in config.metrics
            ):
                direction = self._get_direction_indicator(metric, config)
            row.append(Text(direction, style="bold steel_blue"))

        # Add statistic values
        for stat_name in stats.index:
            value = stats[stat_name]
            if pd.notna(value):
                formatted_value = f"{value:.{precision}f}"
                style = (
                    "dim white"
                    if is_threshold and self.highlight_thresholds
                    else "bold white"
                )

                text = Text(formatted_value, style=style)
            else:
                text = Text("—", style="dim white")
            row.append(text)

        return row

    def _calculate_metric_column_width(self, metric_names: List[str]) -> int:
        """Calculate optimal column width based on metric names.

        Args:
            metric_names: List of all metric names

        Returns:
            Calculated column width
        """
        max_length = 0
        for metric in metric_names:
            display_name = self._format_metric_name(metric)
            max_length = max(max_length, len(display_name))

        # Apply min/max constraints
        return max(25, min(50, max_length + 2))  # +2 for padding

    def _create_title(self, config: Any) -> str:
        """Create an informative table title based on configuration.

        Args:
            config: Configuration object

        Returns:
            Title string
        """
        # Check if this is a multi-trial analysis
        if hasattr(config, "trial_reduction_mode"):
            mode_descriptions = {
                "mean": "average across trials",
                "best_independent": "best trial per metric",
                "best_by_group": "best trial by group",
            }
            mode = config.trial_reduction_mode
            mode_desc = mode_descriptions.get(mode, mode)
            return f"Multi-Trial Metrics Analysis | Mode: {mode} ({mode_desc})"

        # Default title
        return "Metrics Analysis Results"

    def _sort_metrics_hierarchically(self, metrics: Iterable[str]) -> List[str]:
        """Sort metrics with base metrics first, then grouped thresholds.

        Args:
            metrics: Iterable of metric names

        Returns:
            Hierarchically sorted list of metrics
        """
        base_metrics = []
        threshold_metrics = {}

        for metric in metrics:
            if "_at_" in metric:
                # This is a threshold metric
                base, threshold = metric.rsplit("_at_", 1)
                if base not in threshold_metrics:
                    threshold_metrics[base] = []
                try:
                    threshold_value = float(threshold)
                except ValueError:
                    threshold_value = 0.0
                threshold_metrics[base].append((metric, threshold_value))
            else:
                # This is a base metric
                base_metrics.append(metric)

        # Sort base metrics alphabetically
        base_metrics.sort()

        # Build final sorted list
        sorted_metrics = []
        for base in base_metrics:
            sorted_metrics.append(base)
            # Add thresholds for this base metric, sorted by threshold value
            if base in threshold_metrics:
                sorted_thresholds = sorted(threshold_metrics[base], key=lambda x: x[1])
                sorted_metrics.extend([m[0] for m in sorted_thresholds])

        return sorted_metrics

    def _format_metric_name(self, metric: str) -> str:
        """Format metric name with proper indentation for hierarchy.

        Args:
            metric: Metric name

        Returns:
            Formatted display name
        """
        if "_at_" in metric:
            # Indent threshold metrics
            return f"  └─ {metric}"
        return metric

    def _has_metric_directions(self, config: Any) -> bool:
        """Check if configuration includes metric direction information.

        Args:
            config: Configuration object

        Returns:
            True if metrics have direction information
        """
        return hasattr(config, "metrics") and bool(config.metrics)

    def _get_direction_indicator(self, metric: str, config: Any) -> str:
        """Get direction indicator for a metric.

        Args:
            metric: Metric name
            config: Configuration object

        Returns:
            Direction indicator (↑ or ↓)
        """
        if hasattr(config, "metrics") and metric in config.metrics:
            metric_info = config.metrics[metric]
            if hasattr(metric_info, "direction"):
                return "↑" if metric_info.direction == "maximize" else "↓"
        return ""
