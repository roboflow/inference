"""Rich table formatter for comparison results."""

import pandas as pd
from typing import Any, List
from rich.table import Table
from rich.text import Text
from rich import box

from .base import Formatter
from .rich_table import RichTableFormatter


class ComparisonFormatter(Formatter):
    """Rich table formatter for comparison results."""

    def __init__(self, transpose: bool = False, highlight_thresholds: bool = True):
        self.transpose = transpose
        self.highlight_thresholds = highlight_thresholds
        self.base_formatter = RichTableFormatter(
            show_direction_indicators=False,  # Not applicable for comparisons
            highlight_thresholds=highlight_thresholds,
        )

    def format(self, results: pd.DataFrame, config: Any) -> Table:
        """Format comparison DataFrame as Rich table."""
        # Create header
        title = (
            f"Comparing {config.statistic} across {len(results.columns)} experiments"
        )

        # Create the table
        table = self._create_comparison_table(results, title, config)
        return table

    def _create_comparison_table(
        self, df: pd.DataFrame, title: str, config: Any
    ) -> Table:
        """Create Rich table with comparison-specific styling."""
        # Create table with title
        table = Table(title=title, box=box.ROUNDED, show_header=True, expand=False)

        # Sort metrics hierarchically
        sorted_metrics = self._sort_metrics_hierarchically(df.index)

        # Calculate optimal column width for metric names
        metric_width = self._calculate_metric_column_width(sorted_metrics)

        # Add columns
        index_label = "Experiment" if self.transpose else "Metric"
        table.add_column(
            index_label, style="bold cyan", no_wrap=True, width=metric_width
        )

        for col in df.columns:
            # Use overflow="fold" to wrap long experiment names
            table.add_column(str(col), overflow="fold", min_width=20)

        # Get precision from config or use default
        precision = getattr(config, "format_precision", 6)

        # Add rows with formatting
        previous_base = None
        for metric in sorted_metrics:
            # Add spacing between metric groups
            current_base = metric.split("_at_")[0] if "_at_" in metric else metric
            if previous_base and previous_base != current_base and "_at_" not in metric:
                # Add empty row for visual separation
                table.add_row("", *[""] * len(df.columns))

            # Create formatted row
            row = self._create_row(metric, df.loc[metric], config, precision)
            table.add_row(*row)

            previous_base = current_base

        return table

    def _create_row(
        self, metric: str, values: pd.Series, config: Any, precision: int
    ) -> List[Any]:
        """Create a formatted row for a metric.

        Args:
            metric: Metric name
            values: Series of values for this metric across experiments
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

        # Build row starting with metric name
        row = [metric_text]

        # Add values for each experiment
        for exp in values.index:
            value = values[exp]
            if pd.isna(value) or value == config.show_missing_as:
                text = Text(config.show_missing_as, style="dim white")
            else:
                formatted_value = f"{value:.{precision}f}"
                style = (
                    "dim white"
                    if is_threshold and self.highlight_thresholds
                    else "bold white"
                )
                text = Text(formatted_value, style=style)
            row.append(text)

        return row

    def _sort_metrics_hierarchically(self, metrics) -> List[str]:
        """Sort metrics with base metrics first, then grouped thresholds."""
        return self.base_formatter._sort_metrics_hierarchically(metrics)

    def _format_metric_name(self, metric: str) -> str:
        """Format metric name with proper indentation for hierarchy."""
        return self.base_formatter._format_metric_name(metric)

    def _calculate_metric_column_width(self, metric_names: List[str]) -> int:
        """Calculate optimal column width based on metric names."""
        return self.base_formatter._calculate_metric_column_width(metric_names)
