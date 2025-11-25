"""CSV formatter for analysis results."""

import pandas as pd
from typing import Any
import io
from .base import Formatter


class CSVFormatter(Formatter):
    """CSV formatter for DataFrame results."""

    def format(self, results: pd.DataFrame, config: Any) -> str:
        """Format DataFrame as CSV string."""
        precision = getattr(config, "format_precision", 6)

        # Copy to avoid modifying original
        formatted_df = results.copy()

        # Format numeric columns with specified precision
        for col in formatted_df.select_dtypes(include=["float64", "float32"]).columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.{precision}f}" if pd.notna(x) else ""
            )

        # Convert to CSV with consistent line endings
        output = io.StringIO()
        formatted_df.to_csv(output, lineterminator="\n")
        return output.getvalue()
