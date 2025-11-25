"""Report configuration definitions using dataclasses for type safety."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set


@dataclass
class Threshold:
    thresholds: List[float]
    higher_is_better: bool

    def validate(self, metric_name: str, available_columns: Set[str]) -> None:
        if metric_name not in available_columns:
            raise ValueError(
                f"Threshold metric '{metric_name}' not in available columns"
            )
        if not self.thresholds:
            raise ValueError(f"Threshold '{metric_name}' has empty thresholds list")


@dataclass
class BestOfNSelection:
    select_column: str
    select_max: bool

    def validate(self, table_name: str, available_columns: Set[str]) -> None:
        if self.select_column not in available_columns:
            raise ValueError(
                f"Table '{table_name}' select_column '{self.select_column}' not in available columns"
            )


@dataclass
class Table:
    columns: List[str]
    best_of_trials: Optional[BestOfNSelection] = None
    thresholds: Optional[Dict[str, Threshold]] = None

    def validate(self, table_name: str, available_columns: Set[str]) -> None:
        # Validate columns exist
        missing_cols = set(self.columns) - available_columns
        if missing_cols:
            raise ValueError(
                f"Table '{table_name}' references missing columns: {missing_cols}"
            )

        # Validate best_of_trials if present
        if self.best_of_trials:
            self.best_of_trials.validate(table_name, available_columns)

        # Validate thresholds if present
        if self.thresholds:
            for metric_name, threshold_config in self.thresholds.items():
                threshold_config.validate(metric_name, available_columns)


@dataclass
class Report:
    """Complete report configuration."""

    tables: Dict[str, Table] = field(default_factory=dict)

    def validate(self, available_columns: Set[str]) -> None:
        """Validate entire report configuration."""
        for table_name, table_config in self.tables.items():
            table_config.validate(table_name, available_columns)
