"""Metrics analyzer for lidra evaluation results."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import yaml

import pandas as pd
from omegaconf import DictConfig, OmegaConf


from ..definitions import Report, Threshold, BestOfNSelection, Table
from .base import Base
from ..utils import combine_tables


class TrialMetricsProcessor(Base):
    """Single-run multi-trial metrics processor for lidra csv evaluations."""

    def __init__(self, df: pd.DataFrame, report: Report):
        """
        Initialize processor with data and report configuration.

        Args:
            df: Input metrics DataFrame
            report: Report configuration dict, DictConfig, or ReportConfig
        """
        self.df = df
        self.original_row_count = len(df)  # Store original count for metadata
        self.report = report

        # Validate
        Report.validate(self.report, set(df.columns))

    def generate_run_metadata(
        self,
        mode: str,
        results: Dict[str, pd.DataFrame],
        input_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate metadata for the analysis."""
        metadata = super().generate_run_metadata(mode, results, input_file)
        # Add trials per sample if available
        if "trial" in self.df.columns:
            trials_counts = self.df.groupby("sample_uuid")["trial"].count()
            if not trials_counts.nunique() == 1:
                raise ValueError(
                    f"Inconsistent trial counts across samples. Found counts: {sorted(trials_counts.unique())}"
                )
            metadata["input_data_summary"]["n_trials"] = int(trials_counts.iloc[0])

        return metadata

    def create_report(
        self, mode: str, concat_tables: bool = False
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for table_name in self.report.tables:
            results[table_name] = self.create_table(table_name, mode)

        # Prepare results for output
        if concat_tables:
            results = {"combined": combine_tables(results)}

        return results

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        report: Optional[Union[Dict[str, Any], DictConfig, Report]] = None,
    ) -> "TrialMetricsProcessor":
        """Create TrialMetricsProcessor from CSV file with report config."""
        df = pd.read_csv(filepath)
        cls._validate_df(df)
        return cls(df, report)

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
        """Validate the DataFrame."""
        required_cols = ["sample_uuid", "trial"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if df.empty:
            raise ValueError("Input CSV file is empty")

    def filter_trials(self, max_trials: int) -> None:
        """
        Filter to keep only the first N trials per sample.

        Args:
            max_trials: Maximum number of trials to keep per sample
        """
        if "trial" not in self.df.columns:
            raise ValueError("Cannot filter trials: 'trial' column not found")

        # Keep only trials 1 to max_trials
        self.df = self.df[self.df["trial"] <= max_trials].copy()

        # Verify we still have data
        if self.df.empty:
            raise ValueError(
                f"No data remaining after filtering to first {max_trials} trials"
            )

    def drop_missing_values(
        self,
        drop_all: bool = False,
        columns: Optional[List[str]] = None,
        group_by: str = "sample_uuid",
        drop_incomplete_trials: bool = True,
    ) -> None:
        """Drop rows with missing values and incomplete trials, grouped by specified column."""
        initial_groups = self.df[group_by].nunique()
        initial_rows = len(self.df)
        groups_to_drop = set()

        # First, drop incomplete trials if requested
        if drop_incomplete_trials and "trial" in self.df.columns:
            trial_counts = self.df.groupby(group_by)["trial"].count()
            expected_trials = trial_counts.mode()[0]
            incomplete_groups = trial_counts[trial_counts != expected_trials].index
            groups_to_drop.update(incomplete_groups)
            if len(incomplete_groups) > 0:
                print(
                    f"Found {len(incomplete_groups)} {group_by}s with incomplete trials (expected {expected_trials})"
                )

        # Then handle missing values
        if drop_all or columns:
            if not drop_all and not columns:
                raise ValueError(
                    "Must specify either drop_all or columns to drop missing values from"
                )

            if drop_all:
                columns = self.df.columns
            else:
                missing_cols = [col for col in columns if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

            # Find groups that have missing values in specified columns
            groups_with_na = self.df[self.df[columns].isna().any(axis=1)][
                group_by
            ].unique()
            groups_to_drop.update(groups_with_na)
            if len(groups_with_na) > 0:
                print(f"Found {len(groups_with_na)} {group_by}s with missing values")

        # Remove all rows for groups to drop
        self.df = self.df[~self.df[group_by].isin(groups_to_drop)].copy()

        # Report filtering results
        final_groups = self.df[group_by].nunique()
        final_rows = len(self.df)
        dropped_groups = len(groups_to_drop)
        dropped_rows = initial_rows - final_rows

        if dropped_groups > 0:
            print(f"Dropped {dropped_groups} {group_by}s ({dropped_rows} rows) total")
            print(f"Remaining: {final_groups} {group_by}s ({final_rows} rows)")

        if self.df.empty:
            raise ValueError("No data remaining after dropping rows")

    def drop_incomplete_trials(
        self, expected_trials: Optional[int] = None, group_by: str = "sample_uuid"
    ) -> None:
        """Drop samples that don't have all expected trials."""
        if "trial" not in self.df.columns:
            raise ValueError("Cannot check trials: 'trial' column not found")

        initial_groups = self.df[group_by].nunique()
        initial_rows = len(self.df)

        # Count trials per group
        trial_counts = self.df.groupby(group_by)["trial"].count()

        if expected_trials is None:
            # Use the most common trial count as expected
            expected_trials = trial_counts.mode()[0]
            print(f"Using most common trial count as expected: {expected_trials}")

        # Find groups with incomplete trials
        incomplete_groups = trial_counts[trial_counts != expected_trials].index

        # Remove all rows for incomplete groups
        self.df = self.df[~self.df[group_by].isin(incomplete_groups)].copy()

        # Report filtering results
        final_groups = self.df[group_by].nunique()
        final_rows = len(self.df)
        dropped_groups = len(incomplete_groups)
        dropped_rows = initial_rows - final_rows

        if dropped_groups > 0:
            print(
                f"Dropped {dropped_groups} {group_by}s ({dropped_rows} rows) due to incomplete trials"
            )
            print(f"Remaining: {final_groups} {group_by}s ({final_rows} rows)")

        if self.df.empty:
            raise ValueError("No data remaining after dropping incomplete trials")

    def calculate_overall_metrics(self, columns: List[str]) -> pd.DataFrame:
        """Calculate mean across all samples and trials for specified columns."""
        results = []
        for metric in columns:
            if metric in self.df.columns:
                results.append(
                    {"metric": f"{metric}_mean", "value": self.df[metric].mean()}
                )

        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index("metric", inplace=True)
            df.sort_index(inplace=True)
        return df

    def calculate_best_of_trials_metrics(
        self,
        columns: List[str],
        select_column: str,
        select_max: bool = True,
        thresholds: Optional[Dict[str, Threshold]] = None,
    ) -> pd.DataFrame:
        """Calculate metrics using best trial for each sample based on select_column."""
        results = []

        if select_column not in self.df.columns:
            raise ValueError(f"Select column '{select_column}' not found in DataFrame")

        # Get best trial for each sample based on select_column
        grouped = self.df.groupby("sample_uuid")[select_column]
        if select_max:
            idx = grouped.idxmax()
        else:
            idx = grouped.idxmin()
        best_df = self.df.loc[idx]

        # Calculate mean for each metric
        for metric in columns:
            if metric in best_df.columns:
                results.append({"metric": metric, "value": best_df[metric].mean()})

        # Calculate threshold metrics if provided
        if thresholds:
            for metric, threshold_config in thresholds.items():
                if metric in best_df.columns:
                    for threshold in threshold_config.thresholds:
                        acc_value = self._compute_threshold_metric(
                            best_df[metric],
                            threshold,
                            threshold_config.higher_is_better,
                        )
                        results.append(
                            {"metric": f"{metric}_acc{threshold}", "value": acc_value}
                        )

        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index("metric", inplace=True)
            df.sort_index(inplace=True)
        return df

    @staticmethod
    def _compute_threshold_metric(
        series: pd.Series, threshold: float, higher_is_better: bool = True
    ) -> float:
        if higher_is_better:
            return (series >= threshold).mean()
        else:
            return (series < threshold).mean()

    def create_table(self, table_name: str, mode: str) -> pd.DataFrame:
        """Analyze a single table from the report."""
        if mode not in ["mean", "best"]:
            raise ValueError(f"Mode must be 'mean' or 'best', got '{mode}'")

        if table_name not in self.report.tables:
            raise ValueError(f"Table '{table_name}' not found in report configuration")

        table_config = self.report.tables[table_name]

        if mode == "mean":
            return self.calculate_overall_metrics(table_config.columns)
        elif mode == "best":
            if table_config.best_of_trials is None:
                raise ValueError(
                    f"Table '{table_name}' missing required 'best_of_trials' field"
                )

            bot_config = table_config.best_of_trials
            return self.calculate_best_of_trials_metrics(
                columns=table_config.columns,
                select_column=bot_config.select_column,
                select_max=bot_config.select_max,
                thresholds=table_config.thresholds,
            )

    def _count_available_metrics(self) -> int:
        """Count the number of available metrics in the data."""
        all_metrics = set()
        for table_config in self.report.tables.values():
            all_metrics.update(table_config.columns)
        return len([m for m in all_metrics if m in self.df.columns])


from ..console.rich import RichOutput as RichOutputBase


class RichOutput(RichOutputBase):
    def __init__(self, results: pd.DataFrame, metadata: Dict[str, Any], args):
        super().__init__(results, metadata, args)

    def _print_header(self) -> None:
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Console

        subtitle = Text(self._header_text(), style="white")

        # Create subtitle panel
        panel = Panel(
            subtitle,
            title="",
            border_style="steel_blue",
            box=box.DOUBLE,
            padding=(1, 2),
        )

        console = Console()
        console.print(panel)
        console.print()

    def _print_footer_notes(self, text: Optional[str] = None) -> None:
        """Print data summary using Rich formatting."""
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich.console import Console

        summary = self.metadata.get("input_data_summary", {})

        # Create summary table - Ocean/Arctic theme
        summary_table = Table(
            title="Data Summary",
            title_style="bold light_sky_blue1",
            border_style="steel_blue",
            box=box.ROUNDED,
            show_header=False,
            padding=(0, 1),
        )

        summary_table.add_column("Property", style="light_sky_blue1")
        summary_table.add_column("Value", style="bright_white")

        # Add rows based on available data
        if "original_rows" in summary:
            orig = summary["original_rows"]
            filtered = summary["rows_after_filtering"]
            percentage = (filtered / orig * 100) if orig > 0 else 0
            summary_table.add_row(
                "Total rows", f"{orig} → {filtered} ({percentage:.1f}%)"
            )
        else:
            total_rows = summary.get("total_rows", "N/A")
            summary_table.add_row("Total rows", str(total_rows))

        summary_table.add_row(
            "Unique samples", str(summary.get("unique_samples", "N/A"))
        )

        if "n_trials" in summary:
            summary_table.add_row("Trials per sample", str(summary["n_trials"]))

        summary_table.add_row(
            "Metrics analyzed", str(summary.get("metrics_count", "N/A"))
        )

        console = Console()
        console.print(summary_table)
        console.print()

        super()._print_footer_notes(text)
