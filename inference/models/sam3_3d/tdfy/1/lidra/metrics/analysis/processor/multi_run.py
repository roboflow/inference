"""Multi-run processor for comparing metrics across multiple experiments."""

import tempfile
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import os
import sys

from .base import Base
from ..utils import combine_tables

logger = logging.getLogger(__name__)


class MultiRunProcessor(Base):
    """Aggregates metrics from multiple single-run analyses."""

    def __init__(
        self,
        cli,
        tags: List[Tuple[str, str]],
        report_config: Optional[str] = None,
        transpose: bool = False,
        metrics: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        max_trials: Optional[int] = None,
        design: str = "rich",
        **kwargs,
    ):
        """
        Initialize multi-run processor.

        Args:
            cli: LidraCLI instance for tag resolution
            tags: List of (tag_name, directory_path) tuples
            report_config: Path to report config file
            transpose: Whether to transpose the output
            metrics: Comma-separated list of metrics to include
            sort_by: Metric to sort experiments by
            ascending: Sort in ascending order
            max_trials: Maximum trials per experiment
            design: Console output design
            **kwargs: Additional arguments passed to single-run analysis
        """
        self.cli = cli
        self.tags = tags  # Now contains (tag_name, directory_path) tuples
        self.report_config = (
            report_config or "etc/lidra/reports/tdfy/multi_trial_stage_1_small.yaml"
        )
        self.transpose = transpose
        self.metrics_filter = (
            [m.strip() for m in metrics.split(",")] if metrics else None
        )
        self.sort_by = sort_by
        self.ascending = ascending
        self.max_trials = max_trials
        self.design = design
        self.single_run_kwargs = kwargs

    def process(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main processing pipeline.

        Returns:
            Tuple of (results_df, metadata_dict)
        """
        # Validate tags exist
        self._validate_tags()

        # Run individual analyses
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self._run_single_analyses(Path(temp_dir))
            merged_df = self._collect_and_merge(results)
            if self.metrics_filter:
                merged_df = self._apply_metric_filter(merged_df)

            merged_df = self._apply_sorting(merged_df)
            metadata = self._create_metadata()

            merged_df = {"combined": merged_df}
            return merged_df, metadata

    def _validate_tags(self):
        """Validate that all tag directories exist and contain metrics files."""
        for tag_name, directory_path in self.tags:
            try:
                path = Path(directory_path)
                if not path.exists():
                    raise ValueError(f"Directory does not exist: {directory_path}")

                # Check for metrics files
                # Look for both patterns: *.metrics.*.csv and metrics.csv
                metrics_files = list(path.glob("*.metrics.*.csv"))

                # Also check in metrics_tdfy subdirectory
                metrics_tdfy_dir = path / "metrics_tdfy"
                if metrics_tdfy_dir.exists():
                    metrics_files.extend(list(metrics_tdfy_dir.glob("metrics.csv")))
                    metrics_files.extend(list(metrics_tdfy_dir.glob("*.metrics.*.csv")))

                if not metrics_files:
                    raise ValueError(
                        f"No metrics files found for tag: {tag_name} in {directory_path}"
                    )

            except Exception as e:
                raise ValueError(f"Error validating tag '{tag_name}': {str(e)}")

    def _run_single_analyses(self, temp_dir: Path) -> List[Tuple[str, Path]]:
        """
        Run individual analyses in temporary directories.

        Returns:
            List of (tag, output_path) tuples
        """
        results = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_analysis(tag):
            tag_name, directory_path = tag
            logger.info(f"Running analysis for {tag_name}")

            # Create output directory for this tag
            tag_output = temp_dir / f"{tag_name}_output"
            tag_output.mkdir()

            try:
                # Run single analysis
                self._run_single_analysis(tag_name, directory_path, tag_output)
                return (tag_name, tag_output)
            except Exception as e:
                logger.error(f"Failed to analyze {tag_name}: {str(e)}")
                return None

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_analysis, tag): tag for tag in self.tags}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        if not results:
            raise RuntimeError("No experiments were successfully analyzed")

        return results

    def _run_single_analysis(
        self, tag_name: str, directory_path: str, output_dir: Path
    ):
        """Run single-run analysis for one tag."""
        # Find metrics file in the directory
        path = Path(directory_path)
        metrics_files = list(path.glob("*.metrics.*.csv"))

        # Also check in metrics_tdfy subdirectory
        metrics_tdfy_dir = path / "metrics_tdfy"
        if metrics_tdfy_dir.exists():
            metrics_files.extend(list(metrics_tdfy_dir.glob("metrics.csv")))
            metrics_files.extend(list(metrics_tdfy_dir.glob("*.metrics.*.csv")))

        if not metrics_files:
            raise ValueError(
                f"No metrics files found for {tag_name} in {directory_path}"
            )

        # Prefer metrics.csv over individual metric files
        metrics_csv_files = [f for f in metrics_files if f.name == "metrics.csv"]
        if metrics_csv_files:
            metrics_file = metrics_csv_files[0]
        else:
            # Use the most recent metrics file
            metrics_file = sorted(metrics_files)[-1]

        # Get lidra root
        lidra_root = self.cli._get_lidra_root()
        if not lidra_root:
            raise RuntimeError("Could not find LIDRA root directory")

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "lidra.metrics.analysis",
            "-i",
            str(metrics_file),
            "--format",
            "single",
            "--report-config",
            self.report_config,
            "--output",
            str(output_dir),
            "--single-file",
        ]

        # Add transpose if requested
        if self.transpose:
            cmd.append("--transpose")

        # Add max-trials if specified
        if self.max_trials:
            cmd.extend(["--max-trials", str(self.max_trials)])

        # Add any additional kwargs
        for key, value in self.single_run_kwargs.items():
            if value is not None:
                # Handle boolean flags specially
                if isinstance(value, bool):
                    if value:  # Only add the flag if True
                        cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")

        # Set environment to find lidra module
        env = os.environ.copy()
        env["PYTHONPATH"] = str(lidra_root) + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"Analysis failed for {tag_name}: {result.stderr}")

    def _collect_and_merge(self, results: List[Tuple[str, Path]]) -> pd.DataFrame:
        """Load CSV files and merge into single DataFrame."""
        dfs = []

        for tag_name, output_dir in results:
            combined_path = output_dir / "combined.csv"
            if not combined_path.exists():
                logger.warning(f"No combined.csv found for {tag_name}")
                continue
            df = pd.read_csv(combined_path)
            if "Unnamed: 0" in df.columns:
                df = df.drop("Unnamed: 0", axis=1)
            df.insert(0, "experiment", tag_name)
            dfs.append(df)

        if not dfs:
            raise RuntimeError("No valid results to merge")
        merged = pd.concat(dfs, ignore_index=True)
        return merged

    def _apply_metric_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only include specified metrics."""
        if "metric" not in df.columns:
            logger.warning("No 'metric' column found, skipping metric filter")
            return df

        # Build filter mask
        mask = pd.Series([False] * len(df))

        for metric_pattern in self.metrics_filter:
            if "/" in metric_pattern:
                # Handle table/metric format
                parts = metric_pattern.split("/", 1)
                if len(parts) == 2 and "table" in df.columns:
                    table_pattern, metric_pattern = parts
                    mask |= df["table"].str.contains(table_pattern) & df[
                        "metric"
                    ].str.contains(metric_pattern)
                else:
                    mask |= df["metric"].str.contains(metric_pattern)
            else:
                # Just metric name
                mask |= df["metric"].str.contains(metric_pattern)

        return df[mask]

    def _apply_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort experiments by specified metric."""
        # For sorting, we need to pivot the data temporarily
        # if "experiment" not in df.columns:
        #     return df
        if self.transpose:
            df.sort_values(by=["metric", "table", "experiment"], inplace=True)
        else:
            df.sort_values(by=["table", "metric", "experiment"], inplace=True)

        return df

    def _create_metadata(self) -> Dict[str, Any]:
        """Create metadata for the comparison."""
        return {
            "comparison_info": {
                "num_experiments": len(self.tags),
                "experiments": [tag_name for tag_name, _ in self.tags],
                "experiment_paths": {tag_name: path for tag_name, path in self.tags},
                "report_config": self.report_config,
                "transpose": self.transpose,
                "metrics_filter": self.metrics_filter,
                "sort_by": self.sort_by,
                "ascending": self.ascending,
            },
            "processor": "MultiRunProcessor",
            "version": "1.0",
        }


from ..console.rich import RichOutput as RichOutputBase


class RichOutput(RichOutputBase):
    def __init__(self, results: pd.DataFrame, metadata: Dict[str, Any], args):
        super().__init__(results, metadata, args)

    def _print_header(self) -> None:
        """Print header for comparison format."""
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table

        # Get experiment info from metadata
        comparison_info = self.metadata.get("comparison_info", {})
        experiments = comparison_info.get("experiments", [])

        header_text = f"[bold pale_turquoise1]Experiment Comparison[/]\n"
        header_text += f"[dim steel_blue]Comparing {len(experiments)} experiments[/]"

        header_panel = Panel(
            Text.from_markup(header_text),
            box=box.ROUNDED,
            border_style="steel_blue",
            padding=(1, 2),
        )

        self.console.print(header_panel)
        self.console.print()

        # Print experiment list
        exp_table = Table(
            title="Experiments",
            title_style="bold pale_turquoise1",
            border_style="steel_blue",
            box=box.SIMPLE,
            show_lines=False,
        )

        exp_table.add_column("Index", style="bright_white")
        exp_table.add_column("Experiment Tag", style="light_sky_blue1")

        for i, exp in enumerate(experiments):
            exp_table.add_row(str(i + 1), exp)

        self.console.print(exp_table)
        self.console.print()
