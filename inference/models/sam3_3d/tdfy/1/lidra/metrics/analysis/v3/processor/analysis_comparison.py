"""AnalysisComparisonProcessor - Always runs analyses before comparing results."""

from contextlib import contextmanager
from loguru import logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..formatting.comparison import ComparisonFormatter
from .base import BaseConfig, Processor


@dataclass
class AnalysisComparisonConfig(BaseConfig):
    """Configuration for running and comparing analyses."""

    # Override base input_file to make it optional
    input_file: str = ""  # Not used by AnalysisComparisonProcessor

    # Core inputs
    experiments: List[str] = field(default_factory=list)
    experiment_names: Optional[List[str]] = None

    # Analysis settings
    analysis_config: str = "stage1"
    overwrites: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Comparison settings
    statistic: str = "mean"
    save_dir: Optional[str] = None

    # Execution
    parallel_analyses: bool = False
    max_workers: int = 4

    # Display options
    metrics_filter: Optional[List[str]] = None
    sort_by: Optional[str] = None
    ascending: bool = True
    transpose: bool = False
    show_missing_as: str = "N/A"

    def __post_init__(self):
        """Validate configuration."""
        if not self.experiments:
            raise ValueError("At least one experiment must be specified")

        if self.experiment_names:
            if len(self.experiment_names) != len(self.experiments):
                raise ValueError(
                    f"Number of experiment names ({len(self.experiment_names)}) "
                    f"must match number of experiments ({len(self.experiments)})"
                )


class AnalysisComparisonProcessor(
    Processor[AnalysisComparisonConfig, Dict[str, pd.DataFrame], pd.DataFrame]
):
    """
    Processor that runs analyses and compares results across experiments.

    This processor:
    1. Takes experiment identifiers (not file paths)
    2. Runs MultiTrialProcessor for each experiment
    3. Loads the fresh results
    4. Compares metrics across experiments
    5. Formats output as a comparison table
    """

    def __init__(self):
        """Initialize the processor."""
        super().__init__()
        self.results_dir: Optional[Path] = None

    @contextmanager
    def _isolated_hydra_context(self, config_dir: Path):
        """Context manager for isolated Hydra execution.

        Saves and restores Hydra state to allow nested Hydra apps.
        """
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir

        # Save current state
        is_initialized = GlobalHydra.instance().is_initialized()
        saved_hydra = GlobalHydra.instance().hydra if is_initialized else None

        # Clear for new context
        if is_initialized:
            GlobalHydra.instance().clear()

        try:
            # Initialize new context
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                yield
        finally:
            # Restore original state
            if is_initialized and saved_hydra:
                GlobalHydra.instance().initialize(saved_hydra)

    def _build_overrides_list(
        self, base_overrides: Dict[str, Any], user_overrides: Dict[str, Any]
    ) -> List[str]:
        """Build Hydra override list from dictionaries.

        Args:
            base_overrides: Base overrides (flat dictionary)
            user_overrides: User overrides (can be nested)

        Returns:
            List of Hydra override strings
        """
        overrides = []

        # Add base overrides (already flat)
        for key, value in base_overrides.items():
            overrides.append(f"{key}={value}")

        # Add user overrides with proper nesting
        def add_nested(prefix: str, nested: Dict[str, Any]):
            for key, value in nested.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    add_nested(full_key, value)
                else:
                    # Let Hydra handle type conversion and escaping
                    overrides.append(f"{full_key}={value}")

        for key, value in user_overrides.items():
            if isinstance(value, dict):
                add_nested(key, value)
            else:
                overrides.append(f"{key}={value}")

        return overrides

    def _resolve_config_name(self, config_name: str) -> str:
        """Resolve analysis config name to full path."""
        if "/" in config_name:
            return f"run/analysis/{config_name}"
        else:
            return f"run/analysis/multi_trial/{config_name}"

    def load_data(self, config: AnalysisComparisonConfig) -> pd.DataFrame:
        """
        Override base to handle our custom loading pattern.

        Returns empty DataFrame since we load data in extract_data.
        """
        return pd.DataFrame()

    def _extract_data(
        self, raw_data: pd.DataFrame, config: AnalysisComparisonConfig
    ) -> Dict[str, pd.DataFrame]:
        """
        Run analyses and load results.

        Args:
            raw_data: Ignored (empty DataFrame)
            config: Configuration with experiments to analyze

        Returns:
            Dictionary mapping experiment names to DataFrames
        """
        # Determine where to save results
        self.results_dir = self._determine_results_directory(config)
        logger.info(f"Analysis results will be saved to: {self.results_dir}")

        # Run all analyses
        result_paths = self._run_all_analyses(config)

        # Load the fresh results
        results = {}
        for i, path in enumerate(result_paths):
            df = pd.read_csv(path, index_col="metric")
            exp_name = (
                config.experiment_names[i]
                if config.experiment_names
                else config.experiments[i]
            )
            results[exp_name] = df
            logger.info(f"Loaded results for {exp_name}: {len(df)} metrics")

        return results

    def transform_data(
        self, data: Dict[str, pd.DataFrame], config: AnalysisComparisonConfig
    ) -> pd.DataFrame:
        """
        Extract the selected statistic and join across experiments.

        Args:
            data: Dictionary of DataFrames (one per experiment)
            config: Configuration specifying which statistic to use

        Returns:
            Combined DataFrame with one column per experiment
        """
        # Extract the statistic column from each DataFrame
        statistic_dfs = []
        for exp_name, df in data.items():
            if config.statistic not in df.columns:
                logger.warning(
                    f"Statistic '{config.statistic}' not found in {exp_name}, available: {list(df.columns)}"
                )
                continue

            stat_df = df[[config.statistic]].rename(
                columns={config.statistic: exp_name}
            )
            statistic_dfs.append(stat_df)

        if not statistic_dfs:
            raise ValueError(
                f"Statistic '{config.statistic}' not found in any experiment results"
            )

        # Join all DataFrames on the metric index
        result = statistic_dfs[0]
        for df in statistic_dfs[1:]:
            result = result.join(df, how="outer")

        # Apply metric filtering if specified
        if config.metrics_filter:
            available_metrics = result.index.intersection(config.metrics_filter)
            if not available_metrics.empty:
                result = result.loc[available_metrics]
                logger.info(f"Filtered to {len(result)} metrics")
            else:
                logger.warning("No metrics matched the filter")

        # Apply sorting if specified
        if config.sort_by and config.sort_by in result.columns:
            result = result.sort_values(config.sort_by, ascending=config.ascending)

        return result

    def compute_results(
        self, data: pd.DataFrame, config: AnalysisComparisonConfig
    ) -> pd.DataFrame:
        result = data.fillna(config.show_missing_as)
        if config.transpose:
            result = result.T

        return result

    def get_output(
        self, results: pd.DataFrame, config: AnalysisComparisonConfig, format: str
    ):
        """Format results for display."""
        if format == "rich":
            formatter = ComparisonFormatter(transpose=config.transpose)
            return formatter.format(results=results, config=config)
        else:
            # For CSV or simple format, just return the DataFrame as string
            return results.to_csv() if format == "csv" else str(results)

    # Private methods for analysis execution

    def _determine_results_directory(self, config: AnalysisComparisonConfig) -> Path:
        """Determine where to save analysis results."""
        if config.output_directory:
            results_dir = Path(config.output_directory) / "experiments"
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile

            results_dir = Path(tempfile.mkdtemp(prefix="lidra_comparison_"))
            logger.warning(
                f"Using temporary directory for results: {results_dir}. "
                "Set save_dir to persist results."
            )

        return results_dir

    def _run_all_analyses(self, config: AnalysisComparisonConfig) -> List[Path]:
        """Run MultiTrialProcessor for all experiments."""
        if config.parallel_analyses and len(config.experiments) > 1:
            return self._run_parallel_analyses(config)
        else:
            return self._run_sequential_analyses(config)

    def _run_sequential_analyses(self, config: AnalysisComparisonConfig) -> List[Path]:
        """Run analyses one at a time."""
        result_paths = []

        for i, experiment in enumerate(config.experiments):
            path = self._run_single_analysis(experiment, i, config)
            result_paths.append(path)

        return result_paths

    def _run_parallel_analyses(self, config: AnalysisComparisonConfig) -> List[Path]:
        """Run analyses in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all analyses
            future_to_exp = {
                executor.submit(self._run_single_analysis, exp, i, config): (exp, i)
                for i, exp in enumerate(config.experiments)
            }

            # Collect results in order
            result_paths = [None] * len(config.experiments)
            for future in as_completed(future_to_exp):
                exp, idx = future_to_exp[future]
                try:
                    path = future.result()
                    result_paths[idx] = path
                    logger.info(f"Completed analysis for {exp}")
                except Exception as e:
                    logger.error(f"Failed to analyze {exp}: {e}")
                    raise

        return result_paths

    def _run_single_analysis(
        self, experiment: str, index: int, config: AnalysisComparisonConfig
    ) -> Path:
        """Run analysis for a single experiment using the Compose API."""
        from hydra import compose
        import hydra

        exp_name = (
            config.experiment_names[index] if config.experiment_names else experiment
        )

        # Build base overrides
        base_overrides = {
            "config.input_file": f'"{experiment}"',
            "config.output_directory": f'"{str(self.results_dir / exp_name)}"',
            "config.quiet": "true",
        }

        # Use new overwrites format (old format handled in __post_init__)
        user_overrides = config.overwrites or {}

        # Build override list
        overrides = self._build_overrides_list(base_overrides, user_overrides)

        # Resolve config name
        config_name = self._resolve_config_name(config.analysis_config)

        # Run analysis with isolated Hydra context
        lidra_root = Path(__file__).parents[5]
        config_dir = lidra_root / "etc" / "lidra"

        with self._isolated_hydra_context(config_dir):
            # Compose configuration
            cfg = compose(config_name=config_name, overrides=overrides)

            # Instantiate and run
            analysis_config = hydra.utils.instantiate(cfg.config)
            processor = hydra.utils.instantiate(cfg.processor)
            processor.run(analysis_config)

        logger.info(
            f"Completed analysis {index+1}/{len(config.experiments)}: {experiment}"
        )
        return Path(str(self.results_dir / exp_name)) / "results.csv"
