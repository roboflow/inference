"""Multi-trial metrics processor implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable, Iterable
import numpy as np
import pandas as pd
import optree
from loguru import logger

from .base import Processor, BaseConfig
from ..functionals import FUNCTIONALS, get_functionals, StatisticalFunctional
from ..trial_reduction import get_trial_reducer, TrialReducer
from ..formatting.csv import CSVFormatter
from ..formatting.rich_table import RichTableFormatter

# Type aliases
MetricData = Dict[str, np.ndarray]  # metric_name -> samples
MetricStats = pd.DataFrame  # DataFrame with metrics as index and statistics as columns


@dataclass
class Metric:
    direction: Literal["minimize", "maximize"]
    trial_reduction_group: Optional[str] = None  # May be grouped in a table

    @staticmethod
    def make_metrics_dict(**kwargs) -> Dict[str, "Metric"]:
        return {name: Metric(**kw) for name, kw in kwargs.items()}


@dataclass
class MultiTrialConfig(BaseConfig):
    """Configuration for multi-trial metrics analysis."""

    # Multi-trial specific fields
    metrics: Dict[str, Metric] = field(default_factory=dict)
    thresholds: Dict[str, List[float]] = field(default_factory=dict)
    trial_reduction_fns: Dict[str, Callable] = field(default_factory=dict)
    trial_reduction_mode: str = (
        "mean"  # Functional for trial reduction: "mean", "min", "max"
    )
    statistics: List[str] = field(default_factory=lambda: ["mean", "std", "p5", "p95"])

    # Data cleaning options specific to multi-trial
    drop_na: bool = False  # Drop rows with any missing values
    drop_na_columns: Optional[List[str]] = (
        None  # Drop rows missing values in these columns
    )
    drop_incomplete_trials: bool = True  # Drop samples with incomplete trials
    max_trials: Optional[int] = (
        None  # Keep only first N trials per sample -- auto-detect if None.
    )

    def __post_init__(self):
        """Validate configuration."""
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
        # if self.mode not in FUNCTIONALS:
        #     raise ValueError(f"Unknown mode: {self.mode}. Available: {list(FUNCTIONALS.keys())}")
        invalid_stats = set(self.statistics) - set(FUNCTIONALS.keys())
        if invalid_stats:
            raise ValueError(f"Unknown statistics: {invalid_stats}")


class MultiTrialProcessor(Processor[MultiTrialConfig, MetricData, MetricStats]):
    """Processor for multi-trial metrics analysis.

    Handles:
    - Multiple trials per sample that need reduction
    - Threshold transformations
    - Statistical functional computations
    """

    def _extract_data(
        self, raw_data: pd.DataFrame, config: MultiTrialConfig
    ) -> MetricData:
        """Extract metric arrays from dataframe with optional cleaning."""
        # Step 1: Apply data cleaning if requested
        if any(
            [
                config.drop_na,
                config.drop_na_columns,
                config.max_trials,
                config.drop_incomplete_trials,
            ]
        ):
            raw_data = self._clean_data(raw_data, config)

        # Step 2: Auto-detect trials if needed
        if config.max_trials is None:
            detected_trials = self._detect_trials_per_sample(raw_data)
            config.max_trials = detected_trials

        # Step 3: Filter to requested metrics that exist
        metric_names = set([m for m in config.metrics.keys()])
        available_metrics = set([m for m in metric_names if m in raw_data.columns])
        if not available_metrics:
            raise ValueError(f"No requested metrics found in data")

        missing_metrics = metric_names - available_metrics
        if missing_metrics:
            logger.info(
                f"Warning: Metrics not found in data and will be dropped from analysis: {missing_metrics}"
            )

        # Step 4: Extract empirical distribution from dataframe
        return self._reduce_multi_trials(raw_data, config)

    def transform_data(self, data: MetricData, config: MultiTrialConfig) -> MetricData:
        """Apply trial reduction and threshold transformations."""

        if config.thresholds:
            threshold_data = self._apply_thresholds(data, config.thresholds, config)
            data = {**data, **threshold_data}

        # TODO: Apply other transformations here
        return data

    def compute_results(
        self, data: MetricData, config: MultiTrialConfig
    ) -> MetricStats:
        """Compute statistical functionals on all metrics using optree."""
        functionals = get_functionals(config.statistics)

        # Use optree to apply all functionals to all metrics
        def compute_stats_for_metric(values: np.ndarray) -> Dict[str, float]:
            """Compute all statistics for a single metric."""
            return optree.tree_map(lambda func: func(values), functionals)

        # Compute all statistics using optree
        results_dict = optree.tree_map(compute_stats_for_metric, data)

        # Convert nested dict to DataFrame where each metric is a row
        df = pd.DataFrame.from_dict(results_dict, orient="index")
        df.index.name = "metric"

        # Ensure columns are in the order specified in config.statistics
        df = df[config.statistics]

        return df

    def get_output(
        self, results: MetricStats, config: MultiTrialConfig, format: str
    ) -> Any:
        """Get formatted output using appropriate formatter."""
        if format == "csv":
            formatter = CSVFormatter()
        else:  # 'rich' or any other format defaults to rich
            formatter = RichTableFormatter(
                show_section_lines=False,
                show_direction_indicators=True,
                highlight_thresholds=True,
            )

        return formatter.format(results, config)

    def _apply_thresholds(
        self,
        data: MetricData,
        thresholds: Dict[str, List[float]],
        config: MultiTrialConfig,
    ) -> MetricData:
        """Apply threshold transformations to create derived metrics."""
        result = {}

        # For each metric that has thresholds defined
        for metric, values in data.items():
            if metric in thresholds:
                # Get direction for this metric
                direction = config.metrics.get(
                    metric, Metric(direction="maximize")
                ).direction

                # Create threshold tree for this metric
                threshold_tree = {
                    f"{metric}_at_{threshold}": threshold
                    for threshold in thresholds[metric]
                }

                # Apply thresholds based on direction
                if direction == "minimize":
                    # For minimize metrics, we want values <= threshold
                    threshold_results = optree.tree_map(
                        lambda t: (values <= t).astype(float), threshold_tree
                    )
                else:
                    # For maximize metrics, we want values >= threshold
                    threshold_results = optree.tree_map(
                        lambda t: (values >= t).astype(float), threshold_tree
                    )

                result.update(threshold_results)

        return result

    def _detect_trials_per_sample(self, df: pd.DataFrame) -> Optional[int]:
        # If we have trial column and sample_uuid, use them
        if "trial" in df.columns and "sample_uuid" in df.columns:
            trial_counts = df.groupby("sample_uuid")["trial"].count()
            mode_count = trial_counts.mode()
            if len(mode_count) > 0:
                return int(mode_count[0])

        raise ValueError(
            "Cannot auto-detect trials_per_sample. Data must have 'trial' and 'sample_uuid' columns, "
        )

    def _clean_data(self, df: pd.DataFrame, config: MultiTrialConfig) -> pd.DataFrame:
        """Apply data cleaning operations to DataFrame."""
        df = df.copy()  # Don't modify input

        initial_rows = len(df)
        initial_samples = (
            df["sample_uuid"].nunique() if "sample_uuid" in df.columns else None
        )

        # Log initial state
        logger.info(f"Initial rows: {initial_rows}")
        if initial_samples is not None:
            logger.info(f"Initial unique samples: {initial_samples}")

        # Track original trials per sample before any filtering
        original_trials_per_sample = None
        if "trial" in df.columns and "sample_uuid" in df.columns:
            original_trials = df.groupby("sample_uuid")["trial"].count().mode()
            if len(original_trials) > 0:
                original_trials_per_sample = int(original_trials[0])

        # Drop rows with missing values first (to ensure data quality)
        if config.drop_na or config.drop_na_columns:
            df, stats = self._drop_missing_values(
                df, drop_all=config.drop_na, columns=config.drop_na_columns
            )
            if stats.get("rows_dropped_na", 0) > 0:
                logger.info(
                    f"Dropped {stats['rows_dropped_na']} rows due to {stats['drop_reason']}"
                )

        # Drop incomplete trials (if requested and trial column exists)
        if config.drop_incomplete_trials and "trial" in df.columns:
            df, stats = self._drop_incomplete_trials(df, config.max_trials)
            if stats.get("incomplete_samples_dropped", 0) > 0:
                logger.info(
                    f"Dropped {stats['incomplete_samples_dropped']} samples with incomplete trials"
                )
                logger.info(f"  Expected trials per sample: {stats['expected_trials']}")

        # Filter trials last (if requested)
        if config.max_trials is not None:
            logger.info(f"Filtering to first {config.max_trials} trials per sample")
            df, stats = self._filter_trials(df, config.max_trials)

            # Log trial filtering info if trials were actually reduced
            if (
                original_trials_per_sample
                and original_trials_per_sample != config.max_trials
            ):
                logger.info(
                    f"Trials per sample filtered: {original_trials_per_sample} → {config.max_trials}"
                )
            logger.info(f"  Rows removed: {stats['trials_filtered']}")

        # Log final state
        final_rows = len(df)
        if "sample_uuid" in df.columns:
            final_samples = df["sample_uuid"].nunique()
            logger.info(f"Final unique samples: {final_samples}")
            if initial_samples and initial_samples != final_samples:
                logger.info(f"  Samples removed: {initial_samples - final_samples}")

        logger.info(f"Final rows: {final_rows}")
        logger.info(
            f"Total rows removed: {initial_rows - final_rows} ({(initial_rows - final_rows) / initial_rows * 100:.1f}%)"
        )

        # Validate we still have data
        if df.empty:
            raise ValueError("No data remaining after cleaning operations")

        return df

    def _filter_trials(
        self, df: pd.DataFrame, max_trials: int
    ) -> Tuple[pd.DataFrame, dict]:
        """Keep only first N trials per sample."""
        if "trial" not in df.columns:
            raise ValueError("Cannot filter trials: 'trial' column not found")

        initial_count = len(df)
        df_filtered = df[df["trial"] <= max_trials].copy()

        if df_filtered.empty:
            raise ValueError(
                f"No data remaining after filtering to first {max_trials} trials"
            )

        return df_filtered, {
            "trials_filtered": initial_count - len(df_filtered),
            "max_trials_kept": max_trials,
        }

    def _drop_incomplete_trials(
        self, df: pd.DataFrame, expected_trials: Optional[int]
    ) -> Tuple[pd.DataFrame, dict]:
        """Drop samples with incomplete trials."""
        if "sample_uuid" not in df.columns:
            raise ValueError(
                "Cannot check for incomplete trials: 'sample_uuid' column not found"
            )

        # Use the existing method to detect trials per sample
        if expected_trials is None:
            expected_trials = self._detect_trials_per_sample(df)
            if expected_trials is None:
                raise ValueError("Cannot determine expected trials per sample")

        trial_counts = df.groupby("sample_uuid").size()
        complete_samples = trial_counts[trial_counts == expected_trials].index
        df_filtered = df[df["sample_uuid"].isin(complete_samples)].copy()

        dropped_samples = len(trial_counts) - len(complete_samples)

        return df_filtered, {
            "incomplete_samples_dropped": dropped_samples,
            "expected_trials": expected_trials,
        }

    def _drop_missing_values(
        self,
        df: pd.DataFrame,
        drop_all: bool = False,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """Drop rows with missing values."""
        initial_count = len(df)

        if drop_all:
            df_filtered = df.dropna()
            dropped_reason = "any missing value"
        elif columns:
            # Validate columns exist
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df_filtered = df.dropna(subset=columns)
            dropped_reason = f"missing values in {columns}"
        else:
            return df, {}

        return df_filtered, {
            "rows_dropped_na": initial_count - len(df_filtered),
            "drop_reason": dropped_reason,
        }

    def _reduce_multi_trials(
        self, df: pd.DataFrame, config: MultiTrialConfig
    ) -> MetricData:
        def reduce_trials_function(metric_name: str) -> np.ndarray:
            """Reduce trials for a single metric."""
            VALID_MODES = ["mean", "best_by_group", "best_independent"]
            if config.trial_reduction_mode not in VALID_MODES:
                raise ValueError(
                    f"Invalid mode: {config.trial_reduction_mode}. Valid modes: {VALID_MODES}"
                )
            if config.trial_reduction_mode == "mean":
                logger.info(f"Reducing {metric_name} by mean")
                return get_trial_reducer("mean")(df, metric_name)
            elif config.trial_reduction_mode == "best_by_group":
                group = config.metrics[metric_name].trial_reduction_group
                fn = config.trial_reduction_fns.get(group)
                return fn(df, metric_name)
            elif config.trial_reduction_mode == "best_independent":
                objective_direction = config.metrics[metric_name].direction
                assert objective_direction in ["minimize", "maximize"]
                should_minimize = objective_direction == "minimize"
                return get_trial_reducer("select_by")(
                    df, metric_name, metric_name, should_minimize
                )
            else:
                raise NotImplementedError(
                    f"Unknown mode: {config.trial_reduction_mode}"
                )

        return {
            metric_name: reduce_trials_function(metric_name)
            for metric_name in config.metrics.keys()
        }
