from typing import Callable
import pandas as pd
import numpy as np

# Type alias for clarity
#  - Takes a dataframe with many trials per sample and a metric column name
#  - Return a np.array: samples from an empirical distribution
#  - Requires column names sample_uuid, trial, and metric_column_name
TrialReducer = Callable[[pd.DataFrame, str], np.ndarray]


def mean(df: pd.DataFrame, metric_column_name: str) -> np.ndarray:
    """Reduce trials to mean of each sample."""
    return df.groupby("sample_uuid")[metric_column_name].mean().values


def select_by(
    df: pd.DataFrame,
    value_column_name: str,
    select_trials_by: str,
    select_trials_should_be_minimized: bool,
) -> np.ndarray:
    """Select best trial per sample based on specified metric."""
    if select_trials_by not in df.columns:
        raise ValueError(f"Selection metric '{select_trials_by}' not found")

    if "sample_uuid" not in df.columns:
        raise ValueError("Cannot select trials without 'sample_uuid' column")

    # Group by sample and get index of best trial
    grouped = df.groupby("sample_uuid")[select_trials_by]
    if select_trials_should_be_minimized:
        best_indices = grouped.idxmin()
    else:
        best_indices = grouped.idxmax()

    # Return only the best trials
    return df[value_column_name].loc[best_indices].values


TRIAL_REDUCERS = {
    "mean": mean,
    "select_by": select_by,
}


def get_trial_reducer(reducer_name: str) -> TrialReducer:
    if reducer_name not in TRIAL_REDUCERS:
        raise ValueError(f"Unknown trial reducer: {reducer_name}")
    return TRIAL_REDUCERS[reducer_name]
