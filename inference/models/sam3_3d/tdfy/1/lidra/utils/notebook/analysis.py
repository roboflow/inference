from typing import Optional
import pandas as pd


def validate_equal_trials(
    df: pd.DataFrame,
    trials_col: str,
    *groupby_args,
    **groupby_kwargs,
) -> None:
    """
    Validates that all groups have the same number of trials and the same trial identifiers.

    Args:
        df: DataFrame containing trial data
        trials_col: Column name for trial identifiers
        *groupby_args: Positional arguments to pass to df.groupby()
        **groupby_kwargs: Keyword arguments to pass to df.groupby()

    Example:
        # Validate that each sample_uuid has the same number of trials
        df = pd.DataFrame({
            'sample_uuid': ['sample1', 'sample1', 'sample2', 'sample2'],
            'trial': [1, 2, 1, 2],
            'score': [0.1, 0.2, 0.3, 0.4]
        })
        validate_equal_trials(df, trials_col='trial', by='sample_uuid')
    """
    all_trials = df[trials_col].unique()
    grouped_df = df.groupby(*groupby_args, **groupby_kwargs)
    trials_count = grouped_df[trials_col].count()
    assert trials_count.max() == trials_count.min(), (
        "All samples must have the same number of trials, but got max: %d and min: %d"
        % (trials_count.max(), trials_count.min())
    )
    assert (
        len(all_trials) == trials_count.max()
    ), f"All samples must have the same trial names, but found disjoint trial ids {len(all_trials)=} and max trials count {trials_count.max()=}"


def keep_first_k_trials_and_group_by_sample_uuid(
    df: pd.DataFrame,
    k_trials: Optional[int] = None,
    sample_uuid_col: str = "sample_uuid",
    trials_col: str = "trial",
):
    if k_trials is None:
        kept_trials = df
    else:
        kept_trials = df[df[trials_col] <= k_trials]
    return kept_trials.groupby(sample_uuid_col)


def calculate_trial_statistics(
    df, functional, values_col="f1", sample_uuid_col="sample_uuid", trials_col="trial"
):
    """
    Calculate mean and standard deviation of values for different numbers of trials.

    Args:
        df: DataFrame containing the data
        values_col: Column name for the values to analyze
        sample_uuid_col: Column name for the sample UUID
        trials_col: Column name for the trial number

    Returns:
        tuple: Lists of k_trials, mean values, and standard deviation values
    """
    n_trials = df[trials_col].max()

    # Store values for plotting
    k_trials_list = []
    mean_values = []
    std_values = []

    for k_trials in range(1, n_trials + 1):
        grouped_df = keep_first_k_trials_and_group_by_sample_uuid(
            df, k_trials, sample_uuid_col, trials_col
        )
        value = grouped_df[values_col].apply(functional)
        k_trials_list.append(k_trials)
        mean_values.append(value.mean())
        std_values.append(value.std())

    return k_trials_list, mean_values, std_values
