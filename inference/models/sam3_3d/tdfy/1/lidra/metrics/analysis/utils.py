"""Utility functions for metrics analysis."""

from typing import Dict
import pandas as pd


def combine_tables(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined_df = pd.concat(results, names=["table", "metric"]).reset_index()
    return combined_df[["table", "metric", "value"]]
