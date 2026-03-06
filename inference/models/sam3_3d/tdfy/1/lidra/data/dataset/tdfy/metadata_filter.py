import pandas as pd
from typing import Callable, List, Optional


def custom_metadata_filter(
    filter_funcs: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if filter_funcs is not None:
            for filter_func in filter_funcs:
                df = filter_func(df)
        return df

    return filter_dataframe


def data_query_filter(query: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if query:
                filtered_df = df.query(query)
            else:
                filtered_df = df
        except Exception as e:
            raise ValueError(f"Error applying query: {e}")
        return filtered_df

    return filter_dataframe


def unique_mesh_filter(col_name="sha256") -> Callable[[pd.DataFrame], pd.DataFrame]:
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df["mesh_group"] = df[col_name].apply(lambda x: x.split("_")[0])
        unique_mesh_df = df.groupby("mesh_group").first().reset_index(drop=True)

        return unique_mesh_df

    return filter_dataframe


def keep_data_fraction(
    proportion: float, col_name="sha256", ascending=True
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    assert 0 < proportion <= 1, "Proportion must be between 0 and 1"

    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=col_name, ascending=ascending)
        num_to_keep = int(len(df) * proportion)
        return df.head(num_to_keep)

    return filter_dataframe
