from pathlib import Path

import numpy as np
import pandas as pd

# Data source:
# https://www.kaggle.com/sveneschlbeck/resale-flat-prices-in-singapore
data_path = Path(__file__).parents[0] / "flat-prices.csv"


def to_mb(bytes: float) -> float:
    """Function to convert bytes to mega bytes (MB)"""
    return bytes / 1024 ** 2


def no_optimization() -> pd.DataFrame:
    """Load the data without any optimization"""
    df = pd.read_csv(data_path)
    return df


def with_numerical_types() -> pd.DataFrame:
    """Load the data with proper numerical types"""
    df = pd.read_csv(
        data_path,
        dtype={
            "floor_area_sqm": np.float16,
            "resale_price": np.uint32,
        },
    )
    return df


def with_numerical_and_categorical_types() -> pd.DataFrame:
    """Load the data with proper numerical and categorical types"""
    df = pd.read_csv(
        data_path,
        dtype={
            "floor_area_sqm": np.float16,
            "resale_price": np.uint32,
            "flat_model": "category",
            "flat_type": "category",
            "storey_range": "category",
            "block": "category",
            "town": "category",
        },
    )
    return df


def with_numerical_and_categorical_types_and_without_unused_columns() -> pd.DataFrame:
    """Load the data with proper numerical and categorical types
    and without unused columns"""
    dtype = {
        "floor_area_sqm": np.float16,
        "resale_price": np.uint32,
        "flat_model": "category",
        "flat_type": "category",
        "storey_range": "category",
        "block": "category",
        "town": "category",
    }
    df = pd.read_csv(data_path, dtype=dtype, usecols=list(dtype.keys()))
    return df


def analyze(df: pd.DataFrame, title: str) -> float:
    """Print the memory used by the dataframe in MB and return the
    bytes"""
    bytes = df.memory_usage(deep=True).sum()
    mb = to_mb(bytes)
    print(f"{title}: {mb:.2f} MB")
    print(df.dtypes)
    return mb


if __name__ == "__main__":
    # First we start with no optimizations
    df_no_opti = no_optimization()
    mb_no_opti = analyze(df_no_opti, "No optimizations")

    print()

    # Then we use proper numerical types
    df_proper_types = with_numerical_types()
    mb_proper_types = analyze(df_proper_types, "Proper numerical types")
    reduction = (mb_no_opti - mb_proper_types) / mb_no_opti * 100
    print(f"Size reduced by: {reduction:.2f}%")

    print()

    # Then we use proper numerical and categorical types
    df_categorical = with_numerical_and_categorical_types()
    mb_categorical = analyze(df_categorical, "Proper numerical and categorical types")
    reduction = (mb_no_opti - mb_categorical) / mb_no_opti * 100
    print(f"Size reduced by: {reduction:.2f}%")

    print()

    # No need to load unused columns. We often use a subset of the
    # columns, releveant to our analysis.
    # We assume that we are only interessed in the columns where we
    # defined the types (numerical and categrical)
    df_without_unused_columns = (
        with_numerical_and_categorical_types_and_without_unused_columns()
    )
    mb_without_unused_columns = analyze(
        df_without_unused_columns,
        "Proper numerical and categorical types, without unused columns",
    )
    reduction = (mb_no_opti - mb_without_unused_columns) / mb_no_opti * 100
    print(f"Size reduced by: {reduction:.2f}%")
