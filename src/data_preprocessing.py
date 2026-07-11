import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load dataset from the given file path.
    """
    return pd.read_csv(filepath)


def drop_high_missing_columns(df):
    """
    Drop columns with more than 80% missing values.
    """

    columns_to_drop = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def fill_missing_values(df):
    """
    Fill missing values in numerical and categorical columns.
    """

    # Numerical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def clean_data(df):
    """
    Perform complete data cleaning.
    """

    df = drop_high_missing_columns(df)

    df = fill_missing_values(df)

    return df