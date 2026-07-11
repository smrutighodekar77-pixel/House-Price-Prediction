import pandas as pd
import numpy as np


def create_features(df):
    """
    Create new features from existing columns.
    """

    # House Age
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Years Since Remodel
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # Total Square Footage
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # Total Bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )

    # Total Porch Area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # Has Garage
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    # Has Basement
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

    # Has Fireplace
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Has Pool
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)

    return df