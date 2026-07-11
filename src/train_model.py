import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.preprocessing import create_preprocessor


def main():

    print("Loading training data...")

    train_df = load_data("data/train.csv")

    train_df = clean_data(train_df)

    train_df = create_features(train_df)

    X = train_df.drop("SalePrice", axis=1)
    y = train_df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
        "model__subsample": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print("Training model...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print("-" * 40)
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"R² Score : {r2:.4f}")

    print("\nBest Parameters")
    print(grid_search.best_params_)

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/house_price_model.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()