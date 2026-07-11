import pandas as pd
import joblib

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features


def main():

    print("Loading test data...")

    test_df = load_data("data/test.csv")

    test_ids = test_df["Id"]

    test_df = clean_data(test_df)

    test_df = create_features(test_df)

    model = joblib.load("models/house_price_model.pkl")

    predictions = model.predict(test_df)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions
    })

    submission.to_csv("submission.csv", index=False)

    print("Submission file created successfully!")

    print(submission.head())


if __name__ == "__main__":
    main()