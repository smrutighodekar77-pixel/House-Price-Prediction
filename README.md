# 🏠 House Price Prediction using Machine Learning

Predict house selling prices using Machine Learning based on property features such as living area, neighborhood, overall quality, garage capacity, basement area, and many other housing characteristics.

This project demonstrates an end-to-end Machine Learning workflow, including data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment through an interactive Streamlit web application.

---


## 🚀 Live Demo

🔗 **Streamlit App:

Example:

https://house-price-prediction-vrp938jxl6ma2mzjjktbkz.streamlit.app/

---

## 📌 Project Overview

The objective of this project is to predict residential house prices using Machine Learning regression models trained on the **House Prices: Advanced Regression Techniques** dataset from Kaggle, which is based on the **Ames Housing Dataset**.

The complete workflow includes:

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- House Price Prediction through Streamlit

---

## 📂 Dataset

- **Dataset:** House Prices: Advanced Regression Techniques
- **Training Records:** 1460
- **Test Records:** 1459
- **Original Features:** 79
- **Target Variable:** SalePrice

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing values
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- ColumnTransformer
- Scikit-learn Pipeline

---

## 🧠 Feature Engineering

Feature engineering was performed before training the models to improve prediction performance.

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## 📈 Best Model Performance

### Gradient Boosting Regressor

| Metric | Score |
|---------|-------|
| MAE | 15,901.08 |
| RMSE | 25,889.42 |
| R² Score | **0.9126** |

### Best Hyperparameters

```python
{
    'model__learning_rate': 0.1,
    'model__max_depth': 4,
    'model__n_estimators': 200,
    'model__subsample': 0.8
}
```

---

## ⭐ Top Important Features

- OverallQual
- GrLivArea
- GarageCars
- 2ndFlrSF
- BsmtFinSF1
- 1stFlrSF
- TotalBsmtSF
- LotArea
- GarageFinish
- YearRemodAdd

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```


## 💻 Streamlit Application

The application allows users to:

- Enter property details
- Predict house selling prices
- Display predicted prices in both **USD** and **INR**

---



## 📄 License

This project is developed for educational and portfolio purposes.
