import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import clean_data
from src.feature_engineering import create_features

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------

st.markdown("""
<style>

/* Main App */
.stApp{
    background:#F5F9FF;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#1E3A8A;
}

/* Sidebar text */
section[data-testid="stSidebar"] *{
    color:white;
}

/* Main Title */
.main-title{
    font-size:42px;
    font-weight:700;
    color:#1E3A8A;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:#5B6475;
    font-size:18px;
    margin-bottom:25px;
}

/* Card */
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom:15px;
}

/* Prediction Box */
.prediction-box{
    background:#E8F5E9;
    border-left:8px solid #2E7D32;
    border-radius:12px;
    padding:25px;
    text-align:center;
}

.prediction-title{
    color:#1E3A8A;
    font-size:20px;
    font-weight:bold;
}

.prediction-price{
    color:#2E7D32;
    font-size:42px;
    font-weight:bold;
}

/* Button */
.stButton>button{
    width:100%;
    height:55px;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

/* Metric */
[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = joblib.load("models/house_price_model.pkl")

# -----------------------------------------------------
# Header
# -----------------------------------------------------

st.markdown(
"""
<div class="main-title">
🏠 House Price Prediction
</div>

<div class="sub-title">
Estimate the selling price of a house using a Machine Learning model trained on the Ames Housing Dataset.
</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/home.png",
    width=80
)

st.sidebar.title("House Details")

st.sidebar.markdown(
"""
Adjust the property details below and click **Predict Price**.
"""
)

# -----------------------------------------------------
# Two Columns
# -----------------------------------------------------

left, right = st.columns(2)

with left:

    st.markdown(
    """
    <div class="card">
    <h3>🏡 Property Details</h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    overall_qual = st.slider(
        "Overall Quality",
        1,10,5
    )

    overall_cond = st.slider(
        "Overall Condition",
        1,10,5
    )

    gr_liv_area = st.number_input(
        "Ground Living Area (sq ft)",
        300,6000,1500
    )

    lot_area = st.number_input(
        "Lot Area",
        1000,50000,8500
    )

    bedrooms = st.slider(
        "Bedrooms",
        1,8,3
    )

    full_bath = st.slider(
        "Full Bathrooms",
        0,5,2
    )

    half_bath = st.slider(
        "Half Bathrooms",
        0,3,1
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "NAmes","CollgCr","OldTown",
            "Edwards","Somerst",
            "NridgHt","Sawyer",
            "Gilbert","NWAmes","BrkSide"
        ]
    )

with right:

    st.markdown(
    """
    <div class="card">
    <h3>🏗 Construction Details</h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    garage_cars = st.slider(
        "Garage Capacity",
        0,5,2
    )

    garage_area = st.number_input(
        "Garage Area",
        0,1500,500
    )

    total_bsmt_sf = st.number_input(
        "Basement Area",
        0,4000,900
    )

    first_floor = st.number_input(
        "1st Floor Area",
        300,3000,1200
    )

    second_floor = st.number_input(
        "2nd Floor Area",
        0,2500,300
    )

    year_built = st.slider(
        "Year Built",
        1875,2025,2000
    )

    year_remod = st.slider(
        "Year Remodeled",
        1950,2025,2005
    )

    kitchen_quality = st.selectbox(
        "Kitchen Quality",
        ["Ex","Gd","TA","Fa"]
    )

    garage_finish = st.selectbox(
        "Garage Finish",
        ["Fin","RFn","Unf"]
    )

    house_style = st.selectbox(
        "House Style",
        ["1Story","2Story","1.5Fin","SLvl","SFoyer"]
    )

st.markdown("---")

# -----------------------------------------------------
# Default Values for Remaining Features
# -----------------------------------------------------

input_data = {
    "Id": 1,
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 70,
    "LotArea": lot_area,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": neighborhood,
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": house_style,
    "OverallQual": overall_qual,
    "OverallCond": overall_cond,
    "YearBuilt": year_built,
    "YearRemodAdd": year_remod,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1": total_bsmt_sf,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": total_bsmt_sf,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": first_floor,
    "2ndFlrSF": second_floor,
    "LowQualFinSF": 0,
    "GrLivArea": gr_liv_area,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": full_bath,
    "HalfBath": half_bath,
    "BedroomAbvGr": bedrooms,
    "KitchenAbvGr": 1,
    "KitchenQual": kitchen_quality,
    "TotRmsAbvGrd": 7,
    "Functional": "Typ",
    "Fireplaces": 1,
    "FireplaceQu": "Gd",
    "GarageType": "Attchd",
    "GarageYrBlt": year_built,
    "GarageFinish": garage_finish,
    "GarageCars": garage_cars,
    "GarageArea": garage_area,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 100,
    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": None,
    "Fence": None,
    "MiscFeature": None,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",
    "SaleCondition": "Normal"
}

df = pd.DataFrame([input_data])

df = clean_data(df)
df = create_features(df)

# -----------------------------------------------------
# Prediction Section
# -----------------------------------------------------

st.markdown(
"""
<div style='text-align:center'>
<h2 style='color:#1E3A8A;'>💰 House Price Estimation</h2>
<p style='color:#666;'>Click the button below to estimate the selling price.</p>
</div>
""",
unsafe_allow_html=True
)

if st.button("🔮 Predict House Price"):

    prediction = model.predict(df)[0]
    usd_to_inr = 86      # You can update this value later if needed
    prediction_inr = prediction * usd_to_inr
    st.markdown(f"""
                <div class="prediction-box">
                <h3 class="prediction-title">
                🏠 Estimated House Price
                </h3>
                <h2 style="color:#2563EB;">
                💵 ${prediction:,.2f}
                </h2>
                <h1 class="prediction-price">
                🇮🇳 ₹ {prediction_inr:,.0f}
                </h1>
                </div>
                """, unsafe_allow_html=True)

    

    st.success("✅ Prediction generated successfully!")

st.markdown("<br>", unsafe_allow_html=True)

