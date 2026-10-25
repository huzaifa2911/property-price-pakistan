import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pakistan Property Price Predictor", page_icon="🏠", layout="centered")

MODEL_FILE = "property_price_model.joblib"
LOOKUP_FILE = "app_lookup.json"

SQFT_PER_UNIT = {"Marla": 272.25, "Kanal": 5445.0, "Square Feet": 1.0}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_lookup():
    with open(LOOKUP_FILE) as f:
        return json.load(f)


def format_pkr(amount: float) -> str:
    """Show a price the way Pakistanis read it: Crore / Lakh."""
    if amount >= 1e7:
        return f"{amount / 1e7:,.2f} Crore"
    if amount >= 1e5:
        return f"{amount / 1e5:,.2f} Lakh"
    return f"{amount:,.0f}"


model = load_model()
lookup = load_lookup()

st.title("🏠 Pakistan Property Price Predictor")
st.caption(
    f"Model: **{lookup['model_name']}**  |  trained on {lookup['n_listings']:,} Zameen.com sale listings  |  "
    f"10-fold CV RMSE: {lookup['cv_mean_rmse'] / 1e6:,.1f} million PKR  |  R²: {lookup['cv_mean_r2']:.2f}"
)

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", sorted(lookup["locations"].keys()))
    property_type = st.selectbox("Property type", lookup["property_types"])
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=lookup["bedrooms_max"], value=3)
    baths = st.number_input("Bathrooms", min_value=0, max_value=lookup["baths_max"], value=3)
with col2:
    locations = sorted(lookup["locations"][city].keys())
    location = st.selectbox("Location / Society", locations)
    unit = st.selectbox("Area unit", list(SQFT_PER_UNIT.keys()))
    default_size = {"Marla": 5.0, "Kanal": 1.0, "Square Feet": 1500.0}[unit]
    area_value = st.number_input(f"Area ({unit})", min_value=0.1, value=default_size, step=0.5)

if st.button("Predict price", type="primary"):
    area_sqft = area_value * SQFT_PER_UNIT[unit]

    lat, lon = lookup["locations"][city][location]
    if lat is None or lon is None:
        lat, lon = lookup["city_coords"][city]

    # Column names/order must match the training data exactly
    row = pd.DataFrame([{
        "bedrooms": bedrooms,
        "baths": baths,
        "area_sqft": area_sqft,
        "latitude": lat,
        "longitude": lon,
        "property_type": property_type,
        "city": city,
        "location": location,
    }])

    price = float(model.predict(row)[0])
    st.success(f"Estimated price: **PKR {format_pkr(price)}**")
    st.write(f"≈ PKR {price:,.0f}  ·  {area_sqft:,.0f} sq ft  ·  about PKR {price / area_sqft:,.0f} per sq ft")
    st.caption("This is an estimate from past asking prices, not an official valuation.")
