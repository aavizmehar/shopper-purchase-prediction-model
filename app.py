"""Streamlit app: Online Shopper Purchase Intention (Simplified Prototype)."""
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

# Features expected by the model
FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend"
]

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒")
st.title("Shopper Purchase Predictor")
st.write("Enter a visitor's session details to predict if they will buy something.")

# Load the model
model_path = BASE / "model.pkl"
if not model_path.exists():
    st.error("Model file `model.pkl` not found!")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# --- Simple Input Form ---
st.subheader("Session Details")

col1, col2 = st.columns(2)

with col1:
    product_related = st.number_input("Product pages visited", 0, 100, 10)
    product_duration = st.number_input("Time on product pages (seconds)", 0.0, 10000.0, 300.0)
    page_values = st.number_input("Page value", 0.0, 500.0, 0.0, help="Average value of pages viewed.")
    bounce_rates = st.number_input("Bounce rate", 0.0, 0.2, 0.02, step=0.01)
    exit_rates = st.number_input("Exit rate", 0.0, 0.2, 0.04, step=0.01)

with col2:
    administrative = st.number_input("Admin pages visited", 0, 50, 1)
    admin_duration = st.number_input("Time on admin pages (seconds)", 0.0, 5000.0, 30.0)
    informational = st.number_input("Info pages visited", 0, 50, 0)
    info_duration = st.number_input("Time on info pages (seconds)", 0.0, 5000.0, 0.0)
    
    month = st.selectbox("Month", MONTHS, index=MONTHS.index("Nov"))
    visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("Visit on a weekend")

# Defaults for remaining minor features
special_day = 0.0
operating_system = 1
browser = 1
region = 1
traffic_type = 1

# --- Prediction ---
if st.button("Predict Purchase", type="primary"):
    # Create input row
    data = [[
        administrative, admin_duration,
        informational, info_duration,
        product_related, product_duration,
        bounce_rates, exit_rates, page_values, special_day,
        month, operating_system, browser, region, traffic_type, visitor_type,
        int(weekend)
    ]]
    
    row = pd.DataFrame(data, columns=FEATURES)
    
    # Get probability
    proba = float(model.predict_proba(row)[0, 1])
    
    if proba >= 0.5:
        st.success(f"**Likely to buy!** (Confidence: {proba:.0%})")
    else:
        st.warning(f"**Unlikely to buy.** (Confidence: {1-proba:.0%})")
        
    st.progress(proba)