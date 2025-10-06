# to code 

# streamlit_app.py

import streamlit as st
import requests

st.set_page_config(page_title="GDP Growth Predictor", layout="centered")
st.title("📈 GDP Growth Predictor")

# --- User Inputs ---
st.sidebar.header("Enter Macroeconomic Data")
country_code = st.sidebar.text_input("Country Code (e.g., IND, USA, ABW)")
year = st.sidebar.number_input("Year", min_value=1960, max_value=2100, value=2025)
inflation = st.sidebar.number_input("Inflation (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
unemployment = st.sidebar.number_input("Unemployment (%)", min_value=0.0, max_value=100.0, value=7.0, step=0.1)

# --- Button ---
if st.button("Predict GDP Growth"):
    if not country_code:
        st.error("Please enter a valid country code!")
    else:
        payload = {
            "country_code": country_code.upper(),
            "year": year,
            "inflation": inflation,
            "unemployment": unemployment
        }
        
        # URL of your Flask API
        API_URL = "http://127.0.0.1:5000/predict"
        
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted GDP Growth for {result['country_code']} in {result['year']}: **{result['predicted_gdp_growth']}%**")
            else:
                st.error(f"API Error: {response.json().get('error')}")
        
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
