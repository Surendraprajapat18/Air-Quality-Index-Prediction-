# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the trained model
model = joblib.load("Model/RandomForestRegressor.joblib")

# Mapping of abbreviated names to full forms
feature_mapping = {
    'PM2.5': 'Particulate Matter 2.5 micrometers or smaller',
    'PM10': 'Particulate Matter 10 micrometers or smaller',
    'NO': 'Nitric Oxide',
    'NO2': 'Nitrogen Dioxide',
    'NOx': 'Nitrogen Oxides (NO + NO2)',
    'NH3': 'Ammonia',
    'CO': 'Carbon Monoxide',
    'SO2': 'Sulfur Dioxide',
    'O3': 'Ozone',
    'Benzene': '"Benzene" volatile organic compound (VOC)',
    'Toluene': '"Toluene" volatile organic compound (VOC)'
}

# AQI ranges and categories with corresponding emojis
aqi_ranges = {
    'Good': (0, 50, '😊'),
    'Moderate': (51, 100, '😊'),
    'Unhealthy for Sensitive Groups': (101, 150, '😷'),
    'Unhealthy': (151, 200, '😷'),
    'Very Unhealthy': (201, 300, '😷'),
    'Hazardous': (301, 500, '☠️')
}

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Streamlit app
def main():

    # Customizing the layout
    st.markdown(
        """
        <style>
        .big-font {
            font-size: 24px !important;
        }
        .stSidebar {
            background-color: #f5f5f5;
        }
        .st-cw {
            max-width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 style='text-align: center; color: #008080;'>Capstone Project IITD</h2>", unsafe_allow_html=True)

    # Adding title and subtitle
    st.title("Air Quality Index Prediction App")
    
    # Sidebar with input values
    st.sidebar.header("Enter the values:")
    input_features = {}
    features = list(feature_mapping.keys())

    for feature in features:
        # Append a unique identifier to the key
        unique_key = f"{feature}_input"
        input_features[feature] = st.sidebar.number_input(f"Enter {feature_mapping[feature]} value:", min_value=0.0, key=unique_key)

    # Predict button
    if st.sidebar.button("Predict AQI"):
        input_arr = [input_features[feature] for feature in features]
        input_np_arr = np.asarray(input_arr)
        reshaped_arr = input_np_arr.reshape((1, -1))
        prediction = model.predict(reshaped_arr)[0]

        # Determine AQI category based on prediction
        aqi_category = 'Unknown'
        for category, (lower, upper, emoji) in aqi_ranges.items():
            if lower <= prediction <= upper:
                aqi_category = category
                break

        # Display the prediction with a colorful success message and AQI category
        st.success(f"🌬️ The predicted AQI is: {prediction}, which falls in the category of '{aqi_category}' {emoji}.")

    # Display the data
    st.subheader("Input Data:")
    input_data = pd.DataFrame([input_features], columns=features)
    st.table(input_data)

if __name__ == "__main__":
    main()
