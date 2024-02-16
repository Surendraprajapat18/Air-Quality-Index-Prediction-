import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the trained model
model = joblib.load("Model/RandomForestRegressorM.joblib")

# Mapping of abbreviated names to full forms
feature_mapping = {
    'PM2.5': 'Particulate Matter 2.5 micrometers(PM2.5)',
    'PM10': 'Particulate Matter 10 micrometers(PM10)',
    'NO2': 'Nitrogen Dioxide(NO2)',
    'CO': 'Carbon Monoxide(CO)',
    'SO2': 'Sulfur Dioxide(SO2)',
    'O3': 'Ozone(O3)'
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

# City mapping to numbers
city_mapping = {
    'Ahmedabad': 0, 'Aizawl': 1, 'Amaravati': 2, 'Amritsar': 3, 'Bengaluru': 4,
    'Bhopal': 5, 'Brajrajnagar': 6, 'Chandigarh': 7, 'Chennai': 8, 'Coimbatore': 9,
    'Delhi': 10, 'Ernakulam': 11, 'Gurugram': 12, 'Guwahati': 13, 'Hyderabad': 14,
    'Jaipur': 15, 'Jorapokhar': 16, 'Kochi': 17, 'Kolkata': 18, 'Lucknow': 19,
    'Mumbai': 20, 'Patna': 21, 'Shillong': 22, 'Talcher': 23, 'Thiruvananthapuram': 24,
    'Visakhapatnam': 25
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
    st.sidebar.markdown("<h3 style='text-align: center;'>Give the Inputs:</h3>", unsafe_allow_html=True)

    # Dropdown for selecting the city
    city = st.sidebar.selectbox("Select City:", ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
                                                  'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
                                                  'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
                                                  'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai',
                                                  'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
                                                  'Visakhapatnam'])

    # Convert city name into number
    city_number = city_mapping.get(city, -1)
    if city_number == -1:
        st.warning("Selected city not found in the mapping.")

    input_features = {}
    features = list(feature_mapping.keys())

    for feature in features:
        # Append a unique identifier to the key
        unique_key = f"{feature}_input"
        input_features[feature] = st.sidebar.number_input(f"Enter {feature_mapping[feature]} value:", min_value=0, key=unique_key)

    # Predict button
    if st.sidebar.button("Predict AQI"):
        input_arr = [city_number] + [input_features[feature] for feature in features]
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
    input_data.insert(0, 'City', city)  # Add the selected city to the input data
    st.table(input_data)

if __name__ == "__main__":
    main()
