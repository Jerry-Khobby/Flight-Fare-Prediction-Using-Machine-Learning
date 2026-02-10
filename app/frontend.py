import streamlit as st
import requests
from datetime import datetime, timedelta
import joblib

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")
API_URL = "http://localhost:8000/predict"

st.title("Bangladesh Flight Fare Prediction System")
st.markdown("Predict domestic and international flight prices based on travel details")
st.sidebar.header("Flight Details")

# -----------------------------
# Load allowed categories from training
# -----------------------------
feature_columns = joblib.load("../feature_columns.pkl")

def extract_categories(prefix):
    return sorted([f.replace(prefix, "") for f in feature_columns if f.startswith(prefix)])

airlines = extract_categories("Airline_")
aircraft_types = extract_categories("Aircraft Type_")
sources = extract_categories("Source_")
destinations = extract_categories("Destination_")
seasons = extract_categories("Season_")

# Map source/destination codes to full names
source_names = {
    "BZL": "Barisal", "CGP": "Chittagong", "CXB": "Cox's Bazar",
    "DAC": "Dhaka", "JSR": "Jessore", "RJH": "Rajshahi",
    "SPD": "Saidpur", "ZYL": "Sylhet"
}
destination_names = {
    "BZL": "Barisal", "CCU": "Kolkata", "CGP": "Chittagong",
    "CXB": "Cox's Bazar", "DAC": "Dhaka", "DEL": "Delhi",
    "DOH": "Doha", "DXB": "Dubai", "IST": "Istanbul",
    "JED": "Jeddah", "JFK": "New York", "JSR": "Jessore",
    "KUL": "Kuala Lumpur", "LHR": "London", "RJH": "Rajshahi",
    "SIN": "Singapore", "SPD": "Saidpur", "YYZ": "Toronto",
    "ZYL": "Sylhet"
}

# -----------------------------
# User Input
# -----------------------------
def user_input_features():
    airline = st.sidebar.selectbox("Airline", airlines)
    source_code = st.sidebar.selectbox("Source", sources)
    dest_code = st.sidebar.selectbox("Destination", destinations)
    stopovers = st.sidebar.selectbox("Stopovers", ["Direct", "1 Stop", "2 Stops"])
    aircraft_type = st.sidebar.selectbox("Aircraft Type", aircraft_types)
    travel_class = st.sidebar.selectbox("Class", ["Economy", "Business", "First Class"])
    booking_source = st.sidebar.selectbox("Booking Source", ["Direct Booking", "Online Website", "Travel Agency"])
    season = st.sidebar.selectbox("Season", seasons)
    duration = st.sidebar.number_input("Duration (hours)", 0.5, 24.0, 1.5, 0.5)
    days_before = st.sidebar.number_input("Days Before Departure", 0, 365, 14)
    dep_date = st.sidebar.date_input("Departure Date", datetime.now() + timedelta(days=14), min_value=datetime.now())
    arr_date = st.sidebar.date_input("Arrival Date", datetime.now() + timedelta(days=14), min_value=dep_date)

    data = {
        "Airline": airline,
        "Source": source_code,
        "Destination": dest_code,
        "Stopovers": stopovers,
        "Aircraft_Type": aircraft_type,
        "Class_": travel_class,
        "Booking_Source": booking_source,
        "Season": season,
        "Duration_hrs": duration,
        "Days_Before_Departure": days_before,
        "Departure_Date": str(dep_date),
        "Arrival_Date": str(arr_date)
    }

    return data, source_names.get(source_code, source_code), destination_names.get(dest_code, dest_code)

input_data, source_display, dest_display = user_input_features()

# -----------------------------
# Flight Summary
# -----------------------------
st.subheader("Flight Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("From", source_display)
    st.write(f"**Airline:** {input_data['Airline']}")
    st.write(f"**Aircraft:** {input_data['Aircraft_Type']}")
    st.write(f"**Stopovers:** {input_data['Stopovers']}")

with col2:
    st.metric("To", dest_display)
    st.write(f"**Class:** {input_data['Class_']}")
    st.write(f"**Duration:** {input_data['Duration_hrs']} hrs")
    st.write(f"**Season:** {input_data['Season']}")

with col3:
    st.metric("Days Before", input_data['Days_Before_Departure'])
    st.write(f"**Booking:** {input_data['Booking_Source']}")
    st.write(f"**Departure:** {input_data['Departure_Date']}")
    st.write(f"**Arrival:** {input_data['Arrival_Date']}")

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Fare", type="primary", use_container_width=True):
    try:
        with st.spinner("Calculating price..."):
            response = requests.post(API_URL, json=input_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            predicted_price = result["predicted_total_fare"]

            st.success("Prediction Complete!")
            st.markdown(
                f"<h1 style='text-align: center; color: #1f77b4;'>৳ {predicted_price:,.2f} BDT</h1>",
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", "Gradient Boosting")
            with col2:
                st.metric("R² Score", "0.66")
            with col3:
                st.metric("Avg Error", "±৳28,465")

            lower = max(0, predicted_price - 28465)
            upper = predicted_price + 28465
            st.info(f"**Expected Range:** ৳{lower:,.2f} - ৳{upper:,.2f} BDT")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to API. Run: `uvicorn backend:app --reload`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("### Quick Insights")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Most Popular Routes:**\n- RJH → SIN\n- DAC → DXB\n- BZL → YYZ")
with col2:
    st.markdown("**Avg Fares by Season:**\n- Hajj: ৳96,190\n- Regular: ৳67,337\n- Winter Holidays: ৳70,500")

st.caption("*Model trained on 57,000+ records*")
