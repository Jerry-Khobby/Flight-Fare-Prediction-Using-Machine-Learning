import streamlit as st
import requests
from datetime import datetime, timedelta
import joblib

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")
API_URL = "http://localhost:8000/predict"


#API_URL = "http://backend:8000/predict" """


st.title("Bangladesh Flight Fare Prediction")
st.sidebar.header("Flight Details")

# Load features
feature_columns = joblib.load("../modelling/outputs/models/feature_columns.pkl")

def extract_categories(prefix):
    return sorted([f.replace(prefix, "") for f in feature_columns if f.startswith(prefix)])

airlines = extract_categories("Airline_")
aircraft_types = extract_categories("Aircraft Type_")
sources = extract_categories("Source_")
destinations = extract_categories("Destination_")
seasons = extract_categories("Season_")

source_names = {"BZL": "Barisal", "CGP": "Chittagong", "CXB": "Cox's Bazar", 
                "DAC": "Dhaka", "JSR": "Jessore", "RJH": "Rajshahi", "SPD": "Saidpur", "ZYL": "Sylhet"}
dest_names = {**source_names, "BKK": "Bangkok", "CCU": "Kolkata", "DEL": "Delhi", "DOH": "Doha",
              "DXB": "Dubai", "IST": "Istanbul", "JED": "Jeddah", "JFK": "New York", 
              "KUL": "Kuala Lumpur", "LHR": "London", "SIN": "Singapore", "YYZ": "Toronto"}

def user_input():
    airline = st.sidebar.selectbox("Airline", airlines)
    source = st.sidebar.selectbox("Source", sources)
    dest = st.sidebar.selectbox("Destination", destinations)
    stopovers = st.sidebar.selectbox("Stopovers", ["Direct", "1 Stop", "2 Stops"])
    aircraft = st.sidebar.selectbox("Aircraft", aircraft_types)
    travel_class = st.sidebar.selectbox("Class", ["Economy", "Business", "First Class"])
    booking = st.sidebar.selectbox("Booking", ["Direct Booking", "Online Website", "Travel Agency"])
    season = st.sidebar.selectbox("Season", seasons)
    duration = st.sidebar.number_input("Duration (hrs)", 0.5, 24.0, 1.5, 0.5)
    days = st.sidebar.number_input("Days Before", 0, 365, 14)
    dep = st.sidebar.date_input("Departure", datetime.now() + timedelta(14), min_value=datetime.now())
    arr = st.sidebar.date_input("Arrival", datetime.now() + timedelta(14), min_value=dep)

    return {
        "Airline": airline, "Source": source, "Destination": dest, "Stopovers": stopovers,
        "Aircraft_Type": aircraft, "Class_": travel_class, "Booking_Source": booking,
        "Season": season, "Duration_hrs": duration, "Days_Before_Departure": days,
        "Departure_Date": str(dep), "Arrival_Date": str(arr)
    }, source_names.get(source, source), dest_names.get(dest, dest)

data, src, dst = user_input()

st.subheader("Summary")
col1, col2, col3,col4, col5= st.columns(5)
with col1:
    st.metric("From", src)
    st.write(f"**Airline:** {data['Airline']}")
with col2:
    st.metric("To", dst)
    st.write(f"**Class:** {data['Class_']}")
with col3:
    st.metric("Days Before", data['Days_Before_Departure'])
with col4:
    st.metric("Departure Date", dst)
    st.write(f"**Departure Date** {data['Class_']}")
with col5:
    st.metric("Arriving on ", dst)
    st.write(f"**Arrival Date** {data['Class_']}")

if st.button("Predict", type="primary", use_container_width=True):
    try:
        with st.spinner("Predicting..."):
            res = requests.post(API_URL, json=data, timeout=10)
        if res.status_code == 200:
            price = res.json()["predicted_total_fare"]
            st.success("Complete!")
            st.markdown(f"<h1 style='text-align:center;color:#1f77b4'>৳ {price:,.2f} BDT</h1>", unsafe_allow_html=True)
            st.info(f"**Range:** ৳{max(0, price-28465):,.2f} - ৳{price+28465:,.2f}")
        else:
            st.error(f"Error: {res.json().get('detail')}")
    except Exception as e:
        st.error(f"Error: {e}")