from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = FastAPI(title="Flight Fare Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Input Schema
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Stopovers: str
    Aircraft_Type: str
    Class_: str
    Booking_Source: str
    Season: str
    Duration_hrs: float
    Days_Before_Departure: int
    Departure_Date: str  # YYYY-MM-DD
    Arrival_Date: str    # YYYY-MM-DD



# Load Model & Artifacts
try:
    model = joblib.load("../flight_price_model.pkl")
    scaler = joblib.load("../scaler.pkl")
    feature_columns = joblib.load("../feature_columns.pkl")
    
    print(feature_columns[:500])
    
    airline_cols = [c for c in feature_columns if c.startswith("Airline_")]
    aircraft_cols = [c for c in feature_columns if c.startswith("Aircraft Type_")]
    source_cols = [c for c in feature_columns if c.startswith("Source_")]
    destination_cols = [c for c in feature_columns if c.startswith("Destination_")]
    season_cols = [c for c in feature_columns if c.startswith("Season_")]

    print("Airlines in training:", [c.replace("Airline_", "") for c in airline_cols])
    print("Aircraft types in training:", [c.replace("Aircraft Type_", "") for c in aircraft_cols])
    print("Sources in training:", [c.replace("Source_", "") for c in source_cols])
    print("Destinations in training:", [c.replace("Destination_", "") for c in destination_cols])
    print("Seasons in training:", [c.replace("Season_", "") for c in season_cols])
    model_loaded = True
    print("Model, scaler, and features loaded successfully")
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    model_loaded = False
    print(f"Error loading model: {e}")


# -----------------------
# Preprocess Input (matches training exactly)
# -----------------------
def preprocess_input(data: dict):
    """
    Preprocess user input to match training pipeline exactly.
    """
    df = pd.DataFrame([data])
    
    # Step 1: Rename columns to match training
    df.rename(columns={
        'Class_': 'Class',
        'Duration_hrs': 'Duration (hrs)',
        'Booking_Source': 'Booking Source',
        'Departure_Date': 'Departure Date & Time',
        'Arrival_Date': 'Arrival Date & Time',
        'Aircraft_Type': 'Aircraft Type',
        'Days_Before_Departure': 'Days Before Departure'
    }, inplace=True)
    
    # Step 2: Ordinal encoding (same as training)
    df["Stopovers"] = df["Stopovers"].replace({
        "Direct": 0, 
        "1 Stop": 1, 
        "2 Stops": 2
    })
    
    df["Class"] = df["Class"].replace({
        "Economy": 0, 
        "Business": 1, 
        "First Class": 2
    })
    
    df["Booking Source"] = df["Booking Source"].replace({
        "Online Website": 0, 
        "Travel Agency": 1, 
        "Direct Booking": 2
    })
    
    # Step 3: Date feature engineering
    df['Departure Date & Time'] = pd.to_datetime(df['Departure Date & Time'])
    df['Arrival Date & Time'] = pd.to_datetime(df['Arrival Date & Time'])
    
    df['Departure Month'] = df['Departure Date & Time'].dt.month
    df['Departure Day'] = df['Departure Date & Time'].dt.day
    df['Departure Weekday'] = df['Departure Date & Time'].dt.weekday
    df['Arrival Month'] = df['Arrival Date & Time'].dt.month
    df['Arrival Day'] = df['Arrival Date & Time'].dt.day
    df['Arrival Weekday'] = df['Arrival Date & Time'].dt.weekday
    
    df.drop(columns=['Departure Date & Time', 'Arrival Date & Time'], inplace=True)
    
    # Step 4: One-hot encoding (must match training)
    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Step 5: Convert all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 6: Align columns with training features
    # Add missing columns (if any) with value 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only columns from training (in same order)
    df = df[feature_columns]
    
    # Step 7: Scale (same as training)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    return df



# API Endpoints
@app.get("/")
def home():
    """Health check endpoint"""
    status = "Ready" if model_loaded else "Model not loaded"
    return {
        "message": "Flight Fare Prediction API",
        "status": status,
        "model_type": "Gradient Boosting",
        "expected_r2": "0.66"
    }


@app.post("/predict")
def predict_fare(flight: FlightInput):
    """
    Predict flight fare based on input features.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Train and save model first."
        )
    
    try:
        # Preprocess input
        processed = preprocess_input(flight.dict())
        
        # Predict (model outputs log-transformed values)
        y_pred_log = model.predict(processed)
        y_pred = np.expm1(y_pred_log)  # Convert back from log
        
        return {
            "predicted_total_fare": float(np.round(y_pred[0], 2)),
            "currency": "BDT",
            "model": "Gradient Boosting (Tuned)",
            "status": "success"
        }
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(tb_str)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}\nFull traceback:\n{tb_str}"
        )


@app.get("/model-info")
def model_info():
    """Get model metadata"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": type(model).__name__,
        "num_features": len(feature_columns),
        "feature_names": feature_columns[:10],  # First 10 features
        "scaler_type": type(scaler).__name__
    }