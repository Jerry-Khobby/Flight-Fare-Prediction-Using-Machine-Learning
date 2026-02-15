from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json 
from monitoring.model_monitor import ModelMonitor 

app = FastAPI(title="Flight Fare Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = ModelMonitor(monitoring_dir="monitoring/data")

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
    Departure_Date: str
    Arrival_Date: str

# Load model
try:
    model = joblib.load("../modelling/outputs/models/flight_price_model.pkl")
    scaler = joblib.load("../modelling/outputs/models/scaler.pkl")
    feature_columns = joblib.load("../modelling/outputs/models/feature_columns.pkl")
    model_loaded = True
    print(f"Model loaded - {len(feature_columns)} features")
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    model_loaded = False
    print(f"Error: {e}")

def preprocess_input(data: dict):
    """Match training preprocessing EXACTLY"""
    df = pd.DataFrame([data])
    
    # Rename columns
    df.rename(columns={
        'Class_': 'Class',
        'Duration_hrs': 'Duration (hrs)',
        'Booking_Source': 'Booking Source',
        'Departure_Date': 'Departure Date & Time',
        'Arrival_Date': 'Arrival Date & Time',
        'Aircraft_Type': 'Aircraft Type',
        'Days_Before_Departure': 'Days Before Departure'
    }, inplace=True)
    
    # Ordinal encoding
    df["Stopovers"] = df["Stopovers"].map({"Direct": 0, "1 Stop": 1, "2 Stops": 2})
    df["Class"] = df["Class"].map({"Economy": 0, "Business": 1, "First Class": 2})
    df["Booking Source"] = df["Booking Source"].map({
        "Online Website": 0, "Travel Agency": 1, "Direct Booking": 2
    })
    
    # Date features
    df['Departure Date & Time'] = pd.to_datetime(df['Departure Date & Time'])
    df['Arrival Date & Time'] = pd.to_datetime(df['Arrival Date & Time'])
    
    df['Departure Month'] = df['Departure Date & Time'].dt.month
    df['Departure Day'] = df['Departure Date & Time'].dt.day
    df['Departure Weekday'] = df['Departure Date & Time'].dt.weekday
    df['Arrival Month'] = df['Arrival Date & Time'].dt.month
    df['Arrival Day'] = df['Arrival Date & Time'].dt.day
    df['Arrival Weekday'] = df['Arrival Date & Time'].dt.weekday
    
    df.drop(columns=['Departure Date & Time', 'Arrival Date & Time'], inplace=True)
    
    # One-hot encoding (drop_first=False to match training)
    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Align columns with training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    
    # Scale (all columns)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns,
        index=df.index
    )
    
    return df_scaled

@app.get("/")
def home():
    return {"message": "Flight Fare API", "status": "ready" if model_loaded else "not_ready"}

@app.post("/predict")
def predict_fare(flight: FlightInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        raw_input  = flight.dict()
        
        #drift detection before processing
        drift_result = monitor.detect_drift(raw_input)
        
        #Save drift log 
        drift_log_file = monitor.monitoring_dir/"drift_log.jsonl"
        with open(drift_log_file,"a") as f: 
            f.write(json.dumps(drift_result) +"\n")
            
            
            
        processed = preprocess_input(flight.dict())
        y_pred_log = model.predict(processed)
        y_pred = np.expm1(y_pred_log)
        
        return {
            "predicted_total_fare": float(np.round(y_pred[0], 2)),
            "currency": "BDT",
            "status": "success"
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.get("/monitoring/metrics")
def get_metrics(days: int = 7):
    """Get model performance metrics"""
    return monitor.get_performance_metrics(days=days)

@app.get("/monitoring/drift")
def get_drift_summary(days: int = 7):
    """Get drift detection summary"""
    return monitor.get_drift_summary(days=days)



@app.get("/monitoring/health")
def health_check():
    metrics = monitor.get_performance_metrics(days=7)
    drift = monitor.get_drift_summary(days=7)

    mae_threshold = 35000
    drift_rate_threshold = 0.1

    health_status = "healthy"
    warnings = []

    mae_value = metrics.get("mae", 0) or 0
    drift_rate_value = drift.get("drift_rate", 0) or 0

    if mae_value > mae_threshold:
        health_status = "degraded"
        warnings.append(f"MAE ({mae_value:.2f}) exceeds threshold ({mae_threshold})")

    if drift_rate_value > drift_rate_threshold:
        health_status = "degraded"
        warnings.append(f"Drift rate ({drift_rate_value:.2%}) exceeds threshold ({drift_rate_threshold:.0%})")

    return {
        "status": health_status,
        "model_loaded": model_loaded,
        "recent_mae": mae_value,
        "drift_rate": drift_rate_value,
        "warnings": warnings,
        "timestamp": pd.Timestamp.now().isoformat()
    }
