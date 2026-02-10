import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from training.logging_config import get_logger
import joblib 

logger = get_logger(log_file="logs/data_cleaning.log")


def preprocess_flight_data(df, target="Total Fare",test_size=0.2,random_state=42,save_csv=True):
    """
    Clean and preprocess flight dataset safely (no data leakage).

    Returns:
        X_train, X_test, y_train, y_test
    """

    df = df.copy()
    logger.info("Started preprocessing")

    #Rename columns first (avoid confusion)
    df.rename(columns={
        'Total Fare (BDT)': 'Total Fare',
        'Base Fare (BDT)': 'Base Fare',
        'Tax & Surcharge (BDT)': 'Tax & Surcharge',
        'Seasonality': 'Season'
    }, inplace=True)

    #Convert numeric columns safely
    numeric_cols = [
        'Duration (hrs)',
        'Base Fare',
        'Tax & Surcharge',
        'Total Fare',
        'Days Before Departure'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #Remove negative fares
    fare_cols = ['Base Fare', 'Tax & Surcharge', 'Total Fare']
    for col in fare_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Handle missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = [
        'Airline', 'Source', 'Destination',
        'Stopovers', 'Aircraft Type',
        'Class', 'Booking Source', 'Season'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)

    # Date Feature Engineering
    date_cols = ['Departure Date & Time', 'Arrival Date & Time']

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'Departure Date & Time' in df.columns:
        df['Departure Month'] = df['Departure Date & Time'].dt.month
        df['Departure Day'] = df['Departure Date & Time'].dt.day
        df['Departure Weekday'] = df['Departure Date & Time'].dt.weekday

    if 'Arrival Date & Time' in df.columns:
        df['Arrival Month'] = df['Arrival Date & Time'].dt.month
        df['Arrival Day'] = df['Arrival Date & Time'].dt.day
        df['Arrival Weekday'] = df['Arrival Date & Time'].dt.weekday

    df.drop(columns=date_cols, inplace=True, errors='ignore')

    #Encode categorical variables

    # Ordinal encoding
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

    # One-hot encoding
    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col],errors="coerce")
    df = df.fillna(0)

    #Separate target BEFORE scaling
    X = df.drop(columns=[target])
    y = df[target]

    # Remove Base Fare & Tax if predicting Total Fare
    if target == "Total Fare":
        X = X.drop(columns=["Base Fare", "Tax & Surcharge"], errors="ignore")


    #Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # STEP 9: Scale ONLY feature columns (not target)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    
    
    X_train.loc[:,numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test.loc[:,numeric_features]  = scaler.transform(X_test[numeric_features])

    #Save cleaned dataset (optional)
    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cleaned_flight_data.csv", index=False)

    logger.info("Preprocessing complete")

    return X_train, X_test, y_train, y_test,df,scaler
