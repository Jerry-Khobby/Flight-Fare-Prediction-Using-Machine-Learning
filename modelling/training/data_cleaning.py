import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from training.logging_config import get_logger
import joblib 

logger = get_logger(log_file="logs/data_cleaning.log")


def preprocess_flight_data(df, target="Total Fare", test_size=0.2, random_state=42, save_csv=True):
    """
    Clean and preprocess flight dataset safely (no data leakage).
    Returns: X_train, X_test, y_train, y_test, df_clean, scaler
    """
    df = df.copy()
    logger.info("Started preprocessing")

    # Rename columns
    df.rename(columns={
        'Total Fare (BDT)': 'Total Fare',
        'Base Fare (BDT)': 'Base Fare',
        'Tax & Surcharge (BDT)': 'Tax & Surcharge',
        'Seasonality': 'Season'
    }, inplace=True)

    # Convert numeric columns
    numeric_cols = ['Duration (hrs)', 'Base Fare', 'Tax & Surcharge', 'Total Fare', 'Days Before Departure']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove negative fares
    fare_cols = ['Base Fare', 'Tax & Surcharge', 'Total Fare']
    for col in fare_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Handle missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = ['Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 
                       'Class', 'Booking Source', 'Season']
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

    # Ordinal encoding (use .map() to avoid warnings)
    df["Stopovers"] = df["Stopovers"].map({"Direct": 0, "1 Stop": 1, "2 Stops": 2})
    df["Class"] = df["Class"].map({"Economy": 0, "Business": 1, "First Class": 2})
    df["Booking Source"] = df["Booking Source"].map({
        "Online Website": 0, "Travel Agency": 1, "Direct Booking": 2
    })

    # CRITICAL FIX: Use drop_first=False for consistency
    # This ensures every category has a column, making prediction easier
    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    
    # Convert all to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    # Separate target
    X = df.drop(columns=[target])
    y = df[target]

    # Remove Base Fare & Tax (target leakage)
    if target == "Total Fare":
        X = X.drop(columns=["Base Fare", "Tax & Surcharge"], errors="ignore")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale ALL features (including one-hot encoded)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Save cleaned dataset
    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cleaned_flight_data.csv", index=False)

    logger.info("Preprocessing complete")
    return X_train_scaled, X_test_scaled, y_train, y_test, df, scaler