import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from training.logging_config import get_logger

logger = get_logger(log_file="logs/data_cleaning.log")


def validation(df, save_csv=True, required_columns=None):
    """Validate and clean the raw flight dataset."""
    df = df.copy()
    logger.info("="*50)
    logger.info("VALIDATION STARTED")
    logger.info("="*50)
    
    if required_columns is None:
        required_columns = [
            'Airline', 'Source', 'Source Name', 'Destination', 'Destination Name',
            'Departure Date & Time', 'Arrival Date & Time', 'Duration (hrs)',
            'Stopovers', 'Aircraft Type', 'Class', 'Booking Source',
            'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Total Fare (BDT)',
            'Seasonality', 'Days Before Departure'
        ]
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info(f"All {len(required_columns)} required columns present")
    
    # Rename columns
    df.rename(columns={
        'Total Fare (BDT)': 'Total Fare',
        'Base Fare (BDT)': 'Base Fare',
        'Tax & Surcharge (BDT)': 'Tax & Surcharge',
        'Seasonality': 'Season'
    }, inplace=True)
    logger.info("Columns renamed")
    
    # Numeric conversion
    numeric_cols = ['Duration (hrs)', 'Base Fare', 'Tax & Surcharge', 'Total Fare', 'Days Before Departure']
    for col in numeric_cols:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after = df[col].isna().sum()
            if after > before:
                logger.warning(f"{col}: {after - before} values coerced to NaN")
    
    # Remove negative fares
    initial_rows = len(df)
    fare_cols = ['Base Fare', 'Tax & Surcharge', 'Total Fare']
    for col in fare_cols:
        if col in df.columns:
            df = df[df[col] >= 0]
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with negative fares")
    
    # Fill missing values
    for col in numeric_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col].fillna(df[col].median(), inplace=True)
                logger.info(f"{col}: filled {missing} missing values with median")
    
    categorical_cols = ['Airline', 'Source', 'Destination', 'Stopovers', 
                       'Aircraft Type', 'Class', 'Booking Source', 'Season']
    for col in categorical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col].fillna("Unknown", inplace=True)
                logger.info(f"{col}: filled {missing} missing values with 'Unknown'")
    
    # Date features
    date_cols = ['Departure Date & Time', 'Arrival Date & Time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'Departure Date & Time' in df.columns:
        df['Departure Month'] = df['Departure Date & Time'].dt.month
        df['Departure Day'] = df['Departure Date & Time'].dt.day
        df['Departure Weekday'] = df['Departure Date & Time'].dt.weekday
        logger.info("Departure date features extracted")
    
    if 'Arrival Date & Time' in df.columns:
        df['Arrival Month'] = df['Arrival Date & Time'].dt.month
        df['Arrival Day'] = df['Arrival Date & Time'].dt.day
        df['Arrival Weekday'] = df['Arrival Date & Time'].dt.weekday
        logger.info("Arrival date features extracted")
    
    df.drop(columns=date_cols, inplace=True, errors='ignore')
    
    # Save
    if save_csv:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cleaned_flight_data.csv", index=False)
        logger.info("Saved: data/cleaned_flight_data.csv")
    
    logger.info(f"Validation complete: {len(df)} rows, {len(df.columns)} columns")
    logger.info("="*50)
    return df


def premodel_training(df, target="Total Fare", test_size=0.2, random_state=42):
    """Preprocess for model training."""
    df = df.copy()
    logger.info("="*50)
    logger.info("PREPROCESSING STARTED")
    logger.info("="*50)
    
    # Ordinal encoding
    df["Stopovers"] = df["Stopovers"].map({"Direct": 0, "1 Stop": 1, "2 Stops": 2})
    df["Class"] = df["Class"].map({"Economy": 0, "Business": 1, "First Class": 2})
    df["Booking Source"] = df["Booking Source"].map({
        "Online Website": 0, "Travel Agency": 1, "Direct Booking": 2
    })
    logger.info("Ordinal encoding complete")
    
    # One-hot encoding
    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    logger.info(f"One-hot encoding: {len(nominal_cols)} columns expanded")
    
    # Convert to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Separate target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Remove target leakage
    if target == "Total Fare":
        X = X.drop(columns=["Base Fare", "Tax & Surcharge"], errors="ignore")
        logger.info("Removed Base Fare & Tax & Surcharge (target leakage)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    # Scale
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
    logger.info(f"Scaling complete: {len(X_train.columns)} features")
    logger.info("="*50)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler