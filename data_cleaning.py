import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from logging_config import get_logger

logger = get_logger(log_file="logs/data_cleaning.log")

def preprocess_flight_data(df, target="Total Fare", test_size=0.2, random_state=42, save_csv=True):
    """
    Preprocess flight dataset:
      - Handle missing values
      - Correct invalid entries (negative fares, inconsistent city names)
      - Feature engineering (Total Fare, departure/arrival dates)
      - Encode categorical variables (ordinal + one-hot)
      - Scale numerical features
      - Split data into train and test sets
      - Save cleaned data for KPI stage
    
    Returns:
        X_train, X_test, y_train, y_test, df_clean
    """

    df = df.copy()
    logger.info("Started preprocessing flight dataset")

    numerical_cols = ['Duration (hrs)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)',
                      'Total Fare (BDT)', 'Days Before Departure']
    categorical_cols = ['Airline', 'Source', 'Source Name', 'Destination',
                        'Destination Name', 'Stopovers', 'Aircraft Type',
                        'Class', 'Booking Source', 'Seasonality']
    date_cols = ['Departure Date & Time', 'Arrival Date & Time']

    # Handle missing values
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled missing numerical values in {col} with median {median_val}")

    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)
            logger.info(f"Filled missing categorical values in {col} with 'Unknown'")

    # Correct invalid fare entries
    fare_cols = ['Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Total Fare (BDT)']
    for col in fare_cols:
        if col in df.columns:
            before_rows = df.shape[0]
            df = df[df[col] >= 0]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_rows = df.shape[0]
            logger.info(f"Removed {before_rows - after_rows} rows with negative values in {col}")

    # Normalize city names
    city_columns = ['Source', 'Destination']
    for col in city_columns:
        if col in df.columns:
            df[col] = df[col].str.upper()
            logger.info(f"Normalized city names in {col} to uppercase")

    # Convert dates
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"Converted {col} to datetime")

    # Feature engineering: Total Fare and date features
    if all(col in df.columns for col in ['Base Fare (BDT)', 'Tax & Surcharge (BDT)']):
        df['Total Fare (BDT)'] = df['Base Fare (BDT)'] + df['Tax & Surcharge (BDT)']
        logger.info("Calculated Total Fare as Base Fare + Tax & Surcharge")

    df['Departure Month'] = df['Departure Date & Time'].dt.month
    df['Departure Day'] = df['Departure Date & Time'].dt.day
    df['Departure Weekday'] = df['Departure Date & Time'].dt.weekday
    df['Arrival Month'] = df['Arrival Date & Time'].dt.month
    df['Arrival Day'] = df['Arrival Date & Time'].dt.day
    df['Arrival Weekday'] = df['Arrival Date & Time'].dt.weekday
    logger.info("Extracted departure and arrival month, day, and weekday features")

    # Rename columns
    df.rename(columns={
        'Total Fare (BDT)': 'Total Fare',
        'Base Fare (BDT)': 'Base Fare',
        'Tax & Surcharge (BDT)': 'Tax & Surcharge',
        'Seasonality': 'Season'
    }, inplace=True)

    # Save the new dataset for KPI stage
    if save_csv:
        os.makedirs("data", exist_ok=True)
        csv_path = os.path.join("data", "cleaned_flight_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved cleaned dataset to {csv_path} for KPI stage")

    # Encode categorical variables
    df.drop(columns=['Source Name', 'Destination Name'], inplace=True, errors='ignore')
    df["Stopovers"] = df["Stopovers"].replace({"Direct": 0, "1 Stopover": 1, "2 Stops": 2})
    df["Class"] = df["Class"].replace({"Economy": 0, "First Class": 1, "Business": 2})
    df["Booking Source"] = df["Booking Source"].replace({"Online Website": 0, "Travel Agency": 1, "Direct Booking": 2})
    logger.info("Encoded Stopovers, Class, and Booking Source")

    nominal_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    logger.info(f"One-hot encoded nominal columns: {', '.join(nominal_cols)}")

    # Scale numerical features
    numerical_cols_scaled = ['Duration (hrs)', 'Base Fare', 'Tax & Surcharge', 'Total Fare', 'Days Before Departure']
    scaler = StandardScaler()
    df[numerical_cols_scaled] = scaler.fit_transform(df[numerical_cols_scaled])
    logger.info("Scaled numerical features with StandardScaler")

    # Train-test split
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Performed train-test split with test_size={test_size}")

    logger.info("Preprocessing complete")
    return X_train, X_test, y_train, y_test, df
