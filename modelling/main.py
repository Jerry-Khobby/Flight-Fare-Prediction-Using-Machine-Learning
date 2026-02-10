import os
import joblib
import pandas as pd
from training.data_cleaning import preprocess_flight_data
from training.advanced_model import train_model
from training.logging_config import get_logger

logger = get_logger(log_file="logs/main.log")


def main():

    logger.info("Pipeline started")

    df = pd.read_csv("data/Flight_Price_Dataset_of_Bangladesh.csv")

    X_train, X_test, y_train, y_test, df_clean, scaler = preprocess_flight_data(df)

    best_model = train_model(X_train, y_train, X_test, y_test)

    os.makedirs("outputs/models", exist_ok=True)

    joblib.dump(best_model, "flight_price_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

    logger.info("Model and scaler saved successfully")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
