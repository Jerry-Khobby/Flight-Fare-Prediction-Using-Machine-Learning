import os
import joblib
import pandas as pd
from training.data_cleaning import validation, premodel_training
from training.data_kpis import perform_eda_kpis
from training.advanced_model import train_model
from training.logging_config import get_logger

# Main orchestrator logger
logger = get_logger(log_file="logs/main.log")


def main():
    """
    ML Pipeline Orchestrator
    Each step has its own log file:
    - logs/main.log (this file)
    - logs/data_cleaning.log (validation & preprocessing)
    - logs/kpi_exploration.log (EDA)
    - logs/model_training.log (training)
    """
    
    try:
        logger.info("="*60)
        logger.info("PIPELINE STARTED")
        logger.info("="*60)
        
        #  Load Raw Data
        logger.info("STEP 1: Loading raw data")
        df_raw = pd.read_csv("data/Flight_Price_Dataset_of_Bangladesh.csv")
        logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
        
        # Validation & Cleaning 
        logger.info("STEP 2: Validating and cleaning data (logs/data_cleaning.log)")
        df_clean = validation(df_raw, save_csv=True)
        logger.info(f"Validation complete: {len(df_clean)} rows after cleaning")
        
        #  Exploratory Data Analysis 
        logger.info("STEP 3: Performing EDA and KPIs (logs/kpi_exploration.log)")
        kpi_results = perform_eda_kpis(df_clean, save_plots=True)
        logger.info("EDA complete: plots saved to data/kpi-diagrams/")
        
        # Preprocessing for Model 
        logger.info("STEP 4: Preprocessing for training (logs/data_cleaning.log)")
        X_train, X_test, y_train, y_test, scaler = premodel_training(
            df_clean, 
            target="Total Fare",
            test_size=0.2,
            random_state=42
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Features: {len(X_train.columns)}")
        
        # Model Training 
        logger.info("STEP 5: Training models (logs/model_training.log)")
        best_model = train_model(X_train, y_train, X_test, y_test)
        logger.info("Model training complete")
        
        # Save Outputs 
        logger.info("STEP 6: Saving model artifacts")
        os.makedirs("outputs/models", exist_ok=True)
        
        model_path = "outputs/models/flight_price_model.pkl"
        scaler_path = "outputs/models/scaler.pkl"
        features_path = "outputs/models/feature_columns.pkl"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(X_train.columns.tolist(), features_path)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Scaler saved: {scaler_path}")
        logger.info(f"Features saved: {features_path}")
        
        #  Summary 
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("Summary:")
        logger.info(f"  - Data rows: {len(df_clean)}")
        logger.info(f"  - Features: {len(X_train.columns)}")
        logger.info(f"  - Model type: {type(best_model).__name__}")
        logger.info(f"  - Outputs: outputs/models/")
        logger.info(f"  - Logs: logs/")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error("="*60)
        logger.error("PIPELINE FAILED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error("", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)