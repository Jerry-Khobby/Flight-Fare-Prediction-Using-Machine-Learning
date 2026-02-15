import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from scipy import stats

class ModelMonitor:
    """Simple production-grade model monitoring"""
    
    def __init__(self, monitoring_dir="monitoring/data"):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training statistics
        self.train_stats = self._load_or_create_stats()
        
    def _load_or_create_stats(self):
        """Load training data statistics"""
        stats_file = self.monitoring_dir / "train_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def log_prediction(self, input_data, prediction, actual=None):
        """Log a prediction for monitoring"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction": float(prediction),
            "actual": float(actual) if actual is not None else None
        }
        
        # Append to predictions log
        log_file = self.monitoring_dir / f"predictions_{datetime.now().strftime('%Y%m')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def detect_drift(self, input_data):
        """Detect feature drift using statistical tests"""
        if not self.train_stats:
          return {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
          "features_with_drift": [],
        "message": "No baseline stats available"
        }
        
        numeric_features = ['Duration_hrs', 'Days_Before_Departure']
        
        for feature in numeric_features:
            if feature in input_data and feature in self.train_stats:
                value = input_data[feature]
                mean = self.train_stats[feature]['mean']
                std = self.train_stats[feature]['std']
                
                # Z-score drift detection
                z_score = abs((value - mean) / std) if std > 0 else 0
                
                if z_score > 3:  # 3 standard deviations
                    drift_report["drift_detected"] = True
                    drift_report["features_with_drift"].append({
                        "feature": feature,
                        "value": value,
                        "expected_mean": mean,
                        "z_score": z_score
                    })
        
        return drift_report
    
    def get_performance_metrics(self, days=7):
        """Calculate recent performance metrics"""
        predictions = self._load_recent_predictions(days)
        
        if len(predictions) == 0:
            return {"error": "No predictions found"}
        
        # Filter predictions with actual values
        actual_predictions = [p for p in predictions if p['actual'] is not None]
        
        if len(actual_predictions) == 0:
            return {
                "total_predictions": len(predictions),
                "predictions_with_actuals": 0,
                "message": "No actual values available for comparison"
            }
        
        actuals = [p['actual'] for p in actual_predictions]
        preds = [p['prediction'] for p in actual_predictions]
        
        # Calculate metrics
        mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
        mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100
        
        return {
            "period_days": days,
            "total_predictions": len(predictions),
            "predictions_with_actuals": len(actual_predictions),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "timestamp": datetime.now().isoformat()
        }
    
    def _load_recent_predictions(self, days=7):
        """Load predictions from recent days"""
        predictions = []
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        for log_file in self.monitoring_dir.glob("predictions_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    if entry_date >= cutoff_date:
                        predictions.append(entry)
        
        return predictions
    
    def get_drift_summary(self, days=7):
        """Get summary of drift detections"""
        drift_file = self.monitoring_dir / "drift_log.jsonl"
        if not drift_file.exists():
            return {"total_checks": 0, "drift_detected": 0}
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        total_checks = 0
        drift_detected = 0
        
        with open(drift_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if "timestamp" not in entry:
                  continue 
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if entry_date >= cutoff_date:
                    total_checks += 1
                    if entry['drift_detected']:
                        drift_detected += 1
        
        return {
            "period_days": days,
            "total_checks": total_checks,
            "drift_detected": drift_detected,
            "drift_rate": drift_detected / total_checks if total_checks > 0 else 0
        }