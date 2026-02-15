import pandas as pd
import json
from pathlib import Path

def save_training_stats(X_train, output_dir="monitoring/data"):
    """Save training data statistics for drift detection"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Numeric features
    numeric_features = ['Duration (hrs)', 'Days Before Departure']
    
    for col in numeric_features:
        if col in X_train.columns:
            stats[col.replace(' ', '_').replace('(', '').replace(')', '')] = {
                'mean': float(X_train[col].mean()),
                'std': float(X_train[col].std()),
                'min': float(X_train[col].min()),
                'max': float(X_train[col].max()),
                'q25': float(X_train[col].quantile(0.25)),
                'q75': float(X_train[col].quantile(0.75))
            }
    
    with open(Path(output_dir) / 'train_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training stats saved to {output_dir}/train_stats.json")

# Add to your main.py after training
if __name__ == "__main__":
    from monitoring.save_train_stats import save_training_stats
    # After X_train is created
    save_training_stats(X_train)