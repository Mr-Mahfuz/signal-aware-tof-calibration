import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle

def create_output_dir():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Filter trainable points
    trainable_df = df[(df['valid_return'] == 1) & (df['gt_distance_mm'].notna())].copy()
    
    # Drop rows where critical features are NaN
    features = ['raw_distance_mm', 'pan_calibrated_deg', 'tilt_calibrated_deg', 
                'signal_rate_kcps', 'ambient_rate_kcps', 'sigma_mm']
    
    initial_len = len(trainable_df)
    trainable_df = trainable_df.dropna(subset=features + ['residual_error_mm'])
    print(f"Filtered to {len(trainable_df)} points (dropped {initial_len - len(trainable_df)} due to NaNs).")
    
    X = trainable_df[features]
    y = trainable_df['residual_error_mm']
    
    return X, y, features

def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pct95 = np.percentile(np.abs(y_true - y_pred), 95)
    
    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.2f} mm")
    print(f"RMSE: {rmse:.2f} mm")
    print(f"95th: {pct95:.2f} mm")
    print(f"R2:   {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, '95th': pct95, 'R2': r2}

def plot_feature_importance(model, feature_names, out_dir):
    # Works for tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances (Gradient Boosted Trees)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
        plt.close()
        print("Saved feature_importance.png")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_dataset.csv')
    out_dir = create_output_dir()
    
    X, y, feature_names = load_data(data_path)
    
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Baseline: Raw Sensor (No correction)
    # y_pred = 0 means we assume residual error is 0, i.e., raw = ground truth
    y_pred_baseline = np.zeros_like(y_test)
    results['Raw Sensor'] = evaluate_model("Raw Sensor (No Calibration)", y_test, y_pred_baseline)
    
    # 2. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    results['Linear Regression'] = evaluate_model("Linear Regression", y_test, y_pred_lr)
    
    # 3. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf)
    
    # 4. XGBoost (Gradient Boosted Trees)
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model("XGBoost", y_test, y_pred_xgb)
    
    plot_feature_importance(xgb_model, feature_names, out_dir)
    
    # Save best model (XGBoost) for later evaluation
    model_path = os.path.join(out_dir, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"\nSaved XGBoost model to {model_path}")
    
    # Also save Random Forest model as it's easier to transpile to C via m2cgen
    rf_path = os.path.join(out_dir, 'rf_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Saved Random Forest model to {rf_path}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Training complete!")
