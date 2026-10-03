"""
train_day2.py — Enhanced ML training pipeline for Day 1 + Day 2 LiDAR data.

Trains multiple ML models across three scenarios:
  1. Day 2 only (80/20 train/test split)
  2. Day 1 → Day 2 (cross-day generalization test)
  3. Day 1 + Day 2 combined (unified model)

Models: Linear Regression, Random Forest, XGBoost, Gradient Boosting, MLP Neural Net.

Also performs per-condition accuracy analysis and generates comparison plots.

Output:
  - ml_pipeline/output_day2/  — models, metrics, plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY2_CSV = os.path.join(BASE_DIR, "datasets", "day2_combined.csv")
ALL_CSV = os.path.join(BASE_DIR, "datasets", "all_days_combined.csv")
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_day2")

# Feature columns for Day 2 data (sigma_mm is always -1, so excluded)
FEATURES_DAY2 = ['tof_mm', 'pan_deg', 'tilt_deg', 'signal_rate_kcps', 'ambient_rate_kcps', 'env_temp_C']
TARGET = 'residual_error_mm'


def create_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)


def load_day2_data():
    """Load and prepare Day 2 dataset for training."""
    print("Loading Day 2 data...")
    df = pd.read_csv(DAY2_CSV, low_memory=False)
    
    # Filter trainable: valid return + has ground truth + not extreme tilt
    mask = (df['valid_return'] == 1) & (df['residual_error_mm'].notna())
    trainable = df[mask].copy()
    
    # Drop rows with NaN in features
    trainable = trainable.dropna(subset=FEATURES_DAY2 + [TARGET])
    
    print(f"  Total rows: {len(df)}")
    print(f"  Trainable rows: {len(trainable)}")
    print(f"  Conditions: {trainable['condition_label'].value_counts().to_dict()}")
    
    return trainable


def load_day1_data_for_training():
    """Load Day 1 processed data from raw_dataset.csv for cross-day testing."""
    if not os.path.exists(RAW_DATASET):
        print("  raw_dataset.csv not found — skipping Day 1")
        return None
    
    print("Loading Day 1 data (raw_dataset.csv)...")
    df = pd.read_csv(RAW_DATASET, low_memory=False)
    
    # Map column names to match Day 2 schema
    rename_map = {
        'raw_distance_mm': 'tof_mm',
        'pan_calibrated_deg': 'pan_deg',
        'tilt_calibrated_deg': 'tilt_deg',
        'temperature_C': 'env_temp_C',
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Use pan_commanded_deg if pan_calibrated_deg wasn't available
    if 'pan_deg' not in df.columns:
        if 'pan_commanded_deg' in df.columns:
            df['pan_deg'] = df['pan_commanded_deg']
    if 'tilt_deg' not in df.columns:
        if 'tilt_commanded_deg' in df.columns:
            df['tilt_deg'] = df['tilt_commanded_deg']
    
    # Ensure valid_return
    if 'valid_return' not in df.columns:
        df['valid_return'] = (df['tof_mm'] > 0).astype(int)
    
    # Compute residual if needed
    if 'residual_error_mm' not in df.columns:
        if 'gt_distance_mm' in df.columns:
            df['residual_error_mm'] = df['gt_distance_mm'] - df['tof_mm']
    
    # Filter trainable
    mask = (df['valid_return'] == 1) & (df['residual_error_mm'].notna())
    trainable = df[mask].copy()
    
    # Check which features we have
    available_features = [f for f in FEATURES_DAY2 if f in trainable.columns]
    missing = [f for f in FEATURES_DAY2 if f not in trainable.columns]
    if missing:
        print(f"  Warning: Missing features in Day 1: {missing}")
        for f in missing:
            trainable[f] = 0  # Fill with 0 for missing features
    
    trainable = trainable.dropna(subset=FEATURES_DAY2 + [TARGET])
    
    print(f"  Total rows: {len(df)}")
    print(f"  Trainable rows: {len(trainable)}")
    
    return trainable


def get_models():
    """Define the model suite."""
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, verbosity=0
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        ),
        'MLP Neural Net': MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.1
        ),
    }


def evaluate(model_name, y_true, y_pred):
    """Compute and return metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pct95 = np.percentile(np.abs(y_true - y_pred), 95)
    
    return {
        'MAE_mm': round(mae, 2),
        'RMSE_mm': round(rmse, 2),
        'R2': round(r2, 4),
        '95th_pct_mm': round(pct95, 2),
    }


def train_scenario(name, X_train, y_train, X_test, y_test, test_df=None):
    """Train all models for a given scenario and return results."""
    print(f"\n{'─' * 60}")
    print(f"SCENARIO: {name}")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"{'─' * 60}")
    
    results = {}
    trained_models = {}
    predictions = {}
    
    # Baseline: no correction (assume residual = 0)
    y_pred_baseline = np.zeros_like(y_test)
    results['Raw Sensor (No ML)'] = evaluate('Raw Sensor', y_test.values, y_pred_baseline)
    print(f"  Raw Sensor: MAE = {results['Raw Sensor (No ML)']['MAE_mm']:.2f} mm")
    
    models = get_models()
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...", end=" ", flush=True)
        
        try:
            # MLP needs scaled features
            if model_name == 'MLP Neural Net':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                trained_models[model_name] = (model, scaler)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                trained_models[model_name] = model
            
            metrics = evaluate(model_name, y_test.values, y_pred)
            results[model_name] = metrics
            predictions[model_name] = y_pred
            
            print(f"MAE = {metrics['MAE_mm']:.2f} mm, R² = {metrics['R2']:.4f}")
        
        except Exception as e:
            print(f"FAILED: {e}")
            results[model_name] = {'MAE_mm': float('nan'), 'error': str(e)}
    
    return results, trained_models, predictions


def plot_model_comparison(all_results, filename):
    """Bar chart comparing model MAEs across scenarios."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    scenarios = list(all_results.keys())
    models = set()
    for r in all_results.values():
        models.update(r.keys())
    models = sorted(models)
    
    x = np.arange(len(models))
    width = 0.8 / len(scenarios)
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    
    for i, (scenario, results) in enumerate(all_results.items()):
        maes = [results.get(m, {}).get('MAE_mm', 0) for m in models]
        bars = ax.bar(x + i * width, maes, width, label=scenario, color=colors[i])
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            if mae > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                       f'{mae:.1f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('MAE (mm)')
    ax.set_title('Model Comparison Across Training Scenarios')
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_feature_importance(model, feature_names, title, filename):
    """Plot feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Relative Importance')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_error_distribution_comparison(y_test, predictions, scenario_name, filename):
    """Plot error distributions for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Raw sensor errors
    ax = axes[0]
    ax.hist(y_test.values, bins=80, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f'Raw Sensor Error\nMAE={np.mean(np.abs(y_test.values)):.1f}mm')
    ax.set_xlabel('Error (mm)')
    
    for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
        if i >= len(axes):
            break
        ax = axes[i]
        residuals = y_test.values - y_pred
        ax.hist(residuals, bins=80, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        mae = np.mean(np.abs(residuals))
        ax.set_title(f'{model_name}\nMAE={mae:.1f}mm')
        ax.set_xlabel('Prediction Error (mm)')
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f'Error Distributions — {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_per_condition_accuracy(day2_data, best_model, model_name, features, filename):
    """Show MAE per scan condition for the best model."""
    conditions = day2_data['condition_label'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    raw_maes = []
    ml_maes = []
    labels = []
    
    for cond in sorted(conditions):
        subset = day2_data[day2_data['condition_label'] == cond]
        X_sub = subset[features]
        y_sub = subset[TARGET]
        
        if len(X_sub) < 10:
            continue
        
        # Raw sensor MAE
        raw_mae = np.mean(np.abs(y_sub.values))
        raw_maes.append(raw_mae)
        
        # ML-corrected MAE
        if isinstance(best_model, tuple):
            # MLP with scaler
            model, scaler = best_model
            y_pred = model.predict(scaler.transform(X_sub))
        else:
            y_pred = best_model.predict(X_sub)
        ml_mae = mean_absolute_error(y_sub, y_pred)
        ml_maes.append(ml_mae)
        
        labels.append(cond)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_maes, width, label='Raw Sensor', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, ml_maes, width, label=f'{model_name} (ML)', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('MAE (mm)')
    ax.set_title('Per-Condition Accuracy: Raw Sensor vs ML Calibration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
               f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
               f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_raw_vs_gt_scatter(data, filename):
    """Scatter plot of raw sensor distance vs ground truth."""
    valid = data[(data['valid_return'] == 1) & (data['gt_distance_mm'].notna())]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(valid['gt_distance_mm'], valid['tof_mm'],
                        c=valid['signal_rate_kcps'], cmap='viridis',
                        alpha=0.3, s=5)
    
    max_val = max(valid['gt_distance_mm'].max(), valid['tof_mm'].max())
    min_val = min(valid['gt_distance_mm'].min(), valid['tof_mm'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (1:1)', linewidth=2)
    
    ax.set_xlabel('Ground Truth Distance (mm)')
    ax.set_ylabel('Raw Sensor Distance (mm)')
    ax.set_title('Day 2: Raw Distance vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Signal Rate (kcps)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_error_heatmap(data, filename):
    """Heatmap of mean absolute error over pan-tilt grid."""
    valid = data[(data['valid_return'] == 1) & (data['residual_error_mm'].notna())]
    
    pivot = valid.pivot_table(
        values='residual_error_mm',
        index='tilt_deg',
        columns='pan_deg',
        aggfunc=lambda x: np.mean(np.abs(x))
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'MAE (mm)'})
    ax.set_title('Day 2: Mean Absolute Error Over Pan-Tilt Grid')
    ax.set_xlabel('Pan Angle (°)')
    ax.set_ylabel('Tilt Angle (°)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    create_dirs()
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    model_dir = os.path.join(OUTPUT_DIR, "models")
    
    # Load data
    day2_data = load_day2_data()
    day1_data = load_day1_data_for_training()
    
    all_results = {}
    
    # ─── Scenario 1: Day 2 Only (80/20 split) ─────────────────────────
    X_d2 = day2_data[FEATURES_DAY2]
    y_d2 = day2_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X_d2, y_d2, test_size=0.2, random_state=42
    )
    
    results_d2, models_d2, preds_d2 = train_scenario(
        "Day 2 Only", X_train, y_train, X_test, y_test
    )
    all_results["Day 2 Only"] = results_d2
    
    # Save best model from Day 2
    best_model_name = min(
        [(k, v['MAE_mm']) for k, v in results_d2.items() 
         if k != 'Raw Sensor (No ML)' and not np.isnan(v.get('MAE_mm', float('nan')))],
        key=lambda x: x[1]
    )[0]
    best_model = models_d2[best_model_name]
    
    # Save all trained models
    for mname, model in models_d2.items():
        safe_name = mname.lower().replace(' ', '_').replace('(', '').replace(')', '')
        with open(os.path.join(model_dir, f"day2_{safe_name}.pkl"), 'wb') as f:
            pickle.dump(model, f)
    
    # Plots for Day 2
    plot_error_distribution_comparison(
        y_test, preds_d2, "Day 2 Only",
        os.path.join(plot_dir, "day2_error_distributions.png")
    )
    
    # Feature importance for tree models
    for mname in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        if mname in models_d2:
            m = models_d2[mname]
            plot_feature_importance(
                m, FEATURES_DAY2,
                f'Feature Importance — {mname} (Day 2)',
                os.path.join(plot_dir, f"day2_feature_importance_{mname.lower().replace(' ', '_')}.png")
            )
    
    # Per-condition accuracy
    plot_per_condition_accuracy(
        day2_data, models_d2.get('XGBoost', best_model),
        'XGBoost', FEATURES_DAY2,
        os.path.join(plot_dir, "day2_per_condition_accuracy.png")
    )
    
    # EDA plots
    plot_raw_vs_gt_scatter(day2_data, os.path.join(plot_dir, "day2_raw_vs_gt.png"))
    plot_error_heatmap(day2_data, os.path.join(plot_dir, "day2_error_heatmap.png"))
    
    # ─── Scenario 2: Day 1 → Day 2 (Cross-day generalization) ─────────
    if day1_data is not None and len(day1_data) > 0:
        # Check available features in Day 1
        d1_features = [f for f in FEATURES_DAY2 if f in day1_data.columns]
        
        if len(d1_features) == len(FEATURES_DAY2):
            X_train_d1 = day1_data[FEATURES_DAY2]
            y_train_d1 = day1_data[TARGET]
            X_test_d2_full = day2_data[FEATURES_DAY2]
            y_test_d2_full = day2_data[TARGET]
            
            results_cross, models_cross, preds_cross = train_scenario(
                "Train Day1 → Test Day2",
                X_train_d1, y_train_d1,
                X_test_d2_full, y_test_d2_full
            )
            all_results["Train Day1 → Test Day2"] = results_cross
            
            plot_error_distribution_comparison(
                y_test_d2_full, preds_cross, "Train Day1 → Test Day2",
                os.path.join(plot_dir, "cross_day_error_distributions.png")
            )
        else:
            print(f"\n  Cannot run cross-day scenario: missing features {[f for f in FEATURES_DAY2 if f not in d1_features]}")
        
        # ─── Scenario 2.5: Train Day 2 → Test Day 1 (Cross-day inverse) ───
        if len(d1_features) == len(FEATURES_DAY2):
            X_train_d2_full = day2_data[FEATURES_DAY2]
            y_train_d2_full = day2_data[TARGET]
            X_test_d1_full = day1_data[FEATURES_DAY2]
            y_test_d1_full = day1_data[TARGET]
            
            results_cross_inv, _, preds_cross_inv = train_scenario(
                "Train Day2 → Test Day1",
                X_train_d2_full, y_train_d2_full,
                X_test_d1_full, y_test_d1_full
            )
            all_results["Train Day2 → Test Day1"] = results_cross_inv
            
            plot_error_distribution_comparison(
                y_test_d1_full, preds_cross_inv, "Train Day2 → Test Day1",
                os.path.join(plot_dir, "cross_day_inverse_error_distributions.png")
            )

        # ─── Scenario 3: Combined Day 1 + Day 2 ──────────────────────
        combined = pd.concat([
            day1_data[FEATURES_DAY2 + [TARGET]],
            day2_data[FEATURES_DAY2 + [TARGET]],
        ], ignore_index=True).dropna()
        
        X_combined = combined[FEATURES_DAY2]
        y_combined = combined[TARGET]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42
        )
        
        results_combined, models_combined, preds_combined = train_scenario(
            "Combined Day1+Day2",
            X_train_c, y_train_c, X_test_c, y_test_c
        )
        all_results["Combined Day1+Day2"] = results_combined
        
        # Save combined models
        for mname, model in models_combined.items():
            safe_name = mname.lower().replace(' ', '_').replace('(', '').replace(')', '')
            with open(os.path.join(model_dir, f"combined_{safe_name}.pkl"), 'wb') as f:
                pickle.dump(model, f)
        
        plot_error_distribution_comparison(
            y_test_c, preds_combined, "Combined Day1+Day2",
            os.path.join(plot_dir, "combined_error_distributions.png")
        )
        
        # Feature importance for combined models
        for mname in ['Random Forest', 'XGBoost']:
            if mname in models_combined:
                m = models_combined[mname]
                plot_feature_importance(
                    m, FEATURES_DAY2,
                    f'Feature Importance — {mname} (Combined)',
                    os.path.join(plot_dir, f"combined_feature_importance_{mname.lower().replace(' ', '_')}.png")
                )
    
    # ─── Model comparison chart ───────────────────────────────────────
    plot_model_comparison(all_results, os.path.join(plot_dir, "model_comparison.png"))
    
    # ─── Save all metrics ─────────────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved metrics → {metrics_path}")
    
    # ─── Print summary table ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    for scenario, results in all_results.items():
        print(f"\n  {scenario}:")
        print(f"  {'Model':<25s} {'MAE(mm)':>10s} {'RMSE(mm)':>10s} {'95th(mm)':>10s} {'R²':>10s}")
        print(f"  {'─' * 65}")
        for model_name, metrics in results.items():
            if 'error' in metrics:
                print(f"  {model_name:<25s}   FAILED: {metrics['error']}")
            else:
                print(f"  {model_name:<25s} {metrics['MAE_mm']:>10.2f} {metrics.get('RMSE_mm', 0):>10.2f} "
                      f"{metrics.get('95th_pct_mm', 0):>10.2f} {metrics.get('R2', 0):>10.4f}")
    
    # Best overall
    print(f"\n{'=' * 80}")
    for scenario, results in all_results.items():
        valid = [(k, v['MAE_mm']) for k, v in results.items() 
                 if k != 'Raw Sensor (No ML)' and isinstance(v.get('MAE_mm'), (int, float)) and not np.isnan(v['MAE_mm'])]
        if valid:
            best = min(valid, key=lambda x: x[1])
            print(f"  Best for '{scenario}': {best[0]} (MAE = {best[1]:.2f} mm)")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
