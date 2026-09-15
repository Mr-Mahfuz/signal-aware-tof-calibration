import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_output_dir():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def load_and_prep_data(filepath):
    print(f"Loading data from {filepath}...")
    # Add low_memory=False to suppress DtypeWarning for column 0
    df = pd.read_csv(filepath, low_memory=False)
    
    # Filter for valid returns and rows where ground truth is available
    trainable_df = df[(df['valid_return'] == 1) & (df['gt_distance_mm'].notna())].copy()
    print(f"Filtered to {len(trainable_df)} trainable points.")
    return trainable_df

def plot_error_distribution(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['residual_error_mm'], bins=100, kde=True, color='blue')
    plt.title('Distribution of Residual Errors (Raw Distance - Ground Truth)')
    plt.xlabel('Residual Error (mm)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_distribution.png'))
    plt.close()
    print("Saved error_distribution.png")

def plot_raw_vs_gt(df, out_dir):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['gt_distance_mm'], df['raw_distance_mm'], alpha=0.1, s=5, color='blue')
    
    # Plot ideal 1:1 line
    max_val = max(df['gt_distance_mm'].max(), df['raw_distance_mm'].max())
    min_val = min(df['gt_distance_mm'].min(), df['raw_distance_mm'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (1:1)')
    
    plt.title('Raw Distance vs Ground Truth Distance')
    plt.xlabel('Ground Truth Distance (mm)')
    plt.ylabel('Raw Distance (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'raw_vs_gt.png'))
    plt.close()
    print("Saved raw_vs_gt.png")

def plot_feature_correlation(df, out_dir):
    features = ['raw_distance_mm', 'pan_calibrated_deg', 'tilt_calibrated_deg', 
                'signal_rate_kcps', 'ambient_rate_kcps', 'sigma_mm', 'residual_error_mm']
    
    corr = df[features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_correlation.png'))
    plt.close()
    print("Saved feature_correlation.png")

def plot_error_vs_signal(df, out_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['signal_rate_kcps'], df['residual_error_mm'], alpha=0.1, s=5, color='purple')
    plt.title('Residual Error vs Signal Rate')
    plt.xlabel('Signal Rate (kcps)')
    plt.ylabel('Residual Error (mm)')
    plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_vs_signal.png'))
    plt.close()
    print("Saved error_vs_signal.png")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_dataset.csv')
    out_dir = create_output_dir()
    
    df = load_and_prep_data(data_path)
    
    print("Generating EDA plots...")
    plot_error_distribution(df, out_dir)
    plot_raw_vs_gt(df, out_dir)
    plot_feature_correlation(df, out_dir)
    plot_error_vs_signal(df, out_dir)
    
    print("EDA Complete. Check the 'output' directory for plots.")
