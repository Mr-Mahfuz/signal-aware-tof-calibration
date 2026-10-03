"""
generate_point_cloud_plots.py — Generate static 3D point cloud visualizations.

Creates matplotlib 3D scatter plots for each scan session, comparison views,
error heatmaps, and bird's-eye views.

Output: ml_pipeline/output_day2/plots/point_cloud_*.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY2_CSV = os.path.join(BASE_DIR, "datasets", "day2_combined.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_day2", "plots")

SCAN_LABELS = {
    1: "Scan 1: 50% Light",
    2: "Scan 2: Full Bright",
    3: "Scan 3: Bright (GT Check)",
    4: "Scan 4: Mixed Lighting",
    5: "Scan 5: Dark + Box",
    6: "Scan 6: Dark + Box + Dark Wall",
    7: "Scan 7: Bright + Box + Dark Wall",
    8: "Scan 8: Bright + Box + Dark Wall (Covered)",
    9: "Scan 9: Bright + Dark Wall",
    10: "Scan 10: Bright + Transparent + Dark Wall",
}


def plot_3d_point_cloud(df, title, filename, color_by='distance'):
    """Render a 3D scatter plot of the point cloud."""
    valid = df[df['valid_return'] == 1].copy()
    if len(valid) == 0:
        print(f"  Skipping {title}: no valid points")
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x = valid['x_raw_mm'].values
    y = valid['y_raw_mm'].values
    z = valid['z_raw_mm'].values
    
    if color_by == 'distance':
        c = valid['tof_mm'].values
        cmap = 'viridis'
        clabel = 'Distance (mm)'
    elif color_by == 'error' and 'residual_error_mm' in valid.columns:
        c = valid['residual_error_mm'].values
        c = np.clip(c, -200, 200)  # Clip for visualization
        cmap = 'RdYlGn'
        clabel = 'Residual Error (mm)'
    elif color_by == 'signal':
        c = valid['signal_rate_kcps'].values
        cmap = 'plasma'
        clabel = 'Signal Rate (kcps)'
    else:
        c = valid['tof_mm'].values
        cmap = 'viridis'
        clabel = 'Distance (mm)'
    
    scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=3, alpha=0.6)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label=clabel, shrink=0.6, pad=0.1)
    
    # Set equal aspect ratio
    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(filename)}")


def plot_birds_eye_view(df, title, filename):
    """Top-down (bird's eye) view showing the angular sweep."""
    valid = df[df['valid_return'] == 1].copy()
    if len(valid) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x = valid['x_raw_mm'].values
    y = valid['y_raw_mm'].values
    c = valid['tof_mm'].values
    
    scatter = ax.scatter(x, y, c=c, cmap='viridis', s=3, alpha=0.5)
    
    # Draw sensor origin
    ax.plot(0, 0, 'r*', markersize=15, label='Sensor Origin')
    
    # Draw angular sweep lines at key angles
    max_dist = c.max()
    for angle in [20, 45, 90, 135, 160]:
        rad = np.radians(angle)
        ax.plot([0, max_dist * np.cos(rad)], [0, max_dist * np.sin(rad)],
               'k--', alpha=0.2, linewidth=0.5)
        ax.text(max_dist * 0.55 * np.cos(rad), max_dist * 0.55 * np.sin(rad),
               f'{angle}°', fontsize=8, alpha=0.5)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'{title}\n(Bird\'s Eye View)', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.colorbar(scatter, label='Distance (mm)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(filename)}")


def plot_all_scans_mosaic(day2_data, filename):
    """2x2 mosaic of representative scan point clouds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    target_scans = [2, 5, 9, 10]
    
    for idx, scan_id in enumerate(target_scans):
        ax = axes[idx]
        subset = day2_data[(day2_data['scan_id'] == scan_id) & (day2_data['valid_return'] == 1)]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Scan {scan_id}')
            continue
        
        x = subset['x_raw_mm'].values
        y = subset['y_raw_mm'].values
        c = subset['tof_mm'].values
        
        scatter = ax.scatter(x, y, c=c, cmap='viridis', s=4, alpha=0.8)
        ax.plot(0, 0, 'r*', markersize=12)
        
        ax.set_title(SCAN_LABELS.get(scan_id, f'Scan {scan_id}'), fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(filename)}")


def plot_raw_vs_gt_3d(df, scan_id, filename):
    """Side-by-side 3D: raw point cloud vs ground truth point cloud."""
    valid = df[(df['valid_return'] == 1) & (df['gt_distance_mm'].notna())].copy()
    if len(valid) == 0:
        return
    
    fig = plt.figure(figsize=(18, 8))
    
    # Raw point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(valid['x_raw_mm'], valid['y_raw_mm'], valid['z_raw_mm'],
                     c=valid['tof_mm'], cmap='viridis', s=3, alpha=0.6)
    ax1.set_title('Raw Sensor', fontsize=11)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    
    # Ground truth point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(valid['x_gt_mm'], valid['y_gt_mm'], valid['z_gt_mm'],
                     c=valid['gt_distance_mm'], cmap='viridis', s=3, alpha=0.6)
    ax2.set_title('Ground Truth', fontsize=11)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    
    # Sync view angles
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
    
    fig.suptitle(f'{SCAN_LABELS.get(scan_id, f"Scan {scan_id}")}: Raw vs Ground Truth',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(filename)}")


def plot_error_by_angle(day2_data, filename):
    """Line plot showing mean error vs pan angle for each condition."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    conditions = sorted(day2_data['condition_label'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    
    for i, cond in enumerate(conditions):
        subset = day2_data[
            (day2_data['condition_label'] == cond) & 
            (day2_data['valid_return'] == 1) &
            (day2_data['residual_error_mm'].notna())
        ]
        if len(subset) == 0:
            continue
        
        grouped = subset.groupby('pan_deg')['residual_error_mm'].mean()
        ax.plot(grouped.index, grouped.values, label=cond, color=colors[i], 
                alpha=0.7, linewidth=1.5)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Pan Angle (°)')
    ax.set_ylabel('Mean Residual Error (mm)')
    ax.set_title('Mean Residual Error vs Pan Angle by Condition')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(filename)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Day 2 combined data...")
    day2_data = pd.read_csv(DAY2_CSV, low_memory=False)
    print(f"  {len(day2_data)} rows loaded")
    
    print("\n--- Generating Per-Scan 3D Point Clouds ---")
    for scan_id in range(1, 11):
        subset = day2_data[day2_data['scan_id'] == scan_id]
        label = SCAN_LABELS.get(scan_id, f'Scan {scan_id}')
        
        # 3D point cloud colored by distance
        plot_3d_point_cloud(
            subset, label,
            os.path.join(OUTPUT_DIR, f"point_cloud_scan{scan_id}_distance.png"),
            color_by='distance'
        )
        
        # 3D point cloud colored by error
        plot_3d_point_cloud(
            subset, f'{label} (Error)',
            os.path.join(OUTPUT_DIR, f"point_cloud_scan{scan_id}_error.png"),
            color_by='error'
        )
    
    print("\n--- Generating Overview Plots ---")
    
    # Bird's eye view for select scans
    for scan_id in [2, 5, 9, 10]:
        subset = day2_data[day2_data['scan_id'] == scan_id]
        label = SCAN_LABELS.get(scan_id, f'Scan {scan_id}')
        plot_birds_eye_view(
            subset, label,
            os.path.join(OUTPUT_DIR, f"birds_eye_scan{scan_id}.png")
        )
    
    # All scans mosaic
    plot_all_scans_mosaic(day2_data, os.path.join(OUTPUT_DIR, "all_scans_mosaic.png"))
    
    # Raw vs GT comparison for key scans
    print("\n--- Generating Raw vs Ground Truth Comparisons ---")
    for scan_id in [2, 5, 9]:
        subset = day2_data[day2_data['scan_id'] == scan_id]
        plot_raw_vs_gt_3d(
            subset, scan_id,
            os.path.join(OUTPUT_DIR, f"raw_vs_gt_scan{scan_id}.png")
        )
    
    # Error by angle
    plot_error_by_angle(day2_data, os.path.join(OUTPUT_DIR, "error_by_pan_angle.png"))
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
