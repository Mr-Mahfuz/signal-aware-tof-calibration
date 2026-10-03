"""
parse_day2_data.py — Parse Day 2 raw scan files and assign ground truth distances.

Reads the 10 scan .txt files from datasets/scan_day_2/, strips debug lines,
interpolates ground truth from the protractor measurements, applies tilt correction,
handles box/transparent object conditions, and outputs cleaned CSVs.

Ground truth methodology:
  - Wall distances measured every 5° pan from 0°–180° (horizontal)
  - Interpolated linearly for intermediate 2° pan steps
  - Tilt correction: gt_at_tilt = gt_horizontal / cos(tilt_deg - 90°)
  - Box (scans 5-8): replaces wall GT at pan 35°-80° for tilt 85°-110°
  - Transparent object (scan 10): at pan 55°-80° for tilt 85°-106°

Output:
  - datasets/scan_day_2_processed/scan{N}_processed.csv  (10 files)
  - datasets/day2_combined.csv                           (merged)
  - datasets/all_days_combined.csv                       (day1 + day2)
"""

import pandas as pd
import numpy as np
import os
import re
from scipy.interpolate import interp1d

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY2_RAW_DIR = os.path.join(BASE_DIR, "datasets", "scan_day_2")
DAY1_DIR = os.path.join(BASE_DIR, "datasets", "scan_day_1")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "scan_day_2_processed")
RAW_DATASET_PATH = os.path.join(BASE_DIR, "raw_dataset.csv")

# ─── Ground Truth: Wall distances (mm) at each 5° pan ───────────────────
# Measured with tape measure + protractor at horizontal (tilt=90°)
WALL_GT_RAW = {
    0: 2580, 5: 1980, 10: 2000, 15: 2040, 20: 2105,
    25: 2185, 30: 2292, 35: 2420, 40: 2588, 45: 2795,
    50: 2760, 55: 2560, 60: 2428, 65: 2330, 70: 2240,
    75: 2180, 80: 2136, 85: 2110, 90: 2100, 95: 2108,
    100: 2134, 105: 2178, 110: 2240, 115: 2320, 120: 2428,
    125: 2550, 130: 2668, 135: 2470, 140: 2486, 145: 2326,
    150: 2194, 155: 2090, 160: 2020, 165: 1964, 170: 1922,
    175: 1920, 180: 1919,
}

# ─── Ground Truth: Box distances (mm) — Scans 5–8 ───────────────────────
# Box is 31 cm tall; visible at approx tilt 85°–110°
BOX_GT = {
    35: 960, 40: 890, 45: 820, 50: 765,
    55: 720, 60: 690, 65: 660, 70: 640,
    75: 620, 80: 600,
}
BOX_TILT_MIN = 85
BOX_TILT_MAX = 110

# ─── Ground Truth: Transparent object distances (mm) — Scan 10 ──────────
# 24 cm tall; visible at approx tilt 85°–106°
TRANSPARENT_GT = {
    55: 720, 60: 690, 65: 660, 70: 640,
    75: 620, 80: 600,
}
TRANSPARENT_TILT_MIN = 85
TRANSPARENT_TILT_MAX = 106

# ─── Scan condition metadata ────────────────────────────────────────────
SCAN_CONDITIONS = {
    1:  {"label": "dim_50pct",       "lighting": "50%",         "has_box": False, "has_transparent": False, "dark_wall": False, "notes": "Incomplete tilt (80-150)"},
    2:  {"label": "bright",          "lighting": "full_bright", "has_box": False, "has_transparent": False, "dark_wall": False, "notes": ""},
    3:  {"label": "bright_gt_check", "lighting": "full_bright", "has_box": False, "has_transparent": False, "dark_wall": False, "notes": "Pan 0-50 reads -1 (verified GT)"},
    4:  {"label": "mixed_lighting",  "lighting": "mixed",       "has_box": False, "has_transparent": False, "dark_wall": False, "notes": "Light changed mid-scan (unknown cutoff). Partial tilt 105-150"},
    5:  {"label": "dark_box",        "lighting": "full_dark",   "has_box": True,  "has_transparent": False, "dark_wall": False, "notes": "Box at pan 35-80"},
    6:  {"label": "dark_box_dwall",  "lighting": "full_dark",   "has_box": True,  "has_transparent": False, "dark_wall": True,  "notes": "Box + dark right wall"},
    7:  {"label": "bright_box_dwall","lighting": "full_bright", "has_box": True,  "has_transparent": False, "dark_wall": True,  "notes": "Box + dark wall (28cm gap bottom). Missing header."},
    8:  {"label": "bright_box_dwall_cov","lighting": "full_bright","has_box": True,"has_transparent": False,"dark_wall": True,  "notes": "Box + dark wall (covered bottom)"},
    9:  {"label": "bright_dwall",    "lighting": "full_bright", "has_box": False, "has_transparent": False, "dark_wall": True,  "notes": "Pan 0-45 reads -1"},
    10: {"label": "bright_trans_dwall","lighting": "full_bright","has_box": False,"has_transparent": True,  "dark_wall": True,  "notes": "Transparent obj + dark wall. Pan 0-45 reads -1"},
}


def build_wall_interpolator():
    """Build a linear interpolator for wall ground truth at any pan angle."""
    pan_angles = sorted(WALL_GT_RAW.keys())
    distances = [WALL_GT_RAW[a] for a in pan_angles]
    return interp1d(pan_angles, distances, kind='linear', fill_value='extrapolate')


def build_object_interpolator(gt_dict):
    """Build a linear interpolator for box/transparent object ground truth."""
    pan_angles = sorted(gt_dict.keys())
    distances = [gt_dict[a] for a in pan_angles]
    return interp1d(pan_angles, distances, kind='linear', fill_value='extrapolate')


def get_ground_truth(pan_deg, tilt_deg, scan_id, wall_interp, box_interp, trans_interp):
    """
    Compute ground truth distance for a given (pan, tilt) and scan condition.
    
    For vertical walls: gt = gt_horizontal / cos(tilt - 90°)
    For objects (box/transparent): same tilt correction applied.
    
    Returns (gt_distance_mm, gt_source) where gt_source describes what the beam hits.
    """
    cond = SCAN_CONDITIONS[scan_id]
    tilt_from_horizontal = abs(tilt_deg - 90)
    
    # Safety: at extreme tilt angles (>55° from horizontal), the beam likely
    # hits ceiling/floor, not the wall. Mark as unreliable.
    if tilt_from_horizontal > 55:
        return np.nan, "extreme_tilt"
    
    cos_tilt = np.cos(np.radians(tilt_from_horizontal))
    if cos_tilt < 0.01:  # Nearly parallel to wall — infinite distance
        return np.nan, "parallel"
    
    # Check if beam hits box (scans 5-8)
    if cond["has_box"] and pan_deg >= 35 and pan_deg <= 80:
        if BOX_TILT_MIN <= tilt_deg <= BOX_TILT_MAX:
            gt_horiz = float(box_interp(pan_deg))
            return gt_horiz / cos_tilt, "box"
    
    # Check if beam hits transparent object (scan 10)
    if cond["has_transparent"] and pan_deg >= 55 and pan_deg <= 80:
        if TRANSPARENT_TILT_MIN <= tilt_deg <= TRANSPARENT_TILT_MAX:
            gt_horiz = float(trans_interp(pan_deg))
            return gt_horiz / cos_tilt, "transparent_object"
    
    # Default: wall
    if 0 <= pan_deg <= 180:
        gt_horiz = float(wall_interp(pan_deg))
        return gt_horiz / cos_tilt, "wall"
    
    return np.nan, "out_of_range"


def parse_scan_file(filepath):
    """
    Parse a Day 2 raw .txt file from ESP32 serial output.
    
    Strips preamble lines (INA219 FAILED, SCAN_START, etc.) and CMD debug lines.
    Returns a DataFrame with the CSV data.
    """
    data_lines = []
    header = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Identify the CSV header line
            if line.startswith("session_id,"):
                header = line
                continue
            
            # Skip non-data lines
            if (line.startswith("CMD ") or
                line.startswith("SCAN_START") or
                line.startswith("SCAN_COMPLETE") or
                line.startswith("Using condition") or
                line.startswith("Enter a condition") or
                line.startswith("INA219")):
                continue
            
            # Data lines start with a digit (session_id)
            if line[0].isdigit():
                data_lines.append(line)
    
    if not header:
        # Scan 7 is missing the header — use default
        header = ("session_id,condition_label,point_index,scan_direction,"
                  "pan_deg,tilt_deg,tof_mm,range_status,signal_rate_kcps,"
                  "ambient_rate_kcps,sigma_mm,corrected_mm,inference_us,"
                  "ina_bus_V,ina_current_mA,ina_power_mW,x,y,z,env_temp_C,humidity_pct")
    
    columns = header.split(",")
    
    # Parse each data line
    records = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) == len(columns):
            records.append(parts)
        elif len(parts) > len(columns):
            # Truncate extra fields
            records.append(parts[:len(columns)])
        # Skip lines with too few fields
    
    df = pd.DataFrame(records, columns=columns)
    
    # Convert numeric columns
    numeric_cols = ['point_index', 'pan_deg', 'tilt_deg', 'tof_mm', 'range_status',
                    'signal_rate_kcps', 'ambient_rate_kcps', 'sigma_mm', 'corrected_mm',
                    'inference_us', 'ina_bus_V', 'ina_current_mA', 'ina_power_mW',
                    'x', 'y', 'z', 'env_temp_C', 'humidity_pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def process_scan(scan_id, df, wall_interp, box_interp, trans_interp):
    """
    Process a single scan: assign ground truth, compute error, add metadata.
    """
    cond = SCAN_CONDITIONS[scan_id]
    
    # Add metadata columns
    df = df.copy()
    df['scan_id'] = scan_id
    df['day'] = 2
    df['condition_label'] = cond['label']
    df['lighting'] = cond['lighting']
    df['has_box'] = cond['has_box']
    df['has_transparent'] = cond['has_transparent']
    df['dark_wall'] = cond['dark_wall']
    
    # Mark valid returns
    df['valid_return'] = (df['tof_mm'] > 0).astype(int)
    
    # Assign ground truth for each point
    gt_distances = []
    gt_sources = []
    for _, row in df.iterrows():
        gt_dist, gt_src = get_ground_truth(
            row['pan_deg'], row['tilt_deg'], scan_id,
            wall_interp, box_interp, trans_interp
        )
        gt_distances.append(gt_dist)
        gt_sources.append(gt_src)
    
    df['gt_distance_mm'] = gt_distances
    df['gt_source'] = gt_sources
    
    # Compute residual error (only for valid returns with valid GT)
    df['residual_error_mm'] = np.where(
        (df['valid_return'] == 1) & (df['gt_distance_mm'].notna()),
        df['gt_distance_mm'] - df['tof_mm'],
        np.nan
    )
    
    # Recompute spherical coordinates using corrected/raw distance
    df['x_raw_mm'] = np.where(df['valid_return'] == 1,
        df['tof_mm'] * np.cos(np.radians(df['tilt_deg'] - 90)) * np.cos(np.radians(df['pan_deg'])),
        0)
    df['y_raw_mm'] = np.where(df['valid_return'] == 1,
        df['tof_mm'] * np.cos(np.radians(df['tilt_deg'] - 90)) * np.sin(np.radians(df['pan_deg'])),
        0)
    df['z_raw_mm'] = np.where(df['valid_return'] == 1,
        df['tof_mm'] * np.sin(np.radians(df['tilt_deg'] - 90)),
        0)
    
    # Ground truth xyz
    df['x_gt_mm'] = np.where(df['gt_distance_mm'].notna(),
        df['gt_distance_mm'] * np.cos(np.radians(df['tilt_deg'] - 90)) * np.cos(np.radians(df['pan_deg'])),
        np.nan)
    df['y_gt_mm'] = np.where(df['gt_distance_mm'].notna(),
        df['gt_distance_mm'] * np.cos(np.radians(df['tilt_deg'] - 90)) * np.sin(np.radians(df['pan_deg'])),
        np.nan)
    df['z_gt_mm'] = np.where(df['gt_distance_mm'].notna(),
        df['gt_distance_mm'] * np.sin(np.radians(df['tilt_deg'] - 90)),
        np.nan)
    
    return df


def load_day1_data():
    """
    Load Day 1 data from the processed CSV files in scan_day_1/.
    Harmonize column names to match Day 2 schema.
    """
    day1_dfs = []
    
    for i in range(1, 11):
        fpath = os.path.join(DAY1_DIR, f"dset_{i}.csv")
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping")
            continue
        
        df = pd.read_csv(fpath, low_memory=False)
        
        # Harmonize column names to Day 2 schema
        rename_map = {
            'pan_commanded_deg': 'pan_deg',
            'tilt_commanded_deg': 'tilt_deg',
            'raw_distance_mm': 'tof_mm',
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        
        # Ensure required columns exist
        if 'pan_deg' not in df.columns and 'pan_calibrated_deg' in df.columns:
            df['pan_deg'] = df['pan_calibrated_deg']
        if 'tilt_deg' not in df.columns and 'tilt_calibrated_deg' in df.columns:
            df['tilt_deg'] = df['tilt_calibrated_deg']
        
        # Add metadata
        df['scan_id'] = i
        df['day'] = 1
        df['condition_label'] = 'day1_session'
        df['lighting'] = 'unknown'
        df['has_box'] = False
        df['has_transparent'] = False
        df['dark_wall'] = False
        
        # Ensure valid_return column
        if 'valid_return' not in df.columns:
            df['valid_return'] = (df['tof_mm'] > 0).astype(int)
        
        if 'residual_error_mm' not in df.columns:
            if 'gt_distance_mm' in df.columns:
                df['residual_error_mm'] = np.where(
                    (df['valid_return'] == 1) & (df['gt_distance_mm'].notna()),
                    df['gt_distance_mm'] - df['tof_mm'],
                    np.nan
                )
        
        # Standard columns for combination
        df['gt_source'] = 'day1_original'
        
        # Ensure signal columns exist with proper names
        if 'signal_rate_kcps' not in df.columns:
            df['signal_rate_kcps'] = np.nan
        if 'ambient_rate_kcps' not in df.columns:
            df['ambient_rate_kcps'] = np.nan
        if 'sigma_mm' not in df.columns:
            df['sigma_mm'] = np.nan
        if 'env_temp_C' not in df.columns and 'temperature_C' in df.columns:
            df['env_temp_C'] = df['temperature_C']
        elif 'env_temp_C' not in df.columns:
            df['env_temp_C'] = np.nan
        
        day1_dfs.append(df)
        print(f"  Loaded dset_{i}.csv: {len(df)} rows ({df['valid_return'].sum()} valid)")
    
    if day1_dfs:
        return pd.concat(day1_dfs, ignore_index=True)
    return pd.DataFrame()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build interpolators
    wall_interp = build_wall_interpolator()
    box_interp = build_object_interpolator(BOX_GT)
    trans_interp = build_object_interpolator(TRANSPARENT_GT)
    
    print("=" * 70)
    print("PHASE 1: Parsing Day 2 Scan Files & Assigning Ground Truth")
    print("=" * 70)
    
    all_day2 = []
    
    for scan_id in range(1, 11):
        fname = f"scan{scan_id}.txt"
        fpath = os.path.join(DAY2_RAW_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"\n  WARNING: {fname} not found, skipping")
            continue
        
        print(f"\n--- Scan {scan_id}: {SCAN_CONDITIONS[scan_id]['label']} ---")
        print(f"  Notes: {SCAN_CONDITIONS[scan_id]['notes']}")
        
        # Parse raw file
        df = parse_scan_file(fpath)
        print(f"  Parsed {len(df)} data rows")
        
        # Process: assign GT, compute error, add metadata
        df = process_scan(scan_id, df, wall_interp, box_interp, trans_interp)
        
        # Stats
        valid = df['valid_return'].sum()
        has_gt = df['gt_distance_mm'].notna().sum()
        trainable = ((df['valid_return'] == 1) & (df['gt_distance_mm'].notna())).sum()
        
        print(f"  Valid returns: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
        print(f"  With ground truth: {has_gt}")
        print(f"  Trainable (valid + GT): {trainable}")
        
        if trainable > 0:
            errors = df.loc[(df['valid_return'] == 1) & (df['residual_error_mm'].notna()), 'residual_error_mm']
            print(f"  Residual error: mean={errors.mean():.1f}mm, std={errors.std():.1f}mm, "
                  f"median={errors.median():.1f}mm")
        
        # GT source breakdown
        for src in df['gt_source'].unique():
            count = (df['gt_source'] == src).sum()
            print(f"    GT source '{src}': {count} points")
        
        # Save individual processed file
        out_path = os.path.join(OUTPUT_DIR, f"scan{scan_id}_processed.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
        
        all_day2.append(df)
    
    # Combine all Day 2
    day2_combined = pd.concat(all_day2, ignore_index=True)
    day2_path = os.path.join(BASE_DIR, "datasets", "day2_combined.csv")
    day2_combined.to_csv(day2_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"DAY 2 COMBINED: {len(day2_combined)} total rows")
    print(f"  Valid returns: {day2_combined['valid_return'].sum()}")
    trainable_d2 = ((day2_combined['valid_return'] == 1) & (day2_combined['gt_distance_mm'].notna())).sum()
    print(f"  Trainable: {trainable_d2}")
    print(f"  Saved -> {day2_path}")
    
    # Load Day 1 and combine
    print(f"\n{'=' * 70}")
    print("Loading Day 1 data...")
    day1_combined = load_day1_data()
    
    if len(day1_combined) > 0:
        # Select common columns for combination
        common_cols = [
            'scan_id', 'day', 'condition_label', 'lighting',
            'has_box', 'has_transparent', 'dark_wall',
            'pan_deg', 'tilt_deg', 'tof_mm', 'valid_return',
            'signal_rate_kcps', 'ambient_rate_kcps', 'sigma_mm',
            'env_temp_C', 'gt_distance_mm', 'gt_source', 'residual_error_mm',
        ]
        
        # Only keep columns that exist in both
        d1_cols = [c for c in common_cols if c in day1_combined.columns]
        d2_cols = [c for c in common_cols if c in day2_combined.columns]
        shared = list(set(d1_cols) & set(d2_cols))
        
        all_combined = pd.concat([
            day1_combined[shared],
            day2_combined[shared],
        ], ignore_index=True)
        
        all_path = os.path.join(BASE_DIR, "datasets", "all_days_combined.csv")
        all_combined.to_csv(all_path, index=False)
        
        trainable_all = ((all_combined['valid_return'] == 1) & (all_combined['gt_distance_mm'].notna())).sum()
        print(f"\nALL DAYS COMBINED: {len(all_combined)} total rows")
        print(f"  Day 1: {len(day1_combined)} rows")
        print(f"  Day 2: {len(day2_combined)} rows")
        print(f"  Trainable: {trainable_all}")
        print(f"  Saved -> {all_path}")
    else:
        print("  No Day 1 data loaded (raw_dataset.csv or dset files not found)")
        print("  Day 2 standalone dataset saved successfully")
    
    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nDay 2 per-condition breakdown:")
    for scan_id in range(1, 11):
        subset = day2_combined[day2_combined['scan_id'] == scan_id]
        valid = (subset['valid_return'] == 1).sum()
        trainable = ((subset['valid_return'] == 1) & (subset['residual_error_mm'].notna())).sum()
        cond = SCAN_CONDITIONS[scan_id]
        print(f"  Scan {scan_id:2d} ({cond['label']:25s}): {valid:5d} valid, {trainable:5d} trainable")
    
    print(f"\nOutput files:")
    print(f"  Individual: datasets/scan_day_2_processed/scan{{1-10}}_processed.csv")
    print(f"  Day 2 merged: datasets/day2_combined.csv")
    if len(day1_combined) > 0:
        print(f"  All days: datasets/all_days_combined.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
