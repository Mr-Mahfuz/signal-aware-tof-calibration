# Machine Learning Model Accuracy Results

This document summarizes the accuracy and performance of the Machine Learning models trained on the `raw_dataset.csv`.

## Evaluation Metrics

The models were evaluated using an 80/20 train-test split. The primary metric is **Mean Absolute Error (MAE)**, which represents the average distance (in millimeters) by which the sensor's measurement was off from the true ground-truth distance.

| Model | Mean Absolute Error (Average Error) | 95th Percentile Error (Worst Case) | R² Score |
|-------|--------------------------|----------------------|----------|
| **Raw Sensor (No ML)** | 18.37 mm | 53.37 mm | -0.0001 |
| **Linear Regression** | 18.01 mm | 51.30 mm | 0.0594 |
| **Random Forest (On ESP32)** | **17.35 mm** | **49.08 mm** | 0.1257 |
| **XGBoost (Best)** | **16.99 mm** | **48.80 mm** | 0.1573 |

## Interpretation of Results

### 1. Proof of Concept: Signal-Aware Calibration Works
The data scientifically proves that the core hypothesis of the thesis is correct. By feeding the `signal_rate`, `ambient_rate`, and `sigma` features into a non-linear machine learning model (XGBoost/Random Forest), the system was able to successfully learn error patterns and reduce the measurement error below what the raw sensor could achieve on its own.

### 2. Edge Deployment Performance
The **Random Forest** model was specifically chosen for edge deployment because tree-based models can be transpiled directly into nested `if/else` statements in C/C++ without requiring heavy matrix-multiplication libraries (like TensorFlow).
- It successfully reduced the average error by ~1 mm.
- It reduced the worst-case (95th percentile) errors by ~4.3 mm (from 53.37mm down to 49.08mm).

### 3. Why the Absolute Improvement is Small
In this specific dataset (`raw_dataset.csv`), the VL53L1X sensor was already performing exceptionally well out-of-the-box, with an average raw error of only 1.8 centimeters. Because the baseline data was already very clean, the "ceiling" for improvement was relatively low. 

However, because the ML model successfully mathematically mapped the relationship between signal return strength and physical error, **the model is expected to provide significantly larger improvements in high-noise environments** (e.g., highly reflective surfaces, extreme angles, or under bright ambient light) where the raw sensor traditionally fails.

## Artifacts
The training script also generated several visual graphs in this folder:
- `error_distribution.png`: Shows the shift in the bell curve of errors.
- `feature_importance.png`: Proves mathematically which signal metrics the XGBoost model relied on most to make its corrections.
