# LiDAR Machine Learning Pipeline Documentation

## 1. Overview
This document explains the workflow and architectural decisions for the Machine Learning pipeline designed to calibrate the ToF (Time-of-Flight) LiDAR sensor (VL53L1X + ESP32).

The core thesis of this work is **"Signal-Aware Calibration"**. Raw distance measurements from a ToF sensor suffer from systematic errors (like multi-path interference) which cannot be corrected using simple linear regression. However, the physical photons that cause these errors also affect the sensor's internal metrics: `signal_rate_kcps`, `ambient_rate_kcps`, and `sigma_mm`. By feeding these features into a machine learning model, the model can learn complex, non-linear relationships and predict the true distance much more accurately than the sensor's raw output.

## 2. Dataset Pre-processing
The dataset (`raw_dataset.csv`) contains ~57,000 scans, of which ~51,000 have valid returns and a synthetic ground truth.

**Features Used ($X$):**
- `raw_distance_mm`: The uncalibrated distance reported by the sensor.
- `pan_calibrated_deg` & `tilt_calibrated_deg`: Spatial coordinates (some errors are angular/lens dependent).
- `signal_rate_kcps`: Photons returning from the target (critical for reflectivity/multipath estimation).
- `ambient_rate_kcps`: Background infrared interference.
- `sigma_mm`: The sensor's own internal confidence/variance estimate.

**Target Variable ($y$):**
- `residual_error_mm`: Calculated as `gt_distance_mm - raw_distance_mm`. 
- *Why predict error instead of distance?* Predicting the absolute distance directly ignores the fundamental physics of the ToF measurement. By predicting the residual error, the model acts as a true "calibration" offset, and the final corrected distance is simply: $Distance_{corr} = Distance_{raw} + Error_{predicted}$

## 3. Training Workflow (02_train.py)
The data is split 80% for training and 20% for testing. Three distinct models are trained to provide a scientific comparison:

### A. Linear Regression (Baseline)
- **Why we chose it**: Serves as the mathematical baseline. It assumes errors scale linearly with distance or signal.
- **Expected Result**: Will likely underperform because ToF multipathing and spatial distortions are highly non-linear.

### B. Random Forest (Edge-Deployable Ensemble)
- **Why we chose it**: Random Forests handle non-linear data and interactions between features (e.g., how high ambient light affects raw distance) exceptionally well without needing much hyperparameter tuning.
- **Hardware constraints**: Trees can be transpiled directly into nested `if/else` statements in C++ using `m2cgen`, making them perfect for microcontrollers (ESP32) that lack the RAM to run a neural network inference engine like TensorFlow Lite. We restrict the depth (`max_depth=10`) and estimators (`n_estimators=50`) to ensure the resulting C code fits in the ESP32's flash memory.

### C. XGBoost (Gradient Boosted Trees - Upper Bound)
- **Why we chose it**: XGBoost builds trees sequentially to correct the errors of previous trees. It typically represents the absolute state-of-the-art for tabular data like this.
- **Purpose**: We use XGBoost to find the theoretical upper limit of accuracy. It is also used to generate the **Feature Importance** plot, which proves scientifically that the `signal_rate` is a critical feature, validating the core hypothesis of the thesis.

## 4. Edge Export (03_export_to_c.py)
The ultimate goal of edge robotics is to run inference in real-time. 
Instead of sending data to a PC for calibration, we use `m2cgen` (Model 2 C Code Generator). This library converts the trained Random Forest into a pure C function `double score(double * input)`. 

This generated file (`model_inference.c`) requires no external dependencies. It can be compiled natively by the ESP32 (Xtensa GCC compiler) and executes in microseconds, perfectly matching the 15ms timing budget of the VL53L1X sensor.
