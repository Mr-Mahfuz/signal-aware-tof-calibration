# ML Calibration Model Integration Guide for ESP32

Hello! This document explains how to integrate the machine learning calibration model into the ESP32 firmware for the VL53L1X LiDAR sensor. 

The ML model has been transpiled into pure, dependency-free C code specifically designed to run fast on microcontrollers. 

## What the Model Does
The model is **Signal-Aware**. It takes the raw distance and pairs it with the sensor's physical signal metrics (ambient light, photon return rate, etc.) to calculate a highly accurate **Error Correction** (Residual Error).

Instead of replacing your distance calculation, it provides an offset.
**Formula:** `Corrected Distance = Raw Distance + Model Output`

## Files to Include
1. Add `model_inference.c` to your ESP-IDF or Arduino project.
2. If using C++, you may need to wrap the include/declaration in `extern "C"` or just rename the file to `model_inference.cpp`.

## How to Use the Function

The file contains a single function:
```c
double score(double * input);
```

### 1. Prepare the Input Array
Every time you take a measurement with the VL53L1X, you need to populate an array of 6 `double` values in this exact order:

```c
double features[6];

// 1. The uncalibrated distance reported by the sensor (in mm)
features[0] = raw_distance_mm;       

// 2. The physical pan angle of the servo/stepper (in degrees)
features[1] = pan_calibrated_deg;    

// 3. The physical tilt angle of the servo/stepper (in degrees)
features[2] = tilt_calibrated_deg;   

// 4. Signal Rate (in kcps - kilo counts per second)
// -> From VL53L1X: VL53L1X_GetSignalRate()
features[3] = signal_rate_kcps;      

// 5. Ambient Rate (in kcps)
// -> From VL53L1X: VL53L1X_GetAmbientRate()
features[4] = ambient_rate_kcps;     

// 6. Sigma (in mm) - The sensor's internal variance/confidence
// -> From VL53L1X: VL53L1X_GetSigma()
features[5] = sigma_mm;              
```

### 2. Run Inference and Apply Correction
Call the function and add the result to your raw distance:

```c
// Run the ML model (takes microseconds on ESP32)
double predicted_error_mm = score(features);

// Apply the correction
double final_calibrated_distance_mm = raw_distance_mm + predicted_error_mm;

// Use final_calibrated_distance_mm for your 3D point cloud generation!
```

## Performance Notes
- The model is a deeply nested tree of `if/else` statements.
- The compiled size is roughly ~1-3 MB depending on compiler optimizations, so ensure your ESP32 partition table has enough `app` space (e.g., use a `Huge APP` partition scheme).
- Execution time is extremely fast (usually < 100 microseconds) because it requires no loops, floating-point division, or complex math, just basic comparisons.
