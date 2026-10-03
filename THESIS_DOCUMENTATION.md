# Comprehensive Thesis Documentation

## Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Indoor 3D Reconstruction

---

> **Authors:** Mahfuz Ahmad _et al._
> **Affiliation:** _(To be filled before submission)_
> **Repository:** [github.com/Mr-Mahfuz/signal-aware-tof-calibration](https://github.com/Mr-Mahfuz/signal-aware-tof-calibration)
> **Document compiled:** October 3, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Problem and Motivation](#2-research-problem-and-motivation)
3. [Research Questions](#3-research-questions)
4. [Literature Review and Theoretical Foundations](#4-literature-review-and-theoretical-foundations)
5. [Hardware Architecture](#5-hardware-architecture)
6. [Ground Truth Methodology](#6-ground-truth-methodology)
7. [Data Collection Campaigns](#7-data-collection-campaigns)
8. [Machine Learning Pipeline](#8-machine-learning-pipeline)
9. [Model Training and Results](#9-model-training-and-results)
10. [ESP32 Edge Deployment (TinyML)](#10-esp32-edge-deployment-tinyml)
11. [3D Point Cloud Reconstruction](#11-3d-point-cloud-reconstruction)
12. [Research Paper Evolution](#12-research-paper-evolution)
13. [Repository and File Structure](#13-repository-and-file-structure)
14. [Key Design Decisions and Rationale](#14-key-design-decisions-and-rationale)
15. [Limitations and Failure Modes](#15-limitations-and-failure-modes)
16. [Future Work](#16-future-work)
17. [Complete Bibliography and Citations](#17-complete-bibliography-and-citations)
18. [Appendices](#18-appendices)

---

## 1. Executive Summary

This thesis presents a complete, end-to-end pipeline for building a low-cost 3D LiDAR scanner from commodity components and using Machine Learning (ML) to improve its measurement accuracy in real time. The central innovation -- termed **"Signal-Aware Calibration"** -- exploits the internal photon telemetry signals exposed by the VL53L1X Time-of-Flight (ToF) sensor (signal return rate, ambient noise rate, and sigma confidence) as input features to an ML model that predicts and corrects range measurement errors. The trained model is then transpiled into pure C code and deployed directly onto an ESP32 microcontroller, enabling real-time, on-device calibration without any external compute dependency.

### Key Achievement Summary

| Metric | Value |
|--------|-------|
| **Total sensor data points collected** | ~68,000+ across two campaigns |
| **Trainable data points (valid + ground truth)** | ~59,500 |
| **Best single-environment MAE (XGBoost)** | 16.99 mm (Day 1) |
| **Best multi-environment MAE (XGBoost, Combined)** | 22.41 mm |
| **ESP32-deployed model MAE (Decision Tree, Depth 12)** | 23.42 mm |
| **Inference latency on ESP32** | < 100 us per prediction |
| **Generated C code size (ESP32 model)** | 453.8 KB |
| **Cross-environment transfer MAE (failure case)** | 947-1049 mm |
| **Sensor timing budget** | 50 ms per ranging cycle |
| **Total hardware cost** | < $15 USD |

### One-Sentence Summary

> We taught a $4 ToF sensor to correct its own measurement errors by feeding its internal photon telemetry into a machine learning model that runs natively on a $3 ESP32 microcontroller -- and demonstrated that single-environment calibration fails catastrophically under domain shift, while pooled multi-environment training reduces aggregate error from 149.84 mm to 22.41 mm.

---

## 2. Research Problem and Motivation

### 2.1 The Accessibility Problem of 3D Scanning

Indoor 3D mapping is critical for mobile robotics, structural inspection, augmented reality, and autonomous navigation [Hess et al., 2016; Rusu & Cousins, 2011]. Commercial LiDAR systems (e.g., Velodyne, SICK, Hokuyo) deliver high accuracy and dense point clouds but are prohibitively expensive ($500-$10,000+) and power-hungry for educational labs, hobbyist robotics, and resource-constrained embedded applications [Amann et al., 2001; Behroozpour et al., 2017].

The low-cost alternative is to mount a single-point ToF sensor on a two-axis pan-tilt mechanism driven by servo motors, sweeping it across the environment to build a 3D point cloud one ray at a time [Willis et al., 2009]. The STMicroelectronics **VL53L1X** sensor is one of the most popular choices: it integrates a 940 nm VCSEL emitter, a 16x16 SPAD receiver array, and full timing/signal-processing logic into a single package, costing approximately $4 USD.

### 2.2 Why ToF Sensors Make Measurement Mistakes

The VL53L1X datasheet promises +/-3% ranging accuracy under ideal conditions. In practice, four principal error sources degrade measurements:

1. **Multipath Interference (MPI):** When the laser cone illuminates a surface near a wall corner or concavity, photons travel along multiple paths before reaching the detector. The sensor averages arrival times from both direct and reflected paths, inflating the reported distance. [Sabov & Kruger, 2008; Reynolds et al., 2011; Dorrington et al., 2011].

2. **Target Reflectivity:** Dark or absorptive surfaces return fewer photons, increasing statistical jitter. Highly reflective surfaces produce specular reflections that scatter unpredictably [Corti et al., 2016; He et al., 2017].

3. **Ambient Light Contamination:** Sunlight or room lighting adds shot noise to the SPAD array, degrading the signal-to-background ratio [Foix et al., 2011; Hansard et al., 2012].

4. **Mechanical Jitter:** Cheap SG90 micro servos exhibit angular backlash, vibration, and non-uniform angular response. A 0.25 deg angular error at 3 m range translates to approximately 13.1 mm of lateral displacement.

### 2.3 Why Traditional Calibration Fails

Traditional ToF calibration relies on polynomial look-up tables indexed solely by measured distance [Lindner et al., 2010; Kahlmann et al., 2006]. These assume that range error is a one-dimensional function of distance. In reality, the error depends on multiple variables simultaneously: distance, angle, signal strength, ambient noise, target material, and temperature.

### 2.4 The "Signal-Aware" Hypothesis

The VL53L1X exposes internal diagnostic metrics alongside every distance measurement:

- **Signal Rate (kcps):** Strength of the reflected laser return. A proxy for target reflectivity and distance-dependent signal attenuation.
- **Ambient Rate (kcps):** Background infrared noise level. Captures ambient lighting conditions.
- **Sigma (mm):** The sensor's own internal confidence/variance estimate.

**Our hypothesis:** These telemetry metrics contain sufficient information about the physical state of the photon return to enable an ML model to predict and correct the measurement error.

---

## 3. Research Questions

| # | Research Question |
|---|---|
| **RQ1** | Can sensor-internal telemetry features (signal rate, ambient rate, sigma) improve ML-based distance calibration beyond what geometric features alone achieve? |
| **RQ2** | Can the trained calibration model be deployed directly onto a resource-constrained ESP32 microcontroller via tree-model transpilation? |
| **RQ3** | How does the calibration model generalize across different indoor environments with different lighting, materials, and scene conditions? |
| **RQ4** | What is the trade-off between model accuracy and memory footprint for embedded deployment? |

### Research Question Answers

- **RQ1: Yes.** Feature ablation showed that removing telemetry features increased XGBoost MAE from 22.41 mm to 27.58 mm -- a 23% degradation.
- **RQ2: Yes.** A depth-12 Decision Tree was transpiled to 453.8 KB of C code via m2cgen, achieving < 100 us inference latency on ESP32.
- **RQ3: Poorly in isolation, well when pooled.** Single-environment models transfer catastrophically (947-1049 mm MAE). Pooled training reduced this to 22.41 mm.
- **RQ4:** XGBoost (22.41 mm MAE) produced ~40 MB of C code. A depth-12 Decision Tree sacrificed only ~1 mm MAE (23.42 mm) while reducing code to 453.8 KB.

---

## 4. Literature Review and Theoretical Foundations

### 4.1 ToF Sensor Error Characterization

- **Frangez et al. (2022)** demonstrated that ToF measurement accuracy varies with target distance, device temperature, and material properties.
- **Corti et al. (2016)** characterized the Kinect V2 ToF camera, finding that temperature, range, surface properties, and radial position all affect uncertainty.
- **Kahlmann et al. (2006)** calibrated the SwissRanger showing systematic errors can be reduced through device-specific tables.
- **Fuchs & Hirzinger (2008)** addressed extrinsic and depth calibration of ToF cameras.
- **Lindner & Kolb (2006)** established foundational methods for combined geometric-radiometric correction of PMD sensors.
- **Wasenmuller & Stricker (2016)** compared Kinect V1 and V2 depth images in terms of accuracy and precision.

### 4.2 Multipath Interference (MPI)

- **Sabov & Kruger (2008)** proposed methods for identifying and correcting "flying pixels" caused by mixed-path returns.
- **Reynolds et al. (2011)** developed approaches for capturing ToF data with confidence measures.
- **Dorrington et al. (2011)** demonstrated techniques for separating true range measurements from multi-path interference.

### 4.3 Machine Learning for Depth Correction

- **Su et al. (2018) -- DeepToF:** Demonstrated that learned models can map complex ToF distortions to corrected depth.
- **Agresti et al. (2018):** Showed that CNNs can exploit spatial context to reduce MPI artifacts.
- **Guo et al. (2021):** Tackled multipath interference using deep learning.
- **He et al. (2017):** Formalized residual error analysis for ToF cameras -- a direct inspiration for our "signal-aware" approach.
- **Mersmann et al. (2013):** Applied ToF calibration to intraoperative surface reconstruction.
- **Jung et al. (2015):** Proposed joint calibration of color-depth camera pairs.

**Gap identified:** Most approaches target imaging-class ToF cameras or GPU-capable systems. Our work addresses a single-point sensor on a microcontroller.

### 4.4 Low-Cost LiDAR and Indoor Mapping

- **Willis et al. (2009):** Design and implementation of an inexpensive LIDAR scanning system.
- **Jimenez et al. (2014):** 3D scanner based on a ToF camera and a pan-tilt unit.
- **Kim & Lee (2015):** Design and calibration of a pan-tilt ToF scanning system.
- **Park et al. (2019):** FPGA-based low-cost 3D LiDAR from multiple rotating 2D LiDARs.
- **Low et al. (2024):** Low-cost 3D mapping system using 2D LiDAR and monocular cameras.

### 4.5 TinyML and Edge-Based Calibration

- **Warden & Situnayake (2019):** Definitive text on TinyML with TensorFlow Lite on Arduino.
- **Wang et al. (2020):** ML on ultra-low-power microcontrollers at IEEE AICAS.
- **Hayajneh et al. (2024):** TinyML-empowered transfer learning on edge devices.
- **Banbury et al. (2021):** Standardized benchmarks for TinyML systems.
- **David et al. (2021):** TensorFlow Lite Micro for embedded ML.
- **Lin et al. (2020) -- MCUNet:** Tiny deep learning on IoT devices.
- **Model transpilation:** m2cgen converts scikit-learn [Pedregosa et al., 2011] models into pure C code.

### 4.6 ML Algorithm References

- **Breiman (2001):** Random Forests -- builds multiple decision trees on random subsets, averaging predictions.
- **Chen & Guestrin (2016):** XGBoost -- scalable tree boosting, state-of-the-art for tabular data.

---

## 5. Hardware Architecture

### 5.1 Component Bill of Materials

| Component | Model | Interface | Role | Cost |
|-----------|-------|-----------|------|------|
| **ToF Sensor** | VL53L1X (TOF400) | I2C (GPIO 21/22) | Distance + signal telemetry | ~$4 |
| **Microcontroller** | ESP32 DevKit V1 | -- | Firmware, servos, ML model | ~$3 |
| **Pan Servo** | SG90 | GPIO 13 (PWM) | Horizontal sweep 0-180 deg | ~$1 |
| **Tilt Servo** | SG90 | GPIO 12 (PWM) | Vertical sweep | ~$1 |
| **Temperature** | DHT11 | GPIO 4 | Ambient thermal drift | ~$1 |
| **Current Monitor** | INA219 | I2C (shared bus) | Power fluctuations | ~$2 |
| **Power Supply** | External 5V regulator | Servo rail | Prevent brownouts | ~$1 |
| **Total** | | | | **< $15** |

### 5.2 Critical Hardware Lesson

> **Brownout Prevention:** Servos MUST be powered from an external 5V supply, not the ESP32's onboard regulator. Running SG90 servos from the ESP32 caused voltage brownouts within the first five minutes of scanning.

### 5.3 Firmware Platforms

1. **Data Collection Firmware** (`lidar code/src/main.cpp`) -- PlatformIO project with multi-scan support, CSV output with 21 columns
2. **Inference Firmware** (`esp32_inference/esp32_inference.ino`) -- Arduino project with transpiled ML model, real-time correction

### 5.4 Scanner UI Tool

A Python GUI (`lidar code/scanner_ui.py`) using Tkinter for serial communication, condition labeling, real-time progress monitoring, and automatic CSV file creation.

---

## 6. Ground Truth Methodology

### 6.1 Approach

Ground truth was derived geometrically using ray-plane intersection:
1. Sensor rotation center defined as origin (0, 0, 0)
2. Custom-printed protractor grid on floor establishes angular reference at 5 deg intervals
3. Walls and objects measured with tape measure + laser distance meter (+/-1 mm)
4. Expected range computed as ray-plane intersection at each (pan, tilt)

### 6.2 Coordinate Transform

Spherical to Cartesian: x = r * cos(tilt-90) * cos(pan), y = r * cos(tilt-90) * sin(pan), z = r * sin(tilt-90)

### 6.3 Tilt Correction

For vertical walls: gt_at_tilt = gt_horizontal / cos(|tilt - 90 deg|)

### 6.4 Day 2 Wall Distances (measured every 5 deg)

| Pan | mm | Pan | mm | Pan | mm |
|-----|-----|-----|-----|-----|-----|
| 0 | 2580 | 65 | 2330 | 130 | 2668 |
| 5 | 1980 | 70 | 2240 | 135 | 2470 |
| 10 | 2000 | 75 | 2180 | 140 | 2486 |
| 15 | 2040 | 80 | 2136 | 145 | 2326 |
| 20 | 2105 | 85 | 2110 | 150 | 2194 |
| 25 | 2185 | 90 | 2100 | 155 | 2090 |
| 30 | 2292 | 95 | 2108 | 160 | 2020 |
| 35 | 2420 | 100 | 2134 | 165 | 1964 |
| 40 | 2588 | 105 | 2178 | 170 | 1922 |
| 45 | 2795 | 110 | 2240 | 175 | 1920 |
| 50 | 2760 | 115 | 2320 | 180 | 1919 |
| 55 | 2560 | 120 | 2428 | | |
| 60 | 2428 | 125 | 2550 | | |

### 6.5 Object Ground Truth (Day 2)

- **Cardboard box** (Scans 5-8): Pan 35-80 deg, 600-960 mm, Tilt 85-110 deg
- **Translucent object** (Scan 10): Pan 55-80 deg, 600-720 mm, Tilt 85-106 deg

---

## 7. Data Collection Campaigns

### 7.1 Overview

| Property | Day 1 (Environment A) | Day 2 (Environment B) |
|----------|----------------------|----------------------|
| **Location** | Dark room, uniform walls | Hallway, LED-lit, obstacles |
| **Total raw measurements** | 57,352 | 11,337 |
| **Valid sensor returns** | 53,539 | 9,104 |
| **Trainable rows** | 51,000 | 8,477 |
| **Pan range** | 0-180 deg at 1 deg steps | 20-160 deg at 2 deg steps |
| **Tilt range** | 0-179.9 deg | 70-150 deg at 5 deg steps |
| **Signal telemetry** | Corrupted (constant 15.0 kcps) | Properly logged |
| **Scanning sessions** | 10 | 10 |

### 7.2 Day 2 Scan Conditions

| Scan | Label | Lighting | Obstacles |
|------|-------|----------|-----------|
| 1 | dim_50pct | 50% | None |
| 2 | bright | Full bright | None |
| 3 | bright_gt_check | Full bright | None |
| 4 | mixed_lighting | Mixed | None |
| 5 | dark_box | Full dark | Cardboard box |
| 6 | dark_box_dwall | Full dark | Box + dark wall |
| 7 | bright_box_dwall | Full bright | Box + dark wall |
| 8 | bright_box_dwall_cov | Full bright | Box + dark wall (covered) |
| 9 | bright_dwall | Full bright | Dark wall |
| 10 | bright_trans_dwall | Full bright | Transparent + dark wall |

### 7.3 Data Processing

Day 2 raw data processed via `ml_pipeline/parse_day2_data.py`: parsing serial output, assigning ground truth via interpolation, computing residuals, generating Cartesian coordinates.

---

## 8. Machine Learning Pipeline

### 8.1 Problem Formulation

Residual correction: e = r_GT - r_raw. Model predicts e_hat. Corrected: r_corr = r_raw + e_hat.

**Why residual, not absolute distance?** Training to predict absolute distance caused the model to learn the identity function, ignoring signal features.

### 8.2 Feature Vector (6 dimensions)

| Feature | Description |
|---------|-------------|
| raw_distance / tof_mm | Sensor-reported distance (mm) |
| pan_angle / pan_deg | Pan servo angle (degrees) |
| tilt_angle / tilt_deg | Tilt servo angle (degrees) |
| signal_rate_kcps | Photon return strength (kcps) |
| ambient_rate_kcps | Background IR noise (kcps) |
| sigma_mm / env_temp_C | Confidence or temperature |

**Target:** residual_error_mm

### 8.3 Pipeline Scripts

- **01_eda.py** -- Exploratory Data Analysis (histograms, scatter plots, correlation heatmaps)
- **02_train.py** -- Day 1 training (Linear Regression, Random Forest, XGBoost)
- **03_export_to_c.py** -- Transpile Random Forest to C via m2cgen
- **parse_day2_data.py** -- Parse Day 2 raw .txt files, assign ground truth
- **train_day2.py** -- Multi-environment training (4 scenarios x 5 models)
- **train_esp32_model.py** -- ESP32-optimized Decision Tree export
- **generate_point_cloud_plots.py** -- 3D visualization scripts

---

## 9. Model Training and Results

### 9.1 Day 1 Results (Controlled Environment)

| Model | MAE (mm) | 95th Pct (mm) | R-squared |
|-------|----------|---------------|-----------|
| Raw Sensor | 18.37 | 53.37 | -0.0001 |
| Linear Regression | 18.01 | 51.30 | 0.0594 |
| **Random Forest** | **17.35** | **49.08** | 0.1257 |
| **XGBoost** | **16.99** | **48.80** | 0.1573 |

### 9.2 Day 2 Results (Diverse Environment)

| Model | MAE (mm) | 95th Pct (mm) | R-squared |
|-------|----------|---------------|-----------|
| Raw Sensor | 934.97 | 2617.87 | -0.0085 |
| Linear Regression | 367.94 | 1040.66 | 0.8251 |
| Random Forest | 65.22 | 376.96 | 0.9757 |
| **XGBoost** | **49.34** | **203.70** | **0.9854** |
| Gradient Boosting | 77.01 | 425.26 | 0.9753 |
| MLP Neural Net | 166.40 | 750.32 | 0.9419 |

### 9.3 Cross-Environment Transfer (FAILURE)

| Train | Test | Raw MAE | XGBoost MAE |
|-------|------|---------|-------------|
| Day 1 | Day 2 | 934.97 mm | **947.15 mm** |
| Day 2 | Day 1 | 18.24 mm | **1049.09 mm** |
| **Combined** | **Combined** | **149.84 mm** | **22.41 mm** |

**Critical finding:** Models trained in one environment transfer catastrophically to another.

### 9.4 Combined (Multi-Environment) Results

| Model | MAE (mm) | 95th Pct (mm) | R-squared |
|-------|----------|---------------|-----------|
| Raw Sensor | 149.84 | 973.69 | -0.0009 |
| Linear Regression | 155.41 | 746.98 | 0.4233 |
| **Random Forest** | **23.37** | 56.07 | 0.9778 |
| **XGBoost** | **22.41** | **54.18** | **0.9811** |
| Gradient Boosting | 26.94 | 64.17 | 0.9734 |
| MLP Neural Net | 36.78 | 102.78 | 0.9447 |

**85.0% MAE reduction** with pooled multi-environment training.

### 9.5 Feature Ablation

| Feature Set | XGBoost MAE |
|-------------|-------------|
| All Features | 22.41 mm |
| Geometry Only (no telemetry) | 27.58 mm |

Removing telemetry features increased MAE by **23%**, validating the Signal-Aware hypothesis.

---

## 10. ESP32 Edge Deployment (TinyML)

### 10.1 Model Selection

| Architecture | MAE (mm) | C Code Size | Latency | Feasible? |
|-------------|----------|-------------|---------|-----------|
| XGBoost (200 trees) | 22.41 | ~40 MB | N/A | NO |
| Random Forest (100 trees) | 23.37 | ~3.7 MB | <100 us | Tight |
| Decision Tree (Depth 15) | 21.92 | 1212.7 KB | <100 us | Large |
| **Decision Tree (Depth 12)** | **23.42** | **453.8 KB** | **<100 us** | **YES** |
| Decision Tree (Depth 10) | 26.52 | 160.2 KB | <100 us | Yes |

### 10.2 Transpilation

m2cgen converts the Decision Tree to pure C function: `double score(double * input)` with 6-element input array. Zero external dependencies.

### 10.3 Compilation

- Board: ESP32 Dev Module
- Partition: "Huge APP (3MB No OTA / 1MB SPIFFS)"
- First compile: 15-25 minutes
- Subsequent: ~5 seconds (cached)

### 10.4 Performance

| Metric | Value |
|--------|-------|
| Inference latency | < 100 us |
| Sensor timing budget | 50 ms |
| ML overhead | < 0.2% of budget |
| Flash usage | ~453.8 KB |
| External dependencies | Zero |

---

## 11. 3D Point Cloud Reconstruction

Calibrated ranges converted to Cartesian coordinates on-device or in post-processing. Visualization via:
- `point_cloud_viewer.html` -- Interactive WebGL 3D viewer
- `generate_point_cloud_plots.py` -- Matplotlib 3D scatter plots, bird's-eye views, mosaics

Qualitative results: flatter wall planes, tighter object edges, fewer stray points.

---

## 12. Research Paper Evolution

### Version 1 (Day 1 Only)
- **Title:** "Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Indoor 3D Reconstruction"
- **Scope:** Single environment, 12 references
- **Key result:** XGBoost MAE 16.99 mm

### Version 2 (Day 1 + Day 2 -- Current)
- **Title:** "Multi-Environment TinyML Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Indoor 3D Reconstruction"
- **Scope:** Multi-environment, 33 references
- **Key additions:** Cross-env transfer analysis, pooled calibration, feature ablation, DT depth sweep
- **Target:** IEEE Sensors Conference or Q3/Q4 journal

---

## 13. Repository and File Structure

```
Thesis/
+-- CONTEXT.md                          # Project context
+-- THESIS_DOCUMENTATION.md             # This file
+-- README.md                           # GitHub README
+-- Research_Roadmap.md                 # Planning document
+-- raw_dataset.csv                     # Day 1 dataset (11 MB)
+-- datasets/
|   +-- scan_day_1/                     # Day 1 CSVs (10 files)
|   +-- scan_day_2/                     # Day 2 raw .txt (10 files)
|   +-- scan_day_2_processed/           # Processed CSVs
|   +-- day2_combined.csv               # Merged Day 2
|   +-- all_days_combined.csv           # All days merged
+-- ml_pipeline/
|   +-- 01_eda.py                       # Exploratory Data Analysis
|   +-- 02_train.py                     # Day 1 model training
|   +-- 03_export_to_c.py              # Model transpilation
|   +-- parse_day2_data.py              # Day 2 parser
|   +-- train_day2.py                   # Multi-env training
|   +-- train_esp32_model.py            # ESP32 DT export
|   +-- generate_point_cloud_plots.py   # 3D visualizations
|   +-- output/                         # Day 1 outputs
|   +-- output_day2/                    # Day 2 outputs
+-- lidar code/                         # ESP32 data collection firmware
|   +-- src/main.cpp
|   +-- scanner_ui.py
|   +-- platformio.ini
+-- esp32_inference/                    # ESP32 ML inference firmware
|   +-- esp32_inference.ino
|   +-- model_inference.h (453 KB)
+-- paper_day1_original/               # Paper v1 (Day 1)
+-- paper_day2_extended/               # Paper v2 (Day 1+2)
+-- Media/                              # Hardware photos/videos
+-- Protractor/protractor.html         # Protractor tool
+-- point_cloud_viewer.html            # 3D viewer
```

---

## 14. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Random Forest over XGBoost (Day 1) | Only 0.36 mm worse but fits ESP32 Flash |
| Decision Tree Depth 12 (Day 2) | Optimal accuracy-memory: 453 KB vs 40 MB |
| Predict residual, not distance | Prevents identity function learning |
| m2cgen transpilation | Zero-dependency pure C output |
| Physical protractor for GT | No reference LiDAR available |
| 80/20 sample split | Standard practice; unbiased metrics |
| No deep neural networks | Tree ensembles transpile to if/else |
| External 5V servo power | Prevents ESP32 brownouts |
| Serpentine scanning | Reduces total scan time |

---

## 15. Limitations and Failure Modes

### Limitations

1. Only two environments tested; third-environment generalization is unknown
2. Day 1 signal rate corrupted (constant 15.0 kcps logging bug)
3. Sigma unavailable in Day 2 (always -1)
4. Manual ground truth introduces label uncertainty
5. Point-level 80/20 split risks spatial data leakage (MAE may be optimistic)
6. Dataset imbalance (~85% Day 1)
7. Temperature nearly constant (32-37 C)

### Known Failure Cases

- **Matte black fabric:** Signal rate indistinguishable from noise
- **Glass and mirrors:** Specular reflections produce erratic corrections
- **Extreme tilt angles (>55 deg from horizontal):** GT assumptions break down

---

## 16. Future Work

1. Fill in author details in paper LaTeX
2. Upload raw_dataset.csv for public access
3. Submit paper to IEEE Sensors Conference
4. Test in third unseen environment (most impactful)
5. Outdoor testing (should show larger ML gains)
6. Adversarial scenes (glass, mirrors, sunlight)
7. Scan-level holdout protocol (eliminate data leakage)
8. Fix Day 1 signal rate logging
9. Fix sigma telemetry for Day 2
10. k-fold cross-validation with scan-level folds
11. Systematic feature ablation studies

---

## 17. Complete Bibliography and Citations

### Core Sensor and Calibration

[1] STMicroelectronics (2018). "VL53L1X: Time-of-Flight ranging sensor." Datasheet.
[2] Foix, S., Alenya, G., & Torras, C. (2011). "Lock-in time-of-flight (ToF) cameras: a survey." IEEE Sensors Journal, 11(9), 1917-1926.
[3] Hansard, M., Lee, S., Choi, O., & Horaud, R. (2012). Time-of-flight cameras: principles, methods and applications. Springer.
[4] He, Y., Liang, B., Zou, Y., He, J., & Yang, J. (2017). "Depth errors analysis and correction for ToF cameras." Sensors, 17(1), 92.
[5] Corti, A., Giancola, S., Mainetti, G., & Sala, R. (2016). "Metrological characterization of the Kinect V2 ToF camera." Robotics and Autonomous Systems, 75, 584-594.
[6] Lindner, M., Schiller, I., Kolb, A., & Koch, R. (2010). "Time-of-flight sensor calibration for accurate range sensing." Computer Vision and Image Understanding, 114(12), 1318-1328.
[7] Lindner, M. & Kolb, A. (2006). "Lateral and depth calibration of PMD-distance sensors." Advances in Visual Computing, 524-533.
[8] Frangez, V., Salido-Monzu, D., & Wieser, A. (2022). "Assessment and improvement of distance measurement accuracy for ToF cameras." IEEE Trans. on Instrumentation and Measurement, 71, 1-11.
[9] Kahlmann, T., Remondino, F., & Ingensand, H. (2006). "Calibration for increased accuracy of the SwissRanger." ISPRS Symposium, 36, 136-141.
[10] Fuchs, S. & Hirzinger, G. (2008). "Extrinsic and depth calibration of ToF-cameras." IEEE CVPR, 1-6.
[11] Wasenmuller, O. & Stricker, D. (2016). "Comparison of Kinect V1 and V2 depth images." ACCV, 34-45.

### Multipath Interference

[12] Sabov, A. & Kruger, J. (2008). "Identification and correction of flying pixels." Spring Conference on Computer Graphics, 135-142.
[13] Reynolds, M., Dobos, J., Peel, L., Weyrich, T., & Brostow, G. J. (2011). "Capturing ToF data with confidence." IEEE CVPR, 945-952.
[14] Dorrington, A. A., Godbaz, J. P., Cree, M. J., Payne, A. D., & Jongenelen, A. P. (2011). "Separating true range from multi-path interference." SPIE 7864, 786404.

### ML for Depth Correction

[15] Su, S., Heide, F., Wetzstein, G., & Heidrich, W. (2018). "DeepToF: Off-the-shelf deep learning for ToF sensors." IEEE CVPR, 6014-6025.
[16] Agresti, G., Schaefer, H., Sartor, P., & Zanuttigh, P. (2018). "Deep learning for multi-path error removal in ToF sensors." ECCV, 501-516.
[17] Guo, Y. et al. (2021). "Tackling multipath interference in ToF depth sensors using deep learning." Sensors, 21(4), 1-15.
[18] Mersmann, S., Seitel, A., et al. (2013). "Calibration of ToF cameras for intraoperative surface reconstruction." Medical Physics, 40(8), 082701.
[19] Jung, J., Lee, J.-Y., Jeong, Y., & Kweon, I. S. (2015). "ToF sensor calibration for a color and depth camera pair." IEEE TPAMI, 37(7), 1501-1513.

### Low-Cost LiDAR and Indoor Mapping

[20] Willis, A. R. et al. (2009). "Design of an inexpensive LIDAR scanning system." SPIE 7442, 102-111.
[21] Jimenez, D., Pizarro, D., Mazo, M., & Palazuelos, S. (2014). "3D scanner based on a ToF camera and a pan-tilt unit." Sensors, 14(4), 6330-6353.
[22] Kim, J. & Lee, S. (2015). "Design and calibration of a pan-tilt ToF scanning system." Sensors, 15(8), 20262-20281.
[23] Park, Y.-S. et al. (2019). "FPGA-based low-cost 3D lidar." IEEE Sensors, 1-4.
[24] Low, K.-L. et al. (2024). "Low-cost 3D mapping system based on 2D LiDAR and monocular cameras." Remote Sensing, 16(24), 4712.

### TinyML and Edge AI

[25] Warden, P. & Situnayake, D. (2019). TinyML: Machine Learning with TensorFlow Lite on Arduino. O'Reilly.
[26] Wang, Y. et al. (2020). "TinyML: Machine learning on ultra-low-power microcontrollers." IEEE AICAS, 25-28.
[27] Hayajneh, A. M. et al. (2024). "TinyML empowered transfer learning on the edge." IEEE OJCOMS, 5, 1656-1672.
[28] Banbury, C. R. et al. (2021). "Benchmarking TinyML systems." 2nd tinyML Research Symposium.
[29] David, R. et al. (2021). "TensorFlow Lite Micro: Embedded ML for TinyML." MLSys, 3, 800-811.
[30] Lin, J. et al. (2020). "MCUNet: Tiny Deep Learning on IoT Devices." NeurIPS, 33, 11711-11722.

### ML Algorithms and Tools

[31] Breiman, L. (2001). "Random forests." Machine Learning, 45(1), 5-32.
[32] Chen, T. & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." ACM SIGKDD, 785-794.
[33] Pedregosa, F. et al. (2011). "Scikit-learn: Machine learning in Python." JMLR, 12, 2825-2830.

### General References

[34] Amann, M.-C. et al. (2001). "Laser ranging: critical review of techniques." Optical Engineering, 40(1), 10-19.
[35] Behroozpour, B. et al. (2017). "Lidar system architectures and circuits." IEEE Comm. Magazine, 55(10), 135-142.
[36] Hess, W. et al. (2016). "Real-time loop closure in 2D LIDAR SLAM." IEEE ICRA, 1271-1278.
[37] Rusu, R. B. & Cousins, S. (2011). "3D is here: Point Cloud Library (PCL)." IEEE ICRA, 1-4.
[38] Espressif Systems (2023). ESP32 Technical Reference Manual.

---

## 18. Appendices

### Appendix A: Reproducing the ML Pipeline

```
cd ml_pipeline
pip install -r requirements.txt
python 01_eda.py                    # EDA plots
python 02_train.py                  # Day 1 training
python parse_day2_data.py           # Process Day 2
python train_day2.py                # Multi-env training
python train_esp32_model.py         # ESP32 export
python generate_point_cloud_plots.py # Visualizations
python 03_export_to_c.py            # RF to C export
```

### Appendix B: Flashing ESP32

1. Open `esp32_inference/esp32_inference.ino` in Arduino IDE
2. Board: **ESP32 Dev Module**
3. Partition: **Huge APP (3MB No OTA / 1MB SPIFFS)**
4. Upload (first compile: 15-25 min)
5. Serial Monitor at 115200 baud
6. Type condition label + Enter to start scan

### Appendix C: Dataset Access

- Day 1: `raw_dataset.csv` (11 MB, not in git)
- Day 2 raw: `datasets/scan_day_2/scan{1-10}.txt`
- Day 2 processed: `datasets/day2_combined.csv`
- Combined: `datasets/all_days_combined.csv`

### Appendix D: Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error -- average distance error in mm |
| **RMSE** | Root Mean Squared Error -- penalizes outliers |
| **R-squared** | Proportion of variance explained (1.0 = perfect) |
| **95th Percentile** | Worst-case error excluding top 5% outliers |

### Appendix E: Protractor Tool

Custom HTML protractor (`Protractor/protractor.html`) generates printable angular reference patterns.

---

> **Document last updated:** October 3, 2026
>
> **Prepared by exhaustive analysis of all files in the thesis repository.**
