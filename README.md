# Signal-Aware Machine Learning Calibration for Low-Cost ToF LiDAR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the complete source code, trained model artifacts, and LaTeX paper for the research project:

> **"Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Multi-Object Indoor Reconstruction"**

## Overview

Low-cost single-point Time-of-Flight (ToF) sensors like the VL53L1X suffer from systematic non-linear errors caused by multipath interference, ambient light, and target reflectivity. This project presents an end-to-end pipeline that:

1. **Collects** 57,000+ scan points with full signal telemetry (Signal Rate, Ambient Rate, Sigma) using a custom ESP32 pan-tilt rig.
2. **Trains** a Signal-Aware residual calibration model using Random Forest and XGBoost.
3. **Deploys** the trained model directly onto an ESP32 microcontroller as transpiled C++ code for real-time Edge AI inference (< 100 µs per ray).

## Results

| Model | MAE (mm) | 95th Percentile Error (mm) |
|-------|----------|---------------------------|
| Raw Sensor Baseline | 18.37 | 53.37 |
| Linear Regression | 18.01 | 51.30 |
| **Random Forest (Edge Deployed)** | **17.35** | **49.08** |
| XGBoost (Theoretical Max) | 16.99 | 48.80 |

## Repository Structure

```
├── ml_pipeline/              # Machine Learning training pipeline
│   ├── 01_eda.py             # Exploratory Data Analysis
│   ├── 02_train.py           # Model training (RF, XGBoost, Linear)
│   ├── 03_export_to_c.py     # Transpile model to C code via m2cgen
│   ├── requirements.txt      # Python dependencies
│   └── output/               # Generated plots and documentation
│
├── Lidar_ML_Scanner/         # ESP32 Arduino project (ready to flash)
│   ├── Lidar_ML_Scanner.ino  # Main firmware with ML integration
│   ├── model_inference.c     # Transpiled Random Forest model
│   └── model_inference.h     # C header for the inference function
│
├── paper/                    # IEEE LaTeX source
│   ├── main.tex              # Full paper source
│   ├── references.bib        # Bibliography
│   └── figures/              # All figures used in the paper
│
└── README.md
```

## Hardware Requirements

- **ESP32** DevKit V1 (or compatible)
- **VL53L1X** ToF sensor (TOF400 module)
- **2x SG90** micro servo motors (pan + tilt)
- **DHT11** temperature sensor
- **INA219** current monitor
- External **5V regulated** power supply

## Getting Started

### ML Pipeline (Training)
```bash
cd ml_pipeline
pip install -r requirements.txt
python 01_eda.py
python 02_train.py
python 03_export_to_c.py
```

### ESP32 Firmware (Deployment)
1. Open `Lidar_ML_Scanner/Lidar_ML_Scanner.ino` in Arduino IDE.
2. Select Board: **ESP32 Dev Module**.
3. Set Partition Scheme: **Huge APP (3MB No OTA/1MB SPIFFS)**.
4. Click **Upload** (first compilation takes ~15-20 minutes due to the large model file).

## Citation

If you use this work in your research, please cite:
```bibtex
@inproceedings{yourname2026signalaware,
  title={Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt 
         Time-of-Flight LiDAR for Multi-Object Indoor Reconstruction},
  author={Your Name and Co-Author Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
