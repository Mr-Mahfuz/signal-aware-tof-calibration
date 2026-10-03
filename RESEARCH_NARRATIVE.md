# What We Did and How We Did It

## Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR

---

## 1. The Problem We Set Out to Solve

3D mapping of indoor spaces is essential for robotics, augmented reality, structural inspection, and autonomous navigation. Commercial LiDAR systems that do this well cost anywhere from $500 to $10,000+. We wanted to know: **can we build a functional 3D LiDAR scanner for under $15, and then use machine learning to make it accurate enough to be useful?**

The cheapest path to a 3D scanner is to take a single-point Time-of-Flight (ToF) sensor — specifically the VL53L1X, which costs about $4 — mount it on two cheap servo motors (a pan-tilt rig), and sweep it across a room, collecting one distance measurement at a time. Each measurement gets a pan angle and tilt angle, which lets us convert it into a 3D (x, y, z) point. Do this thousands of times and you get a point cloud — a 3D map of the room.

The problem is that the VL53L1X makes mistakes. Its datasheet promises +/-3% accuracy under ideal conditions, but in real indoor environments, the errors get much worse due to:

- **Multipath interference:** Photons bouncing off multiple surfaces before returning to the sensor, inflating the measured distance
- **Surface reflectivity:** Dark surfaces return fewer photons, increasing measurement noise. Glass and mirrors scatter photons unpredictably
- **Ambient light:** Room lighting and sunlight add noise to the sensor's photodetector array
- **Mechanical vibration:** The cheap SG90 servo motors wobble after each movement, and have angular backlash

Traditional calibration uses simple look-up tables that correct errors based only on measured distance. But these tables assume the error only depends on distance — which is wrong. The real error depends on distance, angle, surface material, lighting, and temperature all at once.

### Our Key Insight

The VL53L1X sensor doesn't just report a distance number. It also exposes internal diagnostic signals with every measurement:

- **Signal Rate** (kilo-counts per second): How many photons came back. Tells you about target reflectivity and distance
- **Ambient Rate** (kcps): How much background infrared noise the sensor is seeing. Tells you about lighting conditions
- **Sigma** (mm): The sensor's own estimate of how confident it is in the measurement

We hypothesized that these internal signals contain enough information about *why* the sensor is making an error to allow a machine learning model to predict and correct that error. We called this **"Signal-Aware Calibration"** — because the model uses the sensor's own signal quality assessment, not just the distance value.

---

## 2. What We Built (Hardware)

We assembled a scanning rig from the following components, totaling under $15:

- **VL53L1X ToF sensor** (on a TOF400 breakout board) — measures distance and exposes signal/ambient/sigma telemetry via I2C
- **ESP32 microcontroller** (DevKit V1, WROOM-32) — runs the firmware, controls servos, and later runs the ML model on-chip
- **Two SG90 micro servo motors** — one for horizontal pan (0-180 degrees), one for vertical tilt
- **DHT11 temperature/humidity sensor** — logs ambient temperature to track thermal drift
- **INA219 current/power monitor** — logs power consumption during scanning
- **External 5V power supply** — critical for servo power (see lesson below)

The sensor is mounted on the tilt servo, which is mounted on the pan servo. The sensor's optical center sits at the intersection of both rotation axes, defining the origin (0, 0, 0) of our coordinate system.

### Critical Hardware Lesson

We learned early that **you cannot power SG90 servos from the ESP32's onboard regulator**. The servos draw current spikes that cause voltage brownouts, crashing the ESP32 within minutes. In both campaigns, we used an Arduino Nano purely as a regulated 5V power source for the servos, keeping the servo power rail electrically isolated from the ESP32.

### Firmware

We wrote two firmware versions in C++ for the ESP32:

**Version 1 — Data Collection Firmware:** This version controls the servos in a serpentine (boustrophedon) scan pattern. At each (pan, tilt) position, it waits for the servo vibration to settle (400 ms), reads the VL53L1X sensor (distance + signal rate + ambient rate + range status), reads the DHT11 (temperature, humidity), reads the INA219 (voltage, current, power), computes Cartesian (x, y, z) coordinates from the spherical measurement, and prints everything as a CSV row over serial at 115200 baud.

The scan pattern sweeps left-to-right on the first tilt row, then right-to-left on the next, alternating to minimize total scan time. Servo movements use smooth 1-degree sub-stepping at 2 ms per step to reduce vibration.

**Version 2 — Inference Firmware:** This version includes the trained ML model (transpiled to C code) as a header file. For each measurement, it packs the 6 features into an array, calls the model's `score()` function to get a predicted error, adds the correction to the raw distance, and outputs both the raw and corrected values.

### Scanner Control UI

We also built a Python GUI application using Tkinter that connects to the ESP32 over serial. It auto-detects COM ports, lets you type a condition label (e.g., "dark_box" or "bright_dwall"), sends it to trigger a scan, shows real-time progress, and automatically saves all incoming data to a timestamped CSV file. It flushes to disk every 10 points so data isn't lost if the USB cable gets disconnected mid-scan.

---

## 3. How We Established Ground Truth

Since we didn't have a reference-grade commercial LiDAR to compare against, we derived ground truth geometrically:

1. We placed a large custom-printed protractor grid on the floor directly beneath the sensor, establishing angular reference lines every 5 degrees from 0 to 180 degrees
2. We physically measured the distance from the sensor to the nearest wall surface at each 5-degree pan angle using a steel tape measure and a laser distance meter (both accurate to +/-1 mm)
3. For any pan angle between the 5-degree marks, we linearly interpolated the wall distance
4. For non-horizontal tilt angles, we applied a tilt correction: the beam travels a longer path to hit a vertical wall when tilted. The corrected ground truth is: `gt_at_tilt = gt_horizontal / cos(|tilt - 90 degrees|)`
5. At extreme tilt angles (more than 55 degrees from horizontal), the beam hits the ceiling or floor rather than the wall, so those points were flagged as unreliable

For the Day 2 campaign, we also placed physical objects in the scene:
- A **cardboard box** (31 cm tall) at pan angles 35-80 degrees — present in scans 5 through 8
- A **translucent plastic object** (24 cm tall) at pan angles 55-80 degrees — present in scan 10

The box and transparent object distances were also measured and encoded as separate ground truth tables.

### Residual Error Computation

For every valid sensor measurement that has a corresponding ground truth value, we computed the residual error:

```
residual_error = ground_truth_distance - raw_sensor_distance
```

A positive residual means the sensor is reading short (underestimating). A negative residual means the sensor is reading long (overestimating). This residual is what our ML model learns to predict.

---

## 4. Data Collection — Two Campaigns

We collected data in two separate indoor environments on different days, deliberately choosing different conditions to test how well calibration transfers across environments.

### Day 1 — Controlled Environment

- **Setting:** A relatively dark, controlled room with uniform painted walls
- **Scans:** 10 scanning sessions covering the full pan range (0-180 degrees at 1-degree steps) and tilt range (0-179.9 degrees)
- **Total data:** 57,352 raw measurements, of which 53,539 had valid sensor returns and approximately 51,000 were trainable (valid return + valid ground truth)
- **Temperature:** 32-37 degrees C throughout
- **Known data issue:** Due to a firmware logging bug, the signal rate column was recorded as a constant 15.0 kcps for every single measurement. The sensor was actually measuring varying signal rates, but the firmware was not correctly extracting and logging the value from the sensor's API. This means the most important "signal-aware" feature was corrupted for the entire Day 1 dataset.

### Day 2 — Diverse Environment

The Day 2 campaign was specifically designed to stress-test the sensor under challenging and varied conditions. We collected 10 scans, each under a different environmental configuration:

| Scan | What We Did | Why |
|------|------------|-----|
| 1 | Dimmed lights to 50% brightness | Test low-light performance |
| 2 | Full room lights on, clean walls | Establish a bright baseline |
| 3 | Full bright, verified ground truth | Cross-check our protractor measurements |
| 4 | Changed lighting partway through scan | Test how the model handles mid-scan lighting changes |
| 5 | Turned all lights off, placed a cardboard box | Test dark conditions + obstacle detection |
| 6 | All dark, box + covered one wall with dark fabric | Test worst-case: dark room + low-reflectivity surfaces |
| 7 | Full bright, box + dark wall (28cm gap at bottom) | Bright room + challenging surfaces |
| 8 | Full bright, box + dark wall (fully covered bottom) | Variation of scan 7 |
| 9 | Full bright, dark wall only (no box) | Isolate the effect of low-reflectivity walls |
| 10 | Full bright, transparent plastic object + dark wall | Test translucent objects (partial photon pass-through) |

- **Total data:** 11,337 raw measurements, 9,104 valid returns, 8,477 trainable
- **Pan range:** 20-160 degrees at 2-degree steps
- **Tilt range:** 70-150 degrees at 5-degree steps
- **Signal telemetry:** Properly logged this time (bug fixed)
- **Sigma:** Always returned -1 from the sensor API (unavailable in this firmware version)

### Data Processing

The Day 2 raw data came as 10 text files of serial output from the ESP32, full of debug lines like "INA219 FAILED" and "SCAN_START". We wrote a parser script that:
1. Strips all non-data lines (debug messages, scan markers, command echoes)
2. Handles the case where scan 7 was missing its CSV header line
3. Converts all numeric columns from strings to proper types
4. Assigns ground truth to every data point based on its (pan, tilt, scan_id)
5. Computes residual errors
6. Computes Cartesian (x, y, z) coordinates for both raw and ground truth distances
7. Outputs individual processed CSVs, a merged Day 2 file, and a combined Day 1 + Day 2 file

---

## 5. Machine Learning — How We Trained the Models

### Problem Formulation

We formulated the calibration task as **residual prediction**: the model takes sensor measurements and predicts how far off the sensor's reading is from the true distance. The corrected distance is then: `corrected = raw + predicted_residual`.

We tried training models to predict the absolute ground truth distance directly, but this failed — the model simply learned the identity function (output = raw distance) and ignored all the signal features. By forcing the model to predict only the *error*, it had to learn what causes errors rather than just echoing the input.

### Features We Used

Every prediction uses 6 input features:

| Feature | What It Captures |
|---------|-----------------|
| Raw distance (mm) | Error magnitude scales with range |
| Pan angle (degrees) | Angular/positional error patterns |
| Tilt angle (degrees) | Tilt-dependent distortion |
| Signal rate (kcps) | Target reflectivity, MPI, distance |
| Ambient rate (kcps) | Lighting conditions |
| Temperature (degrees C) | Thermal drift effects |

### Models We Trained

**Day 1 pipeline (3 models):**
- Linear Regression — a baseline to test if the relationship is linear (it mostly isn't)
- Random Forest (50 trees, depth 10) — handles nonlinear interactions between features
- XGBoost (100 trees, depth 6) — gradient-boosted trees, state of the art for tabular data

**Day 2 / Combined pipeline (5 models per scenario):**
- Linear Regression
- Random Forest (100 trees, depth 12)
- XGBoost (200 trees, depth 8)
- Gradient Boosting (150 trees, depth 6)
- MLP Neural Network (2 hidden layers: 128 and 64 neurons)

### Training Scenarios

For the multi-environment analysis, we ran four different training/testing configurations to answer different questions:

1. **Day 2 only** (80/20 split) — How accurate can we get within the diverse environment?
2. **Train on Day 1, test on Day 2** — Does a model trained in one room work in another?
3. **Train on Day 2, test on Day 1** — Does the reverse transfer work?
4. **Combined Day 1 + Day 2** (80/20 split) — Does pooling data from both environments help?

All splits were 80% training / 20% testing at the individual measurement level, with a fixed random seed (42) for reproducibility.

### ESP32-Targeted Training

For the model that would actually run on the ESP32, we trained Decision Trees at three different maximum depths (10, 12, and 15) because Decision Trees produce much smaller C code than Random Forests or XGBoost. We selected depth 12 as the best trade-off between accuracy and code size.

---

## 6. Results — What We Found

### Day 1: The Sensor Was Already Pretty Good

In the controlled environment, the raw sensor's average error was only 18.37 mm. Our best model (XGBoost) reduced this to 16.99 mm — a modest 1.4 mm improvement. The improvement was small because the sensor was already performing near its ceiling under ideal conditions.

| Model | Average Error | Worst-Case (95th percentile) |
|-------|--------------|------------------------------|
| Raw Sensor | 18.37 mm | 53.37 mm |
| Linear Regression | 18.01 mm | 51.30 mm |
| Random Forest | 17.35 mm | 49.08 mm |
| XGBoost | 16.99 mm | 48.80 mm |

### Day 2: ML Made a Massive Difference

In the diverse environment with varied lighting, obstacles, and surfaces, the raw sensor's average error was a staggering 934.97 mm (nearly a full meter off). Here, XGBoost reduced the error to 49.34 mm — a **94.7% reduction**.

| Model | Average Error | Worst-Case (95th pct) | R-squared |
|-------|--------------|----------------------|-----------|
| Raw Sensor | 934.97 mm | 2617.87 mm | -0.0085 |
| Linear Regression | 367.94 mm | 1040.66 mm | 0.8251 |
| Random Forest | 65.22 mm | 376.96 mm | 0.9757 |
| XGBoost | 49.34 mm | 203.70 mm | 0.9854 |
| Gradient Boosting | 77.01 mm | 425.26 mm | 0.9753 |
| MLP Neural Net | 166.40 mm | 750.32 mm | 0.9419 |

### Cross-Environment Transfer: Catastrophic Failure

This was our most important finding. When we trained a model in one environment and tested it in the other, **the model performed worse than using the raw sensor with no correction at all**:

- Train Day 1, Test Day 2: XGBoost MAE = **947.15 mm** (raw sensor was 934.97 mm — the model made things worse)
- Train Day 2, Test Day 1: XGBoost MAE = **1,049.09 mm** (raw sensor was only 18.24 mm — the model completely destroyed the accuracy)

This happened because the error patterns in each environment are fundamentally different. The model learned environment-specific corrections that became harmful when applied to a different environment. The ~1,300 mm systematic bias between environments is likely due to different wall geometries and ground truth calculation mismatches.

### Pooled Training: The Solution

When we combined data from both environments into a single training set, the model learned generalizable corrections:

| Model | Average Error | Worst-Case | R-squared |
|-------|--------------|------------|-----------|
| Raw Sensor | 149.84 mm | 973.69 mm | -0.0009 |
| XGBoost | **22.41 mm** | **54.18 mm** | **0.9811** |
| Random Forest | 23.37 mm | 56.07 mm | 0.9778 |

This is an **85% reduction** in average error from the raw sensor.

### Feature Ablation: Do Signal Features Actually Help?

To prove that the signal telemetry features (not just distance and angles) contribute to accuracy, we trained XGBoost with and without them:

- All 6 features: **22.41 mm** MAE
- Geometry only (distance + pan + tilt): **27.58 mm** MAE

Removing signal features increased the error by **23%**, confirming that signal rate, ambient rate, and temperature provide measurable calibration value. This validates our "Signal-Aware" hypothesis.

### Feature Importance Ranking

XGBoost's built-in feature importance analysis showed:
1. **Raw Distance** — the dominant feature (expected: error scales with range)
2. **Pan Angle** — captures angular error patterns
3. **Signal Rate** — critical for environment/reflectivity discrimination
4. **Tilt Angle** — captures tilt-dependent distortions
5. **Ambient Rate** — captures lighting condition effects
6. **Temperature** — minor but measurable contribution

---

## 7. Deploying the Model onto the ESP32

### The Challenge

The best model (XGBoost with 200 trees at depth 8) would produce approximately 40 MB of C code when transpiled — far too large for the ESP32's 4 MB flash memory. We needed a model small enough to fit on the chip but accurate enough to be useful.

### The Solution: Model Transpilation with m2cgen

We used a tool called `m2cgen` (Model to C Code Generator) that converts trained scikit-learn models into pure C functions consisting entirely of nested if/else statements. This approach has zero external dependencies — no TensorFlow Lite, no ONNX runtime, no math libraries. Just a single C function.

### Size vs Accuracy Trade-off

We tested three Decision Tree depths:

| Depth | Average Error | C Code Size | Fits ESP32? |
|-------|--------------|-------------|-------------|
| 10 | 26.52 mm | 160.2 KB | Yes (comfortable) |
| **12** | **23.42 mm** | **453.8 KB** | **Yes (optimal)** |
| 15 | 21.92 mm | 1,212.7 KB | Yes (tight) |

We chose **depth 12** because it sacrifices only about 1 mm of accuracy compared to XGBoost while producing code that comfortably fits in the ESP32's flash.

### How the On-Device Inference Works

The transpiled model generates a C function called `score()` that takes a 6-element double array and returns the predicted residual error in millimeters. In the ESP32 inference firmware, each measurement cycle:

1. The servo moves to the next (pan, tilt) position
2. The firmware waits for physical vibration to settle
3. The VL53L1X reports: raw distance, signal rate, ambient rate
4. The firmware packs these into an array: [distance, pan, tilt, signal_rate, ambient_rate, temperature]
5. It calls `score(features)` which returns the predicted error
6. The corrected distance = raw distance + predicted error
7. The (x, y, z) point is computed from the corrected distance
8. Everything is printed as a CSV row

### Performance on the ESP32

- **Inference latency:** Less than 100 microseconds per prediction (measured with `micros()`)
- **Sensor timing budget:** 50 milliseconds per ranging cycle
- **ML overhead:** Less than 0.2% of the sensor's dead time — completely negligible
- **Partition scheme:** Required "Huge APP (3MB No OTA / 1MB SPIFFS)" to fit the generated code
- **First compile time:** 15-25 minutes (the GCC compiler optimizing thousands of if/else branches)
- **Subsequent compiles:** About 5 seconds (the object file is cached)

The model adds no perceivable delay to the scanning process. It runs entirely within the sensor's own dead time between measurements.

---

## 8. 3D Point Cloud Reconstruction

After calibration, each corrected (distance, pan, tilt) measurement is converted to a 3D Cartesian point using the spherical-to-Cartesian transform. The collection of all points forms a 3D point cloud representing the scanned environment.

We built two visualization tools:

1. **An interactive HTML viewer** that loads the CSV data and renders the point cloud in 3D using WebGL, allowing the user to rotate, zoom, and inspect the scan from any angle

2. **A Python plotting script** that generates static 3D scatter plots using matplotlib, including:
   - Per-scan point clouds colored by distance (showing room geometry)
   - Per-scan point clouds colored by residual error (showing where the sensor struggled)
   - Bird's-eye (top-down) views showing the angular sweep pattern
   - Side-by-side comparisons of raw vs ground truth point clouds
   - 2x2 mosaics comparing different scan conditions
   - Error vs pan angle line plots showing how error varies with direction

In the controlled environment, calibration produced visibly flatter wall surfaces and tighter object edges. In the diverse environment, calibration transformed noisy, distorted data into recognizable room geometry — the walls, the box, and the overall room shape became clearly visible after correction.

---

## 9. What We Learned (Key Takeaways)

### Signal-Aware Calibration Works

The central hypothesis was validated. Feeding the sensor's internal photon telemetry (signal rate, ambient rate) into an ML model alongside geometric features (distance, pan, tilt) produced measurably better calibration than geometry alone. The 23% MAE increase when removing signal features proves this is not coincidental.

### Environment-Specific Models Are Dangerous

Single-environment calibration models transfer catastrophically to new environments. A model trained in Room A and applied in Room B can make accuracy *worse* than using no calibration at all. This is because the error patterns — and the relationship between signal features and errors — change fundamentally with the environment.

### Multi-Environment Pooling Is Essential

The solution to the transfer problem is to train on data from multiple environments. Our pooled model (Day 1 + Day 2 combined) achieved 22.41 mm MAE on the combined test set — dramatically better than either single-environment model applied cross-environment.

### Tree Models Are Perfect for Microcontroller Deployment

Decision Trees and Random Forests transpile to pure C if/else code with zero dependencies. This is far simpler than deploying neural networks (which require TFLite Micro, quantization, and runtime libraries). The depth-12 Decision Tree achieved near-XGBoost accuracy (23.42 vs 22.41 mm) while fitting comfortably in 453.8 KB of flash.

### The Inference Is Free

At less than 100 microseconds, the ML inference is completely invisible in the timing budget. The sensor takes 50 milliseconds between measurements — the model runs in less than 0.2% of that time. There is no practical cost to adding ML calibration.

### Cheap Hardware Can Produce Useful 3D Maps

A $15 hardware setup — when properly calibrated with ML — can produce 3D point clouds that clearly resolve room geometry, walls, and obstacles. The resolution is limited by the scanning step size (not the ML accuracy), and the accuracy is limited primarily by the ground truth methodology (not the sensor or model).

---

## 10. Honest Limitations of Our Work

1. **Only two environments tested.** We showed that pooled training across two environments works, but we don't know how the model performs in a completely unseen third environment. This is the most important open question.

2. **Day 1 signal rate was corrupted.** A firmware logging bug recorded constant 15.0 kcps for every measurement, meaning the signal_rate feature was uninformative for ~85% of our training data. The true contribution of signal rate is likely even higher than our ablation study suggests.

3. **Sigma was unavailable in Day 2.** The VL53L1X's sigma (confidence) metric returned -1 for all Day 2 measurements. This potentially valuable feature was never usable.

4. **Manual ground truth has uncertainty.** Our ground truth comes from tape measure + protractor, not a reference-grade LiDAR. Angular alignment errors (~1 degree), surface measurement errors (~1 mm), and the assumption that walls are perfectly flat all introduce systematic GT uncertainty.

5. **Risk of spatial data leakage.** Our 80/20 train-test split was at the individual measurement level, not at the scan or scene level. Spatially adjacent points from the same scan can appear in both training and test sets. This means our reported MAE values (22.41 mm) are likely **optimistic**. A scan-level holdout would give more conservative and more honest numbers.

6. **Dataset imbalance.** Day 1 contributes ~85% of the combined dataset. The combined metric is heavily weighted toward the controlled environment, which had much smaller errors.

7. **Temperature was nearly constant.** The temperature only varied between 32-37 degrees C across all our data. We can't claim the model handles large thermal excursions.

### Known Failure Cases

- **Matte black fabric:** Returns so few photons that the signal rate is indistinguishable from noise. The model has no useful signal features to work with.
- **Glass and mirrors:** Specular reflections produce erratic photon counts. The model sometimes applies corrections larger than the original error.
- **Extreme tilt angles (>55 degrees from horizontal):** The beam hits the ceiling or floor, not the wall. Our ground truth assumptions break down here.
- **Pan 0-45 degrees in Day 2 scans 3, 9, 10:** Consistently returned -1 (no valid measurement), likely due to the beam hitting a highly absorptive or glass surface at those angles.

---

## 11. Exploratory Data Analysis — What the Data Looked Like

Before training any models, we performed exploratory data analysis to understand the characteristics of our datasets:

### Error Distribution
The raw sensor errors in Day 1 followed a roughly normal distribution centered near zero, with most errors between -50 mm and +50 mm. Day 2 errors were dramatically larger and more spread out, reflecting the diverse and challenging conditions.

### Feature Correlations
A correlation heatmap revealed that raw distance and ground truth distance were very strongly correlated (as expected), while signal rate showed moderate negative correlation with distance (photon returns weaken with range). Ambient rate was largely uncorrelated with other features, confirming it captures independent information about lighting conditions.

### Error vs Signal Rate
Scatter plots of residual error against signal rate showed that the largest errors occurred at low signal rates (below ~10 kcps), where the sensor is operating with minimal photon return. This visual confirmation supported the hypothesis that signal rate contains calibration-relevant information.

### Raw vs Ground Truth
Scatter plots of raw distance vs ground truth distance showed strong linearity along the ideal 1:1 line in Day 1 (confirming generally good sensor performance), but significant scatter and systematic offsets in Day 2 (confirming the need for environment-aware calibration).

---

## 12. The Evolution of Our Research

### Phase 1: Single-Environment Proof of Concept

We started with the Day 1 data only. We trained three models (Linear Regression, Random Forest, XGBoost), demonstrated a modest but real improvement over the raw sensor, transpiled the Random Forest to C code, and deployed it on the ESP32. We wrote a research paper documenting this work with the title "Signal-Aware Machine Learning Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Indoor 3D Reconstruction."

The key result was proof of concept: the signal-aware approach works, and ML models can run on a $3 microcontroller.

### Phase 2: Multi-Environment Extension

We then designed and executed the Day 2 campaign specifically to answer the question: **does this calibration generalize to new environments?** The answer was a definitive and scientifically important "no" — at least not when trained on only one environment.

This led us to the pooled training approach, which we demonstrated works far better. We expanded the model comparison to 5 algorithms, added the cross-environment transfer analysis, performed feature ablation, optimized the ESP32 model size with Decision Tree depth sweeps, and wrote an extended paper titled "Multi-Environment TinyML Calibration of a Low-Cost Pan-Tilt Time-of-Flight LiDAR for Indoor 3D Reconstruction."

The Phase 2 contribution is arguably more significant than Phase 1: the cross-environment failure analysis and the pooled training solution are the most publishable and novel results of the entire thesis.

---

> **Summary:** We built a $15 LiDAR scanner, collected 68,000+ measurements across two different indoor environments under 20 different conditions, trained ML models that reduce measurement error by up to 94.7%, discovered that single-environment models fail catastrophically in new environments, solved this with multi-environment pooled training (85% error reduction), and deployed the model on a $3 ESP32 microcontroller with less than 100 microsecond inference latency and zero external dependencies.
