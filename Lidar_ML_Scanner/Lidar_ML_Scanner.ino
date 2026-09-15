#include <ESP32Servo.h>
#include <Wire.h>
#include "Adafruit_VL53L1X.h"
#include "model_inference.h"

// =====================================================
// SERVO PINS
// =====================================================

Servo panServo;
Servo tiltServo;

const int PAN_SERVO_PIN  = 13;
const int TILT_SERVO_PIN = 12;

// =====================================================
// SCAN SETTINGS
// =====================================================

const int PAN_MIN  = 0;
const int PAN_MAX  = 180;
const int PAN_STEP = 1;

const int TILT_MIN  = 80;
const int TILT_MAX  = 170;
const int TILT_STEP = 5;

// =====================================================
// SERVO SETTINGS
// =====================================================

const int SERVO_MOVE_DELAY = 10;
const int SETTLE_DELAY = 120;
const int TILT_SETTLE_DELAY = 250;

// =====================================================
// I2C
// =====================================================

const int SDA_PIN = 21;
const int SCL_PIN = 22;

// =====================================================
// VL53L1X
// =====================================================

Adafruit_VL53L1X tof;
const uint8_t TOF_ADDRESS = 0x29;

// =====================================================
// CURRENT POSITION
// =====================================================

int currentPan  = PAN_MIN;
int currentTilt = TILT_MAX;

// =====================================================
// MOVE SERVO SAFELY
// =====================================================

void moveServoSmooth(Servo &servo, int &currentPosition, int targetPosition)
{
  targetPosition = constrain(targetPosition, 0, 180);
  if (targetPosition == currentPosition) return;

  int direction = (targetPosition > currentPosition) ? 1 : -1;

  while (currentPosition != targetPosition)
  {
    currentPosition += direction;
    servo.write(currentPosition);
    delay(SERVO_MOVE_DELAY);
    yield();
  }
  servo.write(targetPosition);
  delay(SETTLE_DELAY);
}


// =====================================================
// ML FEATURE EXTRACTION WRAPPER
// =====================================================
// Note to developer: Depending on your exact VL53L1X library 
// (Adafruit vs Pololu vs Sparkfun), these specific getter functions 
// might have slightly different names. You need to extract these 3 values.

float get_signal_rate_kcps() {
  // Using Adafruit VL53L1X underlying ST API
  // Convert from Mcps (Mega counts) to kcps (Kilo counts) if necessary
  // If your library returns Mcps, do: return val * 1000.0;
  // This is a placeholder. Update based on your library's exact API:
  return 15.0; // tof.vl53l1x_getSignalRate();
}

float get_ambient_rate_kcps() {
  // Update based on your library's exact API:
  return 20.0; // tof.vl53l1x_getAmbientRate();
}

float get_sigma_mm() {
  // Update based on your library's exact API:
  return 3.0; // tof.vl53l1x_getSigma();
}


// =====================================================
// TAKE ONE SCAN MEASUREMENT
// =====================================================

void scanPoint(int targetPan)
{
  moveServoSmooth(panServo, currentPan, targetPan);

  unsigned long start = millis();
  while (!tof.dataReady())
  {
    if (millis() - start > 250) return; // Timeout
    delay(1);
    yield();
  }

  // 1. Read Raw Distance
  int raw_distance_mm = tof.distance();
  tof.clearInterrupt();

  if (raw_distance_mm <= 0 || raw_distance_mm > 4000) return;

  // 2. Read Signal Metrics
  float signal_rate_kcps = get_signal_rate_kcps();
  float ambient_rate_kcps = get_ambient_rate_kcps();
  float sigma_mm = get_sigma_mm();

  // 3. Prepare features for ML Model
  double features[6];
  features[0] = (double)raw_distance_mm;
  features[1] = (double)currentPan;
  features[2] = (double)currentTilt;
  features[3] = (double)signal_rate_kcps;
  features[4] = (double)ambient_rate_kcps;
  features[5] = (double)sigma_mm;

  // 4. Run ML Inference!
  double predicted_error_mm = score(features);
  
  // 5. Apply Calibration
  float calibrated_distance = raw_distance_mm + predicted_error_mm;

  // 6. XYZ Math
  // If 170 degrees points UP: tiltRad = radians(currentTilt - 90.0)
  // If 170 degrees points DOWN: tiltRad = radians(90.0 - currentTilt)
  float panRad = radians((float)targetPan);
  float tiltRad = radians((float)currentTilt - 90.0); 

  float x = calibrated_distance * cos(tiltRad) * cos(panRad);
  float y = calibrated_distance * cos(tiltRad) * sin(panRad);
  float z = calibrated_distance * sin(tiltRad);

  // 7. Print CSV
  Serial.print(targetPan); Serial.print(",");
  Serial.print(currentTilt); Serial.print(",");
  Serial.print(raw_distance_mm); Serial.print(",");
  Serial.print(predicted_error_mm, 2); Serial.print(",");
  Serial.print(calibrated_distance, 2); Serial.print(",");
  Serial.print(x, 2); Serial.print(",");
  Serial.print(y, 2); Serial.print(",");
  Serial.println(z, 2);
}

// =====================================================
// SCAN ROW FORWARD/BACKWARD
// =====================================================

void scanForward()
{
  for (int pan = PAN_MIN; pan <= PAN_MAX; pan += PAN_STEP) {
    scanPoint(pan); yield();
  }
}

void scanBackward()
{
  for (int pan = PAN_MAX; pan >= PAN_MIN; pan -= PAN_STEP) {
    scanPoint(pan); yield();
  }
}

// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println(" ML-CALIBRATED TOF 2-AXIS SCANNER");
  Serial.println("========================================");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  if (!tof.begin(TOF_ADDRESS, &Wire)) {
    Serial.println("ERROR: VL53L1X NOT DETECTED!");
    while (1) delay(1000);
  }

  if (!tof.setTimingBudget(50)) {
    Serial.println("WARNING: Timing budget failed.");
  }

  if (!tof.startRanging()) {
    Serial.println("ERROR: TOF ranging failed!");
    while (1) delay(1000);
  }

  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_SERVO_PIN, 500, 2500);
  tiltServo.attach(TILT_SERVO_PIN, 500, 2500);

  currentPan = PAN_MIN;
  currentTilt = TILT_MAX;
  panServo.write(currentPan);
  tiltServo.write(currentTilt);
  delay(1000);

  // Updated CSV Header
  Serial.println("pan_deg,tilt_deg,raw_distance_mm,ml_error_mm,calibrated_dist_mm,x_mm,y_mm,z_mm");
  Serial.println("SCAN_START");
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  bool forward = true;

  for (int tilt = TILT_MAX; tilt >= TILT_MIN; tilt -= TILT_STEP)
  {
    moveServoSmooth(tiltServo, currentTilt, tilt);
    delay(TILT_SETTLE_DELAY);

    if (forward) scanForward();
    else scanBackward();

    forward = !forward;
    yield();
  }

  Serial.println("\n========================================");
  Serial.println("SCAN COMPLETE");
  Serial.println("========================================");

  while (true) delay(1000);
}
