#include <ESP32Servo.h>
#include <Wire.h>
#include "Adafruit_VL53L1X.h"
#include <DHT.h>
#include <Adafruit_INA219.h>

#include "model_inference.h"

// ===================== SERVOS =====================
Servo bottomServo, topServo;
const byte BOTTOM_SERVO_PIN = 13;
const byte TOP_SERVO_PIN    = 12;
const int BOTTOM_MIN = 20, BOTTOM_MAX = 160, BOTTOM_STEP = 2;  // pan: 20,22,24...160
const int TOP_MIN = 70, TOP_MAX = 150, TOP_STEP = 5;           // tilt: 170,165,160...95
const int SUBSTEP_DEGREES = 1, SUBSTEP_DELAY_MS = 8;
int currentPan = BOTTOM_MIN, currentTilt = TOP_MAX;

// ===================== CAPTURE TIMING =====================
// Exactly one capture per angle: wait, capture, wait, then move to next angle.
const unsigned long PRE_CAPTURE_DELAY_MS  = 100;
const unsigned long POST_CAPTURE_DELAY_MS = 100;

// ===================== SESSION / LOGGING =====================
long pointIndex = 0;
long sessionId = 0;
String conditionLabel = "unlabeled";

// ===================== I2C =====================
const int SDA_PIN = 21, SCL_PIN = 22;

// ===================== SENSORS =====================
Adafruit_VL53L1X distanceSensor;
Adafruit_INA219 ina219;
bool ina219Ok = false;   // set in setup(); guards every INA219 read so a wiring
                          // fault on this sensor doesn't halt/reboot the whole scan

// VL53L1X timing budget is 50ms -- a measurement is only ready once every
// ~50ms. The 400ms pre-capture wait comfortably covers this, but we still
// check dataReady() rather than assume, in case a reading is delayed.
const unsigned long TIMING_BUDGET_MS = 50;
const unsigned long DATA_READY_TIMEOUT_MS = 100; // safety cap if data isn't ready after the 400ms wait

#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);
float envTempC = -1, envHumidity = -1;

void refreshEnvironment() {
  float h = dht.readHumidity(), t = dht.readTemperature();
  if (!isnan(h) && !isnan(t)) { envHumidity=h; envTempC=t; }
}

void promptForConditionLabel() {
  Serial.println("Enter a condition label for this scan session (15s timeout):");
  unsigned long start = millis();
  while (millis() - start < 15000) {
    if (Serial.available()) {
      conditionLabel = Serial.readStringUntil('\n');
      conditionLabel.trim();
      if (conditionLabel.length() == 0) conditionLabel = "unlabeled";
      break;
    }
  }
  Serial.print("Using condition label: "); Serial.println(conditionLabel);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();
  delay(2000);
  for (int attempt=0; attempt<5; attempt++) {
    envHumidity = dht.readHumidity();
    envTempC = dht.readTemperature();
    if (!isnan(envHumidity) && !isnan(envTempC)) break;
    delay(1000);
  }

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Wire.setTimeOut(50);

  if (!distanceSensor.begin(0x29, &Wire)) {
    Serial.print("VL53L1X FAILED. Error = ");
    Serial.println(distanceSensor.vl_status);
    while (1);   // ToF sensor is mission-critical -- halting here is correct.
  }
  distanceSensor.startRanging();
  distanceSensor.setTimingBudget(TIMING_BUDGET_MS);

  ina219Ok = ina219.begin(&Wire);
  if (!ina219Ok) {
    // Non-fatal: log and continue. Power/current columns will be logged as -1
    // for this session instead of halting the scan.
    Serial.println("INA219 FAILED! Continuing without power logging.");
  }

  bottomServo.attach(BOTTOM_SERVO_PIN);
  topServo.attach(TOP_SERVO_PIN);
  smoothMoveTo(bottomServo, currentPan, BOTTOM_MIN);
  smoothMoveTo(topServo, currentTilt, TOP_MAX);
  currentPan = BOTTOM_MIN; currentTilt = TOP_MAX;
  delay(500);

  promptForConditionLabel();
  sessionId++;

  Serial.println("SCAN_START");
  Serial.println(
    "session_id,condition_label,point_index,scan_direction,"
    "pan_deg,tilt_deg,tof_mm,range_status,signal_rate_kcps,ambient_rate_kcps,sigma_mm,"
    "corrected_mm,inference_us,"
    "ina_bus_V,ina_current_mA,ina_power_mW,"
    "x,y,z,env_temp_C,humidity_pct"
  );
}

void loop() {
  bool forward = true;
  for (int tilt = TOP_MAX; tilt >= TOP_MIN; tilt -= TOP_STEP) {
    smoothMoveTo(topServo, currentTilt, tilt);
    currentTilt = tilt;
    delay(300);
    refreshEnvironment();

    const char* direction = forward ? "forward" : "backward";
    if (forward) {
      for (int pan=BOTTOM_MIN; pan<=BOTTOM_MAX; pan+=BOTTOM_STEP) scanPoint(pan, direction);
    } else {
      for (int pan=BOTTOM_MAX; pan>=BOTTOM_MIN; pan-=BOTTOM_STEP) scanPoint(pan, direction);
    }
    forward = !forward;
  }

  Serial.println("SCAN_COMPLETE");
  while (1) { delay(1000); }
}

void scanPoint(int panDeg, const char* direction) {
  // DIAGNOSTIC: confirms the ESP32 is actually issuing a new commanded angle
  // each point. Remove before a real data-collection run -- it interleaves
  // plain text into the CSV stream on Serial.
  // Serial.print("CMD pan="); Serial.print(panDeg);
  // Serial.print(" tilt="); Serial.println(currentTilt);

  // 1) MOVE to the new angle
  smoothMoveTo(bottomServo, currentPan, panDeg);
  currentPan = panDeg;

  // 2) WAIT 400ms before capturing
  delay(PRE_CAPTURE_DELAY_MS);

  // 3) CAPTURE exactly once
  int tof_mm = -1;
  uint8_t rangeStatus = 255;
  uint16_t signalRate = 0, ambientRate = 0;
  float sigma_mm = -1;

  unsigned long waitStart = millis();
  bool ready = false;
  while (millis() - waitStart < DATA_READY_TIMEOUT_MS) {
    if (distanceSensor.dataReady()) { ready = true; break; }
    delay(2);
  }

  if (ready) {
    tof_mm = distanceSensor.distance();
    distanceSensor.VL53L1X_GetRangeStatus(&rangeStatus);
    distanceSensor.VL53L1X_GetSignalRate(&signalRate);
    distanceSensor.VL53L1X_GetAmbientRate(&ambientRate);
    distanceSensor.clearInterrupt();
  }

  float corrected_mm = tof_mm;
  unsigned long t0 = micros();
  if (tof_mm > 0) {
    // Model expects features: ['tof_mm', 'pan_deg', 'tilt_deg', 'signal_rate_kcps', 'ambient_rate_kcps', 'env_temp_C']
    // Convert kcps to MCPS if necessary? The python training used kcps, so it expects kcps.
    // The m2cgen output uses a double array.
    
    double features[] = { (double)tof_mm, (double)panDeg, (double)currentTilt,
                          (double)signalRate, (double)ambientRate, (double)envTempC };
    
    // Call the generated function
    double predicted_residual = score(features);
    
    corrected_mm = tof_mm + predicted_residual;
  }
  unsigned long inference_us = micros() - t0;

  float busVoltage = ina219Ok ? ina219.getBusVoltage_V() : -1;
  float current_mA = ina219Ok ? ina219.getCurrent_mA()   : -1;
  float power_mW   = ina219Ok ? ina219.getPower_mW()     : -1;

  float x=-1, y=-1, z=-1;
  if (corrected_mm > 0) {
    float panRad = radians(panDeg);
    float tiltRad = radians(currentTilt - 90);
    x = corrected_mm*cos(tiltRad)*cos(panRad);
    y = corrected_mm*cos(tiltRad)*sin(panRad);
    z = corrected_mm*sin(tiltRad);
  }

  pointIndex++;
  Serial.print(sessionId);        Serial.print(",");
  Serial.print(conditionLabel);   Serial.print(",");
  Serial.print(pointIndex);       Serial.print(",");
  Serial.print(direction);        Serial.print(",");
  Serial.print(panDeg);           Serial.print(",");
  Serial.print(currentTilt);      Serial.print(",");
  Serial.print(tof_mm);           Serial.print(",");
  Serial.print(rangeStatus);      Serial.print(",");
  Serial.print(signalRate);       Serial.print(",");
  Serial.print(ambientRate);      Serial.print(",");
  Serial.print(sigma_mm);         Serial.print(",");
  Serial.print(corrected_mm,2);   Serial.print(",");
  Serial.print(inference_us);     Serial.print(",");
  Serial.print(busVoltage,3);     Serial.print(",");
  Serial.print(current_mA,2);     Serial.print(",");
  Serial.print(power_mW,2);       Serial.print(",");
  Serial.print(x,2);              Serial.print(",");
  Serial.print(y,2);              Serial.print(",");
  Serial.print(z,2);              Serial.print(",");
  Serial.print(envTempC,2);       Serial.print(",");
  Serial.println(envHumidity,2);

  // 4) WAIT 400ms after capturing, before the next angle
  delay(POST_CAPTURE_DELAY_MS);
}

void smoothMoveTo(Servo &servo, int from, int to) {
  if (to > from) for (int a=from; a<=to; a+=SUBSTEP_DEGREES) { servo.write(a); delay(SUBSTEP_DELAY_MS); }
  else for (int a=from; a>=to; a-=SUBSTEP_DEGREES) { servo.write(a); delay(SUBSTEP_DELAY_MS); }
  servo.write(to);
}
