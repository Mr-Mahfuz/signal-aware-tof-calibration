#include <ESP32Servo.h>
#include <Wire.h>
#include "Adafruit_VL53L1X.h"
#include <DHT.h>
#include <Adafruit_INA219.h>

// PASTE YOUR REAL TRANSPILED MODEL HERE:
// #include "model_inference.h"
// This must be the actual file from your ml_pipeline/03_export_to_c.py output.

// ===================== SERVOS =====================
Servo bottomServo, topServo;
const byte BOTTOM_SERVO_PIN = 13;
const byte TOP_SERVO_PIN    = 12;
const int BOTTOM_MIN = 20, BOTTOM_MAX = 160, BOTTOM_STEP = 1;   // Reduced from 0-180 to avoid servo stalling at physical limits
const int TOP_MIN = 100, TOP_MAX = 100, TOP_STEP = 5;           // paper's tilt range: 95-170

// --- SPEED / STABILITY OPTIMIZATIONS ---
// Increased SETTLE_DELAY to 1000 to wait 1 second after servo movement for complete physical stability
const int SETTLE_DELAY = 400; 

// Decreased SUBSTEP_DELAY_MS from 8 to 2 to make smooth movements faster
const int SUBSTEP_DEGREES = 1, SUBSTEP_DELAY_MS = 2;
int currentPan = BOTTOM_MIN, currentTilt = TOP_MAX;

// ===================== SESSION / LOGGING =====================
// Decreased from 5 to 1. Reading the sensor 5 times per point takes 5x longer.
const int REPEATS_PER_POINT = 1; 
long pointIndex = 0;
long sessionId = 0;
String conditionLabel = "unlabeled";

// ===================== I2C =====================
const int SDA_PIN = 21, SCL_PIN = 22;

// ===================== SENSORS =====================
Adafruit_VL53L1X distanceSensor;
Adafruit_INA219 ina219;
bool ina219Ok = false;

// Decreased from 50ms to 33ms to allow faster sampling rates (VL53L1X supports down to 20ms)
const unsigned long TIMING_BUDGET_MS = 33; 
const unsigned long DATA_READY_TIMEOUT_MS = 100; // budget + margin

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

void smoothMoveTo(Servo &servo, int from, int to) {
  if (to > from) for (int a=from; a<=to; a+=SUBSTEP_DEGREES) { servo.write(a); delay(SUBSTEP_DELAY_MS); }
  else for (int a=from; a>=to; a-=SUBSTEP_DEGREES) { servo.write(a); delay(SUBSTEP_DELAY_MS); }
  servo.write(to);
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
  Wire.setClock(400000); // Increased I2C clock to 400kHz (Fast Mode) for faster sensor reads
  Wire.setTimeOut(50);

  if (!distanceSensor.begin(0x29, &Wire)) {
    Serial.print("VL53L1X FAILED. Error = ");
    Serial.println(distanceSensor.vl_status);
    while (1);
  }
  distanceSensor.startRanging();
  distanceSensor.setTimingBudget(TIMING_BUDGET_MS);

  ina219Ok = ina219.begin(&Wire);
  if (!ina219Ok) {
    Serial.println("INA219 FAILED! Continuing without power logging.");
  }

  // Default ESP32Servo limits are 544 to 2400.
  // Lowering minimum to 500 pushes the servo further back to reach true 0 degrees.
  bottomServo.attach(BOTTOM_SERVO_PIN, 500, 2400);
  topServo.attach(TOP_SERVO_PIN, 500, 2400);
  smoothMoveTo(bottomServo, currentPan, BOTTOM_MIN);
  smoothMoveTo(topServo, currentTilt, TOP_MAX);
  currentPan = BOTTOM_MIN; currentTilt = TOP_MAX;
  delay(1000);

  Serial.println("\n--- SYSTEM READY ---");
  Serial.println("Send a condition label via Serial to start a scan...");
}

void scanPoint(int panDeg, const char* direction) {
  smoothMoveTo(bottomServo, currentPan, panDeg);
  currentPan = panDeg;
  
  // Wait for the servo's physical wobbling to stop completely
  delay(400);

  if (distanceSensor.dataReady()) {
    distanceSensor.distance(); // Read and discard
    distanceSensor.clearInterrupt();
  }

  for (int rep=1; rep<=REPEATS_PER_POINT; rep++) {
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
      // ==================================================================
      // PLUG IN YOUR REAL MODEL HERE.
      // ==================================================================
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
    Serial.print(rep);              Serial.print(",");
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
  }
}

void runScan() {
  sessionId++;
  pointIndex = 0; // Reset point index for the new scan

  Serial.println("SCAN_START");
  Serial.println(
    "session_id,condition_label,point_index,repeat_index,scan_direction,"
    "pan_deg,tilt_deg,tof_mm,range_status,signal_rate_kcps,ambient_rate_kcps,sigma_mm,"
    "corrected_mm,inference_us,"
    "ina_bus_V,ina_current_mA,ina_power_mW,"
    "x,y,z,env_temp_C,humidity_pct"
  );

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

  // Return servos to home position
  smoothMoveTo(bottomServo, currentPan, BOTTOM_MIN);
  smoothMoveTo(topServo, currentTilt, TOP_MAX);
  currentPan = BOTTOM_MIN; currentTilt = TOP_MAX;

  Serial.println("SCAN_COMPLETE");
  Serial.println("\n--- SYSTEM READY ---");
  Serial.println("Send a condition label via Serial to start another scan...");
}

void loop() {
  // Wait for a serial command to start a new scan
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      conditionLabel = input;
      runScan();
    }
  }
}
