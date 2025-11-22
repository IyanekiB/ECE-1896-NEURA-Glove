/*
 * NEURA GLOVE - ESP32 Firmware
 * 
 * Complete firmware for ESP32-WROOM-32UE with:
 * - 5 Flex Sensors (ADC channels)
 * - BNO085 9-DOF IMU (I2C)
 * - DRV2605 Haptic Driver (I2C)
 * - Joystick (I2C)
 * - BLE Communication at 10Hz
 * 
 * Data Format (19 values, comma-separated):
 * flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz,ax,ay,az,gx,gy,gz,joyX,joyY,zPressed,cPressed
 * 
 * Author: NEURA GLOVE Team
 * Date: 2025
 */

#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <Adafruit_DRV2605.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================

// Flex Sensors (ADC pins)
#define FLEX1_PIN 34  // ADC1_CH6 - Thumb
#define FLEX2_PIN 35  // ADC1_CH7 - Index
#define FLEX3_PIN 32  // ADC1_CH4 - Middle
#define FLEX4_PIN 33  // ADC1_CH5 - Ring
#define FLEX5_PIN 25  // ADC2_CH8 - Pinky

// I2C Pins
#define SDA_PIN 21
#define SCL_PIN 22

// I2C Addresses
#define BNO085_ADDRESS 0x4A
#define DRV2605_ADDRESS 0x5A
#define JOYSTICK_ADDRESS 0x52

// LED Pin for status indication
#define LED_PIN 2

#define BNO08X_RESET -1

// ============================================================================
// BLE CONFIGURATION
// ============================================================================

#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" // ESP32 -> PC
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E" // PC -> ESP32
#define DEVICE_NAME         "ESP32-BLE"

// ============================================================================
// SYSTEM CONSTANTS
// ============================================================================

// ADC Configuration
const float VREF = 3.3;           // Reference voltage
const int ADC_MAX = 4095;         // 12-bit ADC resolution
const int ADC_RESOLUTION = 12;    // 12-bit ADC

// Sampling Configuration
const unsigned long SAMPLE_RATE_MS = 100;  // 10Hz = 100ms per sample
const unsigned long PRINT_INTERVAL_MS = 1000;  // Print debug info every 1s

// Calibration values for flex sensors
const float FLEX1_FLAT = 28884.0;  // Resistance when flat (Ohms)
const float FLEX2_FLAT = 30064.0;
const float FLEX3_FLAT = 27535.0;
const float FLEX4_FLAT = 28306.0;
const float FLEX5_FLAT = 26991.0;

const float R_FIXED = 26712.0;     // Fixed resistor in voltage divider (Ohms)

// Haptic feedback configuration
const int HAPTIC_EFFECT = 15;      // Effect number from DRV2605 library
const unsigned long HAPTIC_DEBOUNCE_MS = 500;  // Debounce for haptic trigger

// IMU Configuration
const int IMU_REPORT_INTERVAL_US = 10000;  // 100Hz internal IMU rate

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

Adafruit_BNO08x bno08x(BNO08X_RESET);
Adafruit_DRV2605 drv;

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = nullptr;


// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

bool deviceConnected = false;
bool oldDeviceConnected = false;

// Sensor readings
float flex1_volt = 0.0;
float flex2_volt = 0.0;
float flex3_volt = 0.0;
float flex4_volt = 0.0;
float flex5_volt = 0.0;

// IMU data
float qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;  // Quaternion
float ax = 0.0, ay = 0.0, az = 0.0;             // Accelerometer
float gx = 0.0, gy = 0.0, gz = 0.0;             // Gyroscope

// Joystick Data 
byte joyX;
byte joyY;
bool zPressed;
bool cPressed;

// Touch Sensor Data
const int touchPins[5] = {4, 15, 13, 12, 14};
const int touchThreshold[5] = {30, 30, 30, 30, 30}; //Can edit when know better values 
int touchValues[5] = {0,0,0,0,0}; 

unsigned long touchDebounce = 20; //Touch debounce (milliseconds)
unsigned long lastTouchTime[5] = {0, 0, 0, 0, 0};
bool lastTouchState[5] = {false, false, false, false, false};
bool areTouch[5] = {false, false, false, false, false};

// Timing variables
unsigned long lastSampleTime = 0;
unsigned long lastPrintTime = 0;
unsigned long lastHapticTime = 0;

// Statistics
unsigned long sampleCount = 0;
unsigned long transmitCount = 0;

// Haptic trigger (using pinky flex sensor as example)
bool motorActive = false;
float hapticThreshold = 0.0;  // Will be calculated in setup

// ============================================================================
// BLE SERVER CALLBACKS
// ============================================================================

//Custom server callbacks to track connection state 
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true; //Set flag when client connects 
      Serial.println("Device Connected");

    };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false; //Reset flag when client disconnects 
      Serial.println("Device Disconnected");
    }
};

class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String rxValue = pCharacteristic->getValue();
      unsigned long triggerTime = millis();

      if (rxValue.length() > 0) {
        Serial.print("RECEIVED: ");
        for (int i=0; i<rxValue.length(); i++) {
          Serial.print(rxValue[i]);
        }
        Serial.println();

        if (rxValue == "1") {
          Serial.println("Motor triggered");
        
          //Sets up the effect sequence -> Slot 0 = Selected Effect, Slot 1 = 0 (signals end of sequence)
          drv.setWaveform(0, HAPTIC_EFFECT);
          drv.setWaveform(1,0);
          //Plays effect - Sends command over I2C
          drv.go(); 

           // Send notification back to Python with timestamp - check latency 
          if (deviceConnected) {
              String notifyMsg = "MOTOR:" + String(triggerTime);
              pCharacteristic->setValue(notifyMsg.c_str());
              pCharacteristic->notify();
              }
          
        } 

      }    
  }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Convert ADC reading to voltage
 */
float adcToVoltage(int adcValue) {
    return (static_cast<float>(adcValue) / ADC_MAX) * VREF;
}

/**
 * Calculate resistance from voltage divider
 */
float voltageToResistance(float vout) {
    // Vout = Vin * (R_flex / (R_flex + R_fixed))
    // R_flex = R_fixed * Vout / (Vin - Vout)
    if (vout >= VREF) return FLEX1_FLAT;  // Avoid division by zero
    return R_FIXED * vout / (VREF - vout);
}

/**
 * Map resistance to normalized flex value (0.0 = flat, 1.0 = fully bent)
 */
float resistanceToFlex(float resistance, float flatResistance) {
    // Flex sensors increase resistance when bent
    // Typical range: 25k (flat) to 125k (90° bend)
    const float BENT_RESISTANCE = flatResistance * 5.0;  // Estimate
    
    float flex = (resistance - flatResistance) / (BENT_RESISTANCE - flatResistance);
    return constrain(flex, 0.0, 1.0);
}

/**
 * Simple moving average filter for noise reduction
 */
class MovingAverageFilter {
private:
    static const int WINDOW_SIZE = 3;
    float buffer[WINDOW_SIZE];
    int index;
    int count;
    
public:
    MovingAverageFilter() : index(0), count(0) {
        for (int i = 0; i < WINDOW_SIZE; i++) {
            buffer[i] = 0.0;
        }
    }
    
    float update(float value) {
        buffer[index] = value;
        index = (index + 1) % WINDOW_SIZE;
        if (count < WINDOW_SIZE) count++;
        
        float sum = 0.0;
        for (int i = 0; i < count; i++) {
            sum += buffer[i];
        }
        return sum / count;
    }
    
    void reset() {
        index = 0;
        count = 0;
    }
};

// Filters for each flex sensor
MovingAverageFilter flex1Filter, flex2Filter, flex3Filter, flex4Filter, flex5Filter;

// ============================================================================
// BLE INITIALIZATION
// ============================================================================

void initializeBLE(){
    Serial.println("Initializing BLE...");
    
    // Initialize BLE with device name
    BLEDevice::init(DEVICE_NAME);
    
    // Create BLE Server
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    // Create BLE Service
    BLEService *pService = pServer->createService(SERVICE_UUID);
    
    // TX characteristic (ESP32 → PC)
    BLECharacteristic *txCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID_TX,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    txCharacteristic->addDescriptor(new BLE2902());

    // RX characteristic (PC → ESP32)
    BLECharacteristic *rxCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID_RX,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    rxCharacteristic->setCallbacks(new MyCharacteristicCallbacks());

    // Save for use in sendSensorData()
    pCharacteristic = txCharacteristic;
    
    // Start the service
    pService->start();
    
    // Start advertising
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);  // Functions for iPhone connections
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    
    Serial.println("BLE Ready! Waiting for connections...");
}

// ============================================================================
// HAPTIC DRIVER INITIALIZATION
// ============================================================================

void initializeHapticDriver() {
    Serial.println("Initializing Haptic Driver (DRV2605)...");
    
    if (!drv.begin()) {
        Serial.println("ERROR: Could not find DRV2605");
        return;
    }
    
    // Select haptic library (1 = ERM library)
    drv.selectLibrary(1);
    
    // Set mode to internal trigger (wait for go() command)
    drv.setMode(DRV2605_MODE_INTTRIG);
    
    Serial.println("Haptic Driver Ready!");
    
    // Test haptic feedback
    drv.setWaveform(0, HAPTIC_EFFECT);
    drv.setWaveform(1, 0);  // End sequence
    drv.go();
    delay(200);
}

// ============================================================================
// IMU INITIALIZATION
// ============================================================================

void initializeIMU() {
    Serial.println("Initializing IMU (BNO085)...");
    
    if (!bno08x.begin_I2C(BNO085_ADDRESS)) {
        Serial.println("ERROR: Failed to find BNO085 chip");
        while (1) {
            delay(10);
        }
    }
    
    Serial.println("BNO085 Found!");
    
    // Enable sensor reports
    if (!bno08x.enableReport(SH2_ROTATION_VECTOR, IMU_REPORT_INTERVAL_US)) {
        Serial.println("ERROR: Could not enable rotation vector");
    }
    
    if (!bno08x.enableReport(SH2_ACCELEROMETER, IMU_REPORT_INTERVAL_US)) {
        Serial.println("ERROR: Could not enable accelerometer");
    }
    
    if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_INTERVAL_US)) {
        Serial.println("ERROR: Could not enable gyroscope");
    }
    
    Serial.println("IMU Ready!");
}

// ============================================================================
// JOYSTICK INITIALIZATION
// ============================================================================
void initializeJoystick() {

  Wire.begin(21,22); //SDA, SCL

  Wire.beginTransmission(0x52);
  Wire.write(0xF0);
  Wire.write(0x55);
  Wire.endTransmission();
  delay(10);

  Wire.beginTransmission(0x52);
  Wire.write(0xFB);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(10);

}

// ============================================================================
// SENSOR READING FUNCTIONS
// ============================================================================

/**
 * Read all flex sensors with filtering
 */
void readFlexSensors() {
    // Read raw ADC values
    int adc1 = analogRead(FLEX1_PIN);
    int adc2 = analogRead(FLEX2_PIN);
    int adc3 = analogRead(FLEX3_PIN);
    int adc4 = analogRead(FLEX4_PIN);
    int adc5 = analogRead(FLEX5_PIN);
    
    // Convert to voltage
    float v1 = adcToVoltage(adc1);
    float v2 = adcToVoltage(adc2);
    float v3 = adcToVoltage(adc3);
    float v4 = adcToVoltage(adc4);
    float v5 = adcToVoltage(adc5);
    
    // Apply moving average filter
    flex1_volt = flex1Filter.update(v1);
    flex2_volt = flex2Filter.update(v2);
    flex3_volt = flex3Filter.update(v3);
    flex4_volt = flex4Filter.update(v4);
    flex5_volt = flex5Filter.update(v5);
}

/**
 * Read IMU data (quaternion, accelerometer, gyroscope)
 */
bool readIMU() {
    sh2_SensorValue_t sensorValue;
    
    if (!bno08x.getSensorEvent(&sensorValue)) {
        return false;
    }
    
    // Process different sensor reports
    switch (sensorValue.sensorId) {
        case SH2_ROTATION_VECTOR:
            // Quaternion (orientation)
            qw = sensorValue.un.rotationVector.real;
            qx = sensorValue.un.rotationVector.i;
            qy = sensorValue.un.rotationVector.j;
            qz = sensorValue.un.rotationVector.k;
            break;
            
        case SH2_ACCELEROMETER:
            // Linear acceleration (m/s²)
            ax = sensorValue.un.accelerometer.x;
            ay = sensorValue.un.accelerometer.y;
            az = sensorValue.un.accelerometer.z;
            break;
            
        case SH2_GYROSCOPE_CALIBRATED:
            // Angular velocity (rad/s)
            gx = sensorValue.un.gyroscope.x;
            gy = sensorValue.un.gyroscope.y;
            gz = sensorValue.un.gyroscope.z;
            break;
    }
    
    return true;
}

void readJoystick() {

  Wire.beginTransmission(0x52);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(3);

  Wire.requestFrom(0x52, 6);
  if (Wire.available() >= 6) {
    joyX = Wire.read();
    joyY = Wire.read();
    byte ax = Wire.read();
    byte ay = Wire.read();
    byte az = Wire.read();
    byte buttons = Wire.read();
    
    zPressed = !(buttons & 0x01);
    cPressed = !((buttons >> 1) & 0x01);

  }
}

void readTouchSensors(unsigned long now){

  //Read all contact sensors for each finger 
  for(int i=0; i<5; i++) {
    touchValues[i] = touchRead(touchPins[i]);

    bool currentTouch = (touchValues[i] < touchThreshold[i]);

    if (currentTouch != lastTouchState[i]) {
      lastTouchTime[i] = now;
      lastTouchState[i] = currentTouch;

    }
    if ((now - lastTouchTime[i]) > touchDebounce) {
      areTouch[i] = currentTouch;
    
    }
  }
}

// ============================================================================
// DATA TRANSMISSION
// ============================================================================

/**
 * Format and send sensor data via BLE
 * Format: flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
 */
void sendSensorData() {
    if (!deviceConnected) {
        return;
    }
    
  // Build comma-separated string (24 values now)
  char buffer[512];
  snprintf(buffer, sizeof(buffer), 
          "SENSOR:%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d",
          flex2_volt,  // Thumb
          flex5_volt,  // Index
          flex4_volt,  // Middle
          flex3_volt,  // Ring
          flex1_volt,  // Pinky
          qw, qx, qy, qz,
          ax, ay, az,
          gx, gy, gz,
          joyX, joyY, cPressed, zPressed, // Joystick Data
          touchValues[0], touchValues[1], touchValues[2], touchValues[3], touchValues[4]); 

    
  pCharacteristic->setValue(buffer);  //Sets value of characteristic 
  pCharacteristic->notify();          //Notify subscribed client 
    
    transmitCount++;
}

// ============================================================================
// HAPTIC FEEDBACK
// ============================================================================

/**
 * Check flex sensor and trigger haptic feedback
 * Using pinky sensor (flex5) as example trigger
 */

 //Don't need this anymore bc of BLE connection 
 /*
void updateHapticFeedback() {
    unsigned long now = millis();
    
    // Check if pinky is bent and enough time has passed
    if (!motorActive && 
        flex5_volt < hapticThreshold && 
        (now - lastHapticTime) > HAPTIC_DEBOUNCE_MS) {
        
        Serial.println("Haptic feedback triggered!");
        
        // Play haptic effect
        drv.setWaveform(0, HAPTIC_EFFECT);
        drv.setWaveform(1, 0);  // End sequence
        drv.go();
        
        motorActive = true;
        lastHapticTime = now;
        
        // Visual feedback
        digitalWrite(LED_PIN, HIGH);
        delay(50);
        digitalWrite(LED_PIN, LOW);
    } 
    // Reset when pinky is extended again
    else if (motorActive && flex5_volt > hapticThreshold) {
        motorActive = false;
    }
}
*/

// ============================================================================
// DEBUG OUTPUT
// ============================================================================

/**
 * Print sensor data to serial monitor for debugging
 */
void printDebugInfo() {
    Serial.println("\n========== NEURA GLOVE STATUS ==========");
    
    // Connection status
    Serial.print("BLE Connected: ");
    Serial.println(deviceConnected ? "YES" : "NO");
    
    // Sample statistics
    Serial.print("Total Samples: ");
    Serial.print(sampleCount);
    Serial.print(" | Transmitted: ");
    Serial.println(transmitCount);
    
    // Flex sensor voltages
    Serial.println("\nFlex Sensors (Voltage):");
    Serial.printf("  Thumb:  %.3f V\n", flex2_volt);
    Serial.printf("  Index:  %.3f V\n", flex5_volt);
    Serial.printf("  Middle: %.3f V\n", flex4_volt);
    Serial.printf("  Ring:   %.3f V\n", flex3_volt);
    Serial.printf("  Pinky:  %.3f V\n", flex1_volt);
    
    // IMU data
    Serial.println("\nIMU Quaternion (Orientation):");
    Serial.printf("  W: %.4f  X: %.4f  Y: %.4f  Z: %.4f\n", qw, qx, qy, qz);
    
    Serial.println("IMU Accelerometer (m/s²):");
    Serial.printf("  X: %.3f  Y: %.3f  Z: %.3f\n", ax, ay, az);
    
    Serial.println("IMU Gyroscope (rad/s):");
    Serial.printf("  X: %.3f  Y: %.3f  Z: %.3f\n", gx, gy, gz);
    
    // Joystick Data
    Serial.println("Joystick Data:");
    Serial.printf("X: %d  Y: %d  Button C: %d  Z: %d\n", joyX, joyY, cPressed, zPressed);
    
    Serial.println("========================================\n");
}

// ============================================================================
// CALIBRATION
// ============================================================================

/**
 * Calibration routine - run at startup
 * User should keep hand flat and relaxed
 */
void calibrateFlexSensors() {
    Serial.println("\n========================================");
    Serial.println("CALIBRATION ROUTINE");
    Serial.println("Keep your hand FLAT and RELAXED");
    Serial.println("========================================\n");
    
    delay(2000);  // Give user time to prepare
    
    Serial.println("Calibrating in:");
    for (int i = 3; i > 0; i--) {
        Serial.printf("%d...\n", i);
        delay(1000);
    }
    Serial.println("Calibrating NOW!");
    
    // Take 50 samples
    const int NUM_SAMPLES = 50;
    float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum1 += adcToVoltage(analogRead(FLEX1_PIN));
        sum2 += adcToVoltage(analogRead(FLEX2_PIN));
        sum3 += adcToVoltage(analogRead(FLEX3_PIN));
        sum4 += adcToVoltage(analogRead(FLEX4_PIN));
        sum5 += adcToVoltage(analogRead(FLEX5_PIN));
        delay(20);
    }
    
    // Calculate averages
    float cal1 = sum1 / NUM_SAMPLES;
    float cal2 = sum2 / NUM_SAMPLES;
    float cal3 = sum3 / NUM_SAMPLES;
    float cal4 = sum4 / NUM_SAMPLES;
    float cal5 = sum5 / NUM_SAMPLES;
    
    Serial.println("\nCalibration Complete!");
    Serial.println("Baseline voltages (flat hand):");
    Serial.printf("  Thumb:  %.3f V\n", cal1);
    Serial.printf("  Index:  %.3f V\n", cal2);
    Serial.printf("  Middle: %.3f V\n", cal3);
    Serial.printf("  Ring:   %.3f V\n", cal4);
    Serial.printf("  Pinky:  %.3f V\n", cal5);
    
    // Set haptic threshold (75% of flat voltage)
    hapticThreshold = cal5 * 0.75;
    Serial.printf("\nHaptic threshold set to: %.3f V\n", hapticThreshold);
    
    Serial.println("========================================\n");
    delay(1000);
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    delay(3000);  // Wait for serial monitor
    
    Serial.println("\n\n");
    Serial.println("========================================");
    Serial.println("   NEURA GLOVE - ESP32 FIRMWARE v1.0   ");
    Serial.println("========================================");
    Serial.println();
    
    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    // Set ADC resolution
    analogReadResolution(ADC_RESOLUTION);
    analogSetAttenuation(ADC_11db);  // Full 0-3.3V range
    
    // Initialize I2C
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);  // 400kHz I2C speed
    
    // Initialize subsystems
    initializeBLE();
    initializeHapticDriver();
    initializeIMU();
    initializeJoystick();
    
    // Run calibration
    calibrateFlexSensors();
    
    Serial.println("System Ready! Starting 10Hz sampling...\n");
    
    // Blink LED to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
    }
    
    lastSampleTime = millis();
    lastPrintTime = millis();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    unsigned long currentTime = millis();
    
    // ========== 10Hz SAMPLING ==========
    if (currentTime - lastSampleTime >= SAMPLE_RATE_MS) {
        lastSampleTime = currentTime;
        sampleCount++;
        
        // Read all sensors
        readFlexSensors();
        readIMU();
        readJoystick();
        readTouchSensors(currentTime);
        
        // Send data via BLE
        sendSensorData();
        
        // Update haptic feedback
        //updateHapticFeedback();
        
        // Visual heartbeat (blink LED briefly every sample)
        if (deviceConnected) {
            digitalWrite(LED_PIN, HIGH);
            delayMicroseconds(500);  // Very brief blink
            digitalWrite(LED_PIN, LOW);
        }
    }
    
    // ========== CONTINUOUS IMU READING ==========
    // Read IMU more frequently than 10Hz for smoother data
    readIMU();
    
    // ========== DEBUG OUTPUT (1Hz) ==========
    if (currentTime - lastPrintTime >= PRINT_INTERVAL_MS) {
        lastPrintTime = currentTime;
        //printDebugInfo();
  
    }
    
    // ========== BLE CONNECTION MANAGEMENT ==========
    // Handle connection state changes
    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = deviceConnected;
        Serial.println("Device connected - starting data transmission");
    }
    
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);  // Give Bluetooth stack time
        pServer->startAdvertising();  // Restart advertising
        Serial.println("Disconnected - restarting advertising");
        oldDeviceConnected = deviceConnected;
    }
    
    // Small delay to prevent watchdog timer reset
    delay(1);
}

// ============================================================================
// END OF FIRMWARE
// ============================================================================