/*
 * NEURA GLOVE - ESP32 UDP Sensor Data Sender
 * 
 * Sends flex sensor + BNO085 IMU data to PC via WiFi UDP
 * Designed for real-time hand pose estimation ML pipeline
 * 
 * Hardware:
 * - ESP32-WROOM-32UE
 * - 5x Flex Sensors (GPIO 34, 35, 32, 33, 25)
 * - BNO085 9-DOF IMU (I2C)
 * 
 * Output Format (JSON over UDP):
 * {
 *   "timestamp": 1234567890,
 *   "flex": [v1, v2, v3, v4, v5],
 *   "quat": [x, y, z, w],
 *   "accel": [x, y, z],
 *   "gyro": [x, y, z]
 * }
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <ArduinoJson.h>

// ===== WiFi Configuration =====
const char* ssid = "Iyan's iPhone";
const char* password = "3XsdCzmkns#O1";

// UDP Configuration
const char* pc_host = "10.0.0.197";  // Your PC's IP address
const int pc_port = 5005;

WiFiUDP udp;

// ===== Sensor Configuration =====
// Flex sensors on ADC pins
const int FLEX_PINS[5] = {34, 35, 32, 33, 25};
const int FLEX_COUNT = 5;

// BNO085 IMU
#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// ===== Timing Configuration =====
const unsigned long SENSOR_INTERVAL_MS = 10;  // 100Hz sampling
unsigned long lastSensorRead = 0;

// ===== Data Buffers =====
float flexValues[5] = {0};
float quatValues[4] = {0, 0, 0, 1};  // x, y, z, w
float accelValues[3] = {0, 0, 0};
float gyroValues[3] = {0, 0, 0};

// Statistics
unsigned long packetsSent = 0;
unsigned long lastStatsTime = 0;

// ===== Function Declarations =====
void setupWiFi();
void setupIMU();
void readFlexSensors();
void readIMU();
void sendUDPPacket();
void printStats();

void scanNetworks() {
  Serial.println("\n=== Scanning for WiFi Networks ===");
  int n = WiFi.scanNetworks();
  
  if (n == 0) {
    Serial.println("No networks found");
  } else {
    Serial.print(n);
    Serial.println(" networks found:");
    for (int i = 0; i < n; ++i) {
      Serial.print(i + 1);
      Serial.print(": ");
      Serial.print(WiFi.SSID(i));
      Serial.print(" (");
      Serial.print(WiFi.RSSI(i));
      Serial.print(" dBm) ");
      Serial.print(WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "Open" : "Encrypted");
      
      // Highlight if it matches your SSID
      if (WiFi.SSID(i) == String(ssid)) {
        Serial.print(" <-- YOUR NETWORK");
      }
      Serial.println();
    }
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== NEURA GLOVE ESP32 ===");
  Serial.println("Initializing...\n");

  // SCAN FIRST
  scanNetworks();

  // Setup ADC for flex sensors
  analogReadResolution(12);  // 0-4095 range
  
  // Initialize I2C for IMU
  Wire.begin();
  
  // Setup WiFi
  setupWiFi();
  
  // Setup IMU
  setupIMU();
  
  Serial.println("\n✓ Setup complete!");
  Serial.println("Streaming sensor data...\n");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Read sensors at fixed interval (100Hz)
  if (currentTime - lastSensorRead >= SENSOR_INTERVAL_MS) {
    lastSensorRead = currentTime;
    
    // Read all sensors
    readFlexSensors();
    readIMU();
    
    // Send UDP packet
    sendUDPPacket();
    
    packetsSent++;
  }
  
  // Print statistics every 5 seconds
  if (currentTime - lastStatsTime >= 5000) {
    lastStatsTime = currentTime;
    printStats();
  }
}

void setupWiFi() {
  Serial.println("\n=== WiFi Debug Info ===");
  Serial.print("SSID: '");
  Serial.print(ssid);
  Serial.println("'");
  Serial.print("Password: '");
  Serial.print(password);
  Serial.println("'");
  Serial.print("Password length: ");
  Serial.println(strlen(password));
  
  // Check WiFi hardware
  Serial.print("WiFi Mode: ");
  WiFi.mode(WIFI_STA);
  Serial.println("STA");
  
  // Disconnect from any previous connection
  WiFi.disconnect(true);
  delay(1000);
  
  Serial.print("\nConnecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {  // Increased to 40
    delay(500);
    Serial.print(".");
    Serial.print(" [");
    Serial.print(WiFi.status());  // Print status code
    Serial.print("]");
    
    attempts++;
    
    // Try reconnecting every 10 attempts
    if (attempts % 10 == 0) {
      Serial.println("\nRetrying connection...");
      WiFi.disconnect();
      delay(500);
      WiFi.begin(ssid, password);
    }
  }
  
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected!");
    Serial.print("  ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("  Signal strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.print("  MAC Address: ");
    Serial.println(WiFi.macAddress());
  } else {
    Serial.println("\n✗ WiFi connection failed!");
    Serial.print("  Final status code: ");
    Serial.println(WiFi.status());
    printWiFiStatus(WiFi.status());
    Serial.println("\n  Troubleshooting:");
    Serial.println("  1. Check SSID is exact (case-sensitive)");
    Serial.println("  2. Verify password is correct");
    Serial.println("  3. Ensure network is 2.4GHz (NOT 5GHz)");
    Serial.println("  4. Move ESP32 closer to router");
    Serial.println("  5. Check if MAC filtering is enabled on router");
    while (1) delay(1000);
  }
}

void printWiFiStatus(wl_status_t status) {
  Serial.print("  Status meaning: ");
  switch(status) {
    case WL_IDLE_STATUS:
      Serial.println("Idle");
      break;
    case WL_NO_SSID_AVAIL:
      Serial.println("SSID not found - Check network name!");
      break;
    case WL_SCAN_COMPLETED:
      Serial.println("Scan completed");
      break;
    case WL_CONNECTED:
      Serial.println("Connected");
      break;
    case WL_CONNECT_FAILED:
      Serial.println("Connection failed - Check password!");
      break;
    case WL_CONNECTION_LOST:
      Serial.println("Connection lost");
      break;
    case WL_DISCONNECTED:
      Serial.println("Disconnected");
      break;
    default:
      Serial.println("Unknown");
      break;
  }
}

void setupIMU() {
  Serial.println("\nInitializing BNO085 IMU...");
  
  if (!bno08x.begin_I2C()) {
    Serial.println("✗ Failed to find BNO085 chip");
    Serial.println("  Check I2C connections (SDA=21, SCL=22)");
    while (1) delay(1000);
  }
  
  Serial.println("✓ BNO085 found!");
  
  // Enable required sensor reports
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {  // 100Hz
    Serial.println("✗ Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_ACCELEROMETER, 10000)) {  // 100Hz
    Serial.println("✗ Could not enable accelerometer");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {  // 100Hz
    Serial.println("✗ Could not enable gyroscope");
  }
  
  Serial.println("✓ IMU sensors enabled");
}

void readFlexSensors() {
  // Read all 5 flex sensors from ADC
  for (int i = 0; i < FLEX_COUNT; i++) {
    int rawValue = analogRead(FLEX_PINS[i]);
    flexValues[i] = rawValue;  // Keep as ADC value (0-4095)
  }
}

void readIMU() {
  // Read latest IMU data
  if (bno08x.wasReset()) {
    Serial.println("⚠ IMU was reset");
    setupIMU();  // Re-enable reports
  }
  
  if (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        // Quaternion (orientation)
        quatValues[0] = sensorValue.un.rotationVector.i;  // x
        quatValues[1] = sensorValue.un.rotationVector.j;  // y
        quatValues[2] = sensorValue.un.rotationVector.k;  // z
        quatValues[3] = sensorValue.un.rotationVector.real;  // w
        break;
        
      case SH2_ACCELEROMETER:
        // Linear acceleration (m/s²)
        accelValues[0] = sensorValue.un.accelerometer.x;
        accelValues[1] = sensorValue.un.accelerometer.y;
        accelValues[2] = sensorValue.un.accelerometer.z;
        break;
        
      case SH2_GYROSCOPE_CALIBRATED:
        // Angular velocity (rad/s)
        gyroValues[0] = sensorValue.un.gyroscope.x;
        gyroValues[1] = sensorValue.un.gyroscope.y;
        gyroValues[2] = sensorValue.un.gyroscope.z;
        break;
    }
  }
}

void sendUDPPacket() {
  // Create JSON packet using ArduinoJson
  StaticJsonDocument<512> doc;
  
  // Timestamp (milliseconds since boot)
  doc["timestamp"] = millis();
  
  // Flex sensors array
  JsonArray flex = doc.createNestedArray("flex");
  for (int i = 0; i < FLEX_COUNT; i++) {
    flex.add(flexValues[i]);
  }
  
  // IMU quaternion (orientation)
  JsonArray quat = doc.createNestedArray("quat");
  for (int i = 0; i < 4; i++) {
    quat.add(quatValues[i]);
  }
  
  // IMU accelerometer
  JsonArray accel = doc.createNestedArray("accel");
  for (int i = 0; i < 3; i++) {
    accel.add(accelValues[i]);
  }
  
  // IMU gyroscope
  JsonArray gyro = doc.createNestedArray("gyro");
  for (int i = 0; i < 3; i++) {
    gyro.add(gyroValues[i]);
  }
  
  // Serialize to string
  char buffer[512];
  size_t len = serializeJson(doc, buffer);
  
  // Send via UDP
  udp.beginPacket(pc_host, pc_port);
  udp.write((uint8_t*)buffer, len);
  udp.endPacket();
}

void printStats() {
  Serial.println("\n=== Status ===");
  Serial.print("Packets sent: ");
  Serial.println(packetsSent);
  Serial.print("WiFi RSSI: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  
  Serial.print("\nFlex sensors: ");
  for (int i = 0; i < FLEX_COUNT; i++) {
    Serial.print(flexValues[i], 0);
    if (i < FLEX_COUNT - 1) Serial.print(", ");
  }
  
  Serial.print("\nQuaternion: [");
  for (int i = 0; i < 4; i++) {
    Serial.print(quatValues[i], 3);
    if (i < 3) Serial.print(", ");
  }
  Serial.println("]");
  
  Serial.print("Accel (m/s²): [");
  for (int i = 0; i < 3; i++) {
    Serial.print(accelValues[i], 2);
    if (i < 2) Serial.print(", ");
  }
  Serial.println("]");
  
  Serial.println("==============\n");
}