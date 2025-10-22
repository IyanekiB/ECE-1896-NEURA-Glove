#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_BNO08x.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define DEVICE_NAME "NEURA_GLOVE_WIFI"
#define SAMPLING_RATE_HZ 100
#define SAMPLE_INTERVAL_MS (1000 / SAMPLING_RATE_HZ)

// Wi-Fi network credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// UDP target (the computer collecting data)
const char* HOST_IP = "192.168.1.100";   // <-- change to your PC's IP address
const uint16_t HOST_PORT = 5555;         // UDP port to send data to

// GPIO Pins
const int FLEX_PINS[5] = {32, 33, 34, 35, 36};
const int LED_PIN = 2;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

WiFiUDP udp;
Adafruit_BNO08x imu;
sh2_SensorValue_t sensorValue;

unsigned long lastSampleTime = 0;
unsigned long lastStatsTime = 0;
int packetsSentThisSecond = 0;

// 50-byte packet structure
struct __attribute__((packed)) SensorPacket {
  uint16_t flexSensors[5];   // 10 bytes
  float accel[3];            // 12 bytes
  float gyro[3];             // 12 bytes
  float quat[4];             // 16 bytes
};
SensorPacket packet;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  for (int i = 0; i < 5; i++) pinMode(FLEX_PINS[i], INPUT);

  // Connect Wi-Fi
  Serial.printf("Connecting to Wi-Fi: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(250);
    Serial.print(".");
  }
  digitalWrite(LED_PIN, HIGH);
  Serial.println("\n✓ Wi-Fi connected");
  Serial.print("Local IP: "); Serial.println(WiFi.localIP());

  // Initialize IMU
  Wire.begin(21, 22);
  if (!imu.begin_I2C(0x4A) && !imu.begin_I2C(0x4B)) {
    Serial.println("IMU connection FAILED");
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
    }
  }

  imu.enableReport(SH2_ACCELEROMETER, SAMPLE_INTERVAL_MS * 1000);
  imu.enableReport(SH2_GYROSCOPE_CALIBRATED, SAMPLE_INTERVAL_MS * 1000);
  imu.enableReport(SH2_ROTATION_VECTOR, SAMPLE_INTERVAL_MS * 1000);

  Serial.printf("UDP target: %s:%u\n", HOST_IP, HOST_PORT);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    // --- Read sensors ---
    for (int i = 0; i < 5; i++) packet.flexSensors[i] = analogRead(FLEX_PINS[i]);
    if (imu.getSensorEvent(&sensorValue)) {
      switch (sensorValue.sensorId) {
        case SH2_ACCELEROMETER:
          memcpy(packet.accel, &sensorValue.un.accelerometer, sizeof(packet.accel));
          break;
        case SH2_GYROSCOPE_CALIBRATED:
          memcpy(packet.gyro, &sensorValue.un.gyroscope, sizeof(packet.gyro));
          break;
        case SH2_ROTATION_VECTOR:
          packet.quat[0] = sensorValue.un.rotationVector.real;
          packet.quat[1] = sensorValue.un.rotationVector.i;
          packet.quat[2] = sensorValue.un.rotationVector.j;
          packet.quat[3] = sensorValue.un.rotationVector.k;
          break;
      }
    }

    // --- Send via UDP ---
    udp.beginPacket(HOST_IP, HOST_PORT);
    udp.write((uint8_t*)&packet, sizeof(SensorPacket));
    udp.endPacket();
    packetsSentThisSecond++;
  }

  if (currentTime - lastStatsTime >= 1000) {
    lastStatsTime = currentTime;
    Serial.printf("Packets/sec: %d\n", packetsSentThisSecond);
    packetsSentThisSecond = 0;
  }

  delay(1);
}
