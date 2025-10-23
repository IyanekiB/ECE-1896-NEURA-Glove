//Flex Sensors, IMU + Haptic Feedback System

//Assigned ADCs
//ADC1_CH6 - GPIO34 - pin 10 = FLEX1
//ADC1_CH7 - GPIO35 - pin 11 = FLEX2
//ADC1_CH4 - GPIO32 - pin 12 = FLEX3
//ADC1_CH5 - GPIO33 - pin 11 = FLEX4
//ADC2_CH8 - GPIO25 - pin 14 = FLEX5

//SCL - GPIO22 - pin 39
//SDA - GPIO21 - pin 42

//BMO085 ADDRESS = 0x4A
//DRV2605 ADDRESS = 0x5A

#include <Wire.h>
#include "Adafruit_DRV2605.h"
#include <Adafruit_BNO08x.h>

#include <BLEDevice.h> // Core BLE functionality 
#include <BLEServer.h> // Creates BLE Server
#include <BLEUtils.h> // Utilities (UUIDs)
#include <BLE2902.h> // Descriptor needed for notifications


//ADC constants 
const float VREF = 3.3;
const int ADC_MAX = 4095;

//Record ESP32s Universally Uniquie ID for BLE
#define SERVICE_UUID        "6E400001-B5A3-F393-E0A9-E50E24DCCA9E" 
#define CHARACTERISTIC_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" 

#define BNO08X_RESET -1
sh2_SensorValue_t sensorValue;

const int HAPTIC_EFFECT = 15;  //Haptic-Feedback Effect Selection 

const float FLEX5_RESIST_FLAT = 26991;
const float R5 = 26712;
const unsigned long debounce = 500;
const unsigned long printTime = 1000;

bool motorActive = false;
bool deviceConnected = false; 

unsigned long lastTrigger = 0;
unsigned long lastPrint = 0;
unsigned long now = 0;

float triggerThresh = 0.75 * VREF * (R5/(R5 + FLEX5_RESIST_FLAT));
float releaseThresh = 0.8 * VREF * (R5/(R5 + FLEX5_RESIST_FLAT));
float flex1_volt, flex2_volt, flex3_volt, flex4_volt, flex5_volt;

Adafruit_DRV2605 drv;
Adafruit_BNO08x bno08x(BNO08X_RESET);
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;

void initializeBLE(); //Sets up devices BLE connection 
void initializeHapticDriver(); //Sets up Haptic driver connection 
void initializeIMU(); //Sets up i2c connection with BNO085 9 DOF imu
void updateMotorState(); //Checks and handles motor control logic
void sendSensorData(); //Sends sensor data through BLE to connected device
void resumeBLE_Advertising(); //Starts advertising BLE connection if no device is connected


//Custom server callbacks to track connection state 
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true; //Set flag when client connects 
    };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false; //Reset flag when client disconnects 
    }
};


//Takes ADC's digital value and calculates its back to voltage 
float calc_voltage(int v){
  return(static_cast<float>(v) / ADC_MAX) * VREF;
}


void setup() {
  //Set up serial monitor 
  Serial.begin(115200);
  delay(3000);

  //Set-up BLE
  initializeBLE();

  //Set-up Haptic driver 
  initializeHapticDriver();

  //Set-up IMU 
  initializeIMU();

  //Set the resolution to 12 bits (0-4095)
  analogReadResolution(12);
}

void loop() {

  //Read voltages at flex sensor outputs 
  flex1_volt = calc_voltage(analogRead(34));
  flex2_volt = calc_voltage(analogRead(35));
  flex3_volt = calc_voltage(analogRead(32));
  flex4_volt = calc_voltage(analogRead(33));
  flex5_volt = calc_voltage(analogRead(25));

  //Update current time
  now = millis();

  //If a second passed -> print sensor readings to Serial Monitor 
  if (now - lastPrint >= printTime){
    Serial.printf("Voltages: %.2f\t%.2f\t%.2f\t%.2f\t%.2f\n\n", flex1_volt, flex2_volt, flex3_volt, flex4_volt, flex5_volt);
    Serial.printf("Accelerometer X: %.2f\t Y: %.2f\t Z: %.2f\t\n", sensorValue.un.accelerometer.x, sensorValue.un.accelerometer.y, sensorValue.un.accelerometer.z);
    lastPrint = now;
  }
  
  updateMotorState();
  
  if(deviceConnected){
    sendSensorData();
  }
  else{
    resumeBLE_Advertising();
  }

  
}//End of main loop


void initializeBLE(){
  //Initialize BLE with device name
  BLEDevice::init("ESP32-BLE"); 

  //Create BLE server and set callbacks for connection events 
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  //Create BLE service with given UUID
  BLEService *pService = pServer->createService(SERVICE_UUID);

  //Create characeristic inside the service, configured for notifications 
  pCharacteristic = pService->createCharacteristic(CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);

  //Add descriptor required for notifications 
  pCharacteristic->addDescriptor(new BLE2902());

  //Start the service and begin advertising so other devices can discover it 
  pService->start();
  pServer->getAdvertising()->start();

  Serial.println("BLE started, now you can connect");
}


void initializeHapticDriver(){

  //Check if driver is connected 
  if (!drv.begin()){
    Serial.println("Could not find DRV2605");
  }

  //Select which library of haptic effects to use (1 = basic library)
  drv.selectLibrary(1);
  //Set the driver to wait until program gives it a "go" command to start the motor
  drv.setMode(DRV2605_MODE_INTTRIG);
}

void initializeIMU(){
  // Try to initialize!
  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip");
  }

  Serial.println("BNO08x Found");
}

void updateMotorState(){
   //If motor is off and flex5 sensor's volt drops below threshold and enough time has passed since last ran --> activate motor 
  if (!motorActive && (flex5_volt < triggerThresh) && (now - lastTrigger > debounce)){
    Serial.println("\nMotor triggered");

    //Sets up the effect sequence -> Slot 0 = Selected Effect, Slot 1 = 0 (signals end of sequence)
    drv.setWaveform(0, HAPTIC_EFFECT);
    drv.setWaveform(1,0);
    //Plays effect - Sends command over I2C
    drv.go(); 

    //Set flag that motor is running and update time of last activated/triggered
    motorActive = true;
    lastTrigger = now;
  } 
  //Else motor is on and flex sensor 5 is above threshold again -> reset 
  else if (motorActive && (flex5_volt > releaseThresh)){\
    motorActive = false; 
  }
}


void sendSensorData(){

  //Sets up string of flex sensor values separated by commas
  String sensors = String(flex1_volt) + "," + String(flex2_volt) + "," + String(flex3_volt) + "," + String(flex4_volt) + "," + String(flex5_volt);
  pCharacteristic->setValue(sensors); //Sets value of characteristic 
  pCharacteristic->notify();          //Notify subscribed client 
  delay(300);  //~3Hz 

}


void resumeBLE_Advertising(){
  //Keep advertising bluetooth connection 
    delay(500);
    pServer->startAdvertising();
}
