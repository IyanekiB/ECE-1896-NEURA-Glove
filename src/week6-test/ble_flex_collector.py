"""
BLE Flex Sensor Data Collector - Single Pose Version
Collects flex sensor + IMU data from ESP32 via BLE for ONE pose with custom duration
"""

import asyncio
import json
import time
import numpy as np
from bleak import BleakClient, BleakScanner
from datetime import datetime
import os

# BLE Configuration (from ESP32 firmware)
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DEVICE_NAME = "ESP32-BLE"

# Flex sensor calibration constants (voltage range from ESP32)
FLEX_MIN_VOLTAGE = 0.55  # Fully bent (90 degrees)
FLEX_MAX_VOLTAGE = 1.65  # Flat (0 degrees)
PINKY_MIN_VOLTAGE = 0.10 # Fully bent (90 degrees)
PINKY_MAX_VOLTAGE = 0.70 # Flat (0 degrees)

class FlexDataCollector:
    def __init__(self, pose_name, session_id, output_dir="data/sensor_recordings"):
        self.pose_name = pose_name
        self.session_id = session_id
        self.output_dir = output_dir
        self.samples = []
        self.tick_index = 0
        self.is_collecting = False
        
        # Create output directory structure
        os.makedirs(f"{output_dir}/{session_id}/{pose_name}", exist_ok=True)
        
        # Calibration values (per-sensor min/max for normalization)
        self.calibration = {
            'thumb': {'min': FLEX_MIN_VOLTAGE, 'max': FLEX_MAX_VOLTAGE},
            'index': {'min': FLEX_MIN_VOLTAGE, 'max': FLEX_MAX_VOLTAGE},
            'middle': {'min': FLEX_MIN_VOLTAGE, 'max': FLEX_MAX_VOLTAGE},
            'ring': {'min': FLEX_MIN_VOLTAGE, 'max': FLEX_MAX_VOLTAGE},
            'pinky': {'min': PINKY_MIN_VOLTAGE, 'max': PINKY_MAX_VOLTAGE}
        }
        
        self.start_time = None
        self.client = None
        
    def voltage_to_angle(self, voltage, sensor_name):
        """Convert voltage reading to bend angle (0-90 degrees)"""
        cal = self.calibration[sensor_name]
        # Clamp voltage to calibration range
        voltage = np.clip(voltage, cal['min'], cal['max'])
        # Linear interpolation: higher voltage = less bend
        normalized = (voltage - cal['min']) / (cal['max'] - cal['min'])
        angle = 90.0 * (1.0 - normalized)  # Invert: high voltage = 0 deg, low = 90 deg
        return float(angle)
    
    def parse_sensor_data(self, data_string):
        """Parse BLE data string from ESP32
        Format: flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz
        """
        try:
            values = [float(x) for x in data_string.split(',')]
            if len(values) != 9:
                print(f"[PARSE ERROR] Expected 9 values, got {len(values)}: {data_string[:50]}")
                return None
                
            # Extract and convert flex sensors (voltages) to angles
            flex_voltages = {
                'thumb': values[4],   # flex1 → Thumb
                'index': values[0],   # flex2 → Index
                'middle': values[3],  # flex3 → Middle
                'ring': values[2],    # flex4 → Ring
                'pinky': values[1]    # flex5 → Pinky
            }
            
            flex_angles = {
                name: self.voltage_to_angle(volt, name)
                for name, volt in flex_voltages.items()
            }
            
            # Extract IMU data
            quat = {'w': values[5], 'x': values[6], 'y': values[7], 'z': values[8]}
            
            return {
                'flex_voltages': flex_voltages,
                'flex_angles': flex_angles,
                'imu': quat
                }
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def notification_handler(self, sender, data):
        """Handle incoming BLE notifications"""
        if not self.is_collecting:
            return

        try:
            data_string = data.decode('utf-8')
        except Exception as e:
            print(f"[DECODE ERROR] Failed to decode BLE data: {e}")
            return

        parsed = self.parse_sensor_data(data_string)

        if parsed:
            sample = {
                'tick_index': self.tick_index,
                'timestamp': time.time() - self.start_time,
                'pose_name': self.pose_name,
                'session_id': self.session_id,
                'data': parsed
            }
            self.samples.append(sample)
            self.tick_index += 1

            # Print progress every 10 samples
            if self.tick_index % 10 == 0:
                print(f"  Collected {self.tick_index} samples for {self.pose_name}...")
        else:
            # Log parse failures
            if self.tick_index == 0:
                print(f"[WARNING] First data packet failed to parse")
    
    async def connect_and_collect(self, duration_seconds):
        """Connect to ESP32 and collect data for specified duration"""
        print(f"\n{'='*60}")
        print(f"Collecting pose: {self.pose_name}")
        print(f"Session: {self.session_id}")
        print(f"Duration: {duration_seconds}s")
        print(f"{'='*60}")
        
        # Find device
        print("\nScanning for ESP32...")
        devices = await BleakScanner.discover(timeout=5.0)
        target_device = None
        
        for device in devices:
            if device.name == DEVICE_NAME:
                target_device = device
                break
        
        if not target_device:
            print(f"ERROR: Device '{DEVICE_NAME}' not found!")
            return False
        
        print(f"Found device: {target_device.name} ({target_device.address})")
        
        # Connect
        print("Connecting...")
        async with BleakClient(target_device.address) as client:
            self.client = client
            print(f"Connected: {client.is_connected}")
            
            # Start notifications
            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            print("Subscribed to notifications")
            
            # Countdown
            print("\nStarting collection in:")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                await asyncio.sleep(1)
            
            print(f"\nCOLLECTING - Hold the '{self.pose_name}' pose for {duration_seconds} seconds!")
            self.is_collecting = True
            self.start_time = time.time()
            
            # Collect for duration
            await asyncio.sleep(duration_seconds)
            
            # Stop
            self.is_collecting = False
            print(f"\n✓ Collection complete! Collected {len(self.samples)} samples")
            
            await client.stop_notify(CHARACTERISTIC_UUID)
            
        return True
    
    def save_data(self):
        """Save collected data to JSON"""
        if not self.samples:
            print("No data to save!")
            return None
        
        output_data = {
            'metadata': {
                'pose_name': self.pose_name,
                'session_id': self.session_id,
                'total_samples': len(self.samples),
                'duration_seconds': self.samples[-1]['timestamp'],
                'collection_time': datetime.now().isoformat(),
                'sampling_rate_hz': len(self.samples) / self.samples[-1]['timestamp']
            },
            'calibration': self.calibration,
            'samples': self.samples
        }
        
        filename = f"{self.output_dir}/{self.session_id}/{self.pose_name}/sensor_data.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved: {filename}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Duration: {self.samples[-1]['timestamp']:.2f}s")
        print(f"  Avg rate: {output_data['metadata']['sampling_rate_hz']:.1f} Hz")
        
        return filename


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 4:
        print("\nUsage:")
        print("  python ble_flex_collector_single.py <session_id> <pose_name> <duration_seconds>")
        print("\nExample:")
        print("  python ble_flex_collector_single.py session_001 fist 60")
        print("\nThis will collect 'fist' pose data for 60 seconds.")
        print("\nCommon pose names: flat_hand, fist, grab, pointing, peace_sign, ok_sign")
        return
    
    session_id = sys.argv[1]
    pose_name = sys.argv[2]
    duration_seconds = int(sys.argv[3])
    
    if duration_seconds < 1 or duration_seconds > 300:
        print("ERROR: Duration must be between 1 and 300 seconds")
        return
    
    # Create collector
    collector = FlexDataCollector(pose_name, session_id)
    
    # Collect data
    success = await collector.connect_and_collect(duration_seconds)
    
    if success:
        collector.save_data()
        print("\nData collection complete!")
    else:
        print("\nData collection failed!")


if __name__ == "__main__":
    asyncio.run(main())