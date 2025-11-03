"""
BLE Flex Sensor Data Collector
Collects flex sensor + IMU data from ESP32 via BLE
Aligned with pose script for synchronization with MediaPipe
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
            'pinky': {'min': FLEX_MIN_VOLTAGE, 'max': FLEX_MAX_VOLTAGE}
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
        Format: flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
        """
        try:
            values = [float(x) for x in data_string.split(',')]
            if len(values) != 15:
                return None
                
            # Extract and convert flex sensors (voltages) to angles
            flex_voltages = {
                'thumb': values[0],   # flex1 → Thumb
                'index': values[1],   # flex2 → Index
                'middle': values[2],  # flex3 → Middle
                'ring': values[3],    # flex4 → Ring
                'pinky': values[4]    # flex5 → Pinky
            }
            
            flex_angles = {
                name: self.voltage_to_angle(volt, name)
                for name, volt in flex_voltages.items()
            }
            
            # Extract IMU data
            quat = {'w': values[5], 'x': values[6], 'y': values[7], 'z': values[8]}
            accel = {'x': values[9], 'y': values[10], 'z': values[11]}
            gyro = {'x': values[12], 'y': values[13], 'z': values[14]}
            
            return {
                'flex_voltages': flex_voltages,
                'flex_angles': flex_angles,
                'imu': {
                    'quaternion': quat,
                    'accelerometer': accel,
                    'gyroscope': gyro
                }
            }
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def notification_handler(self, sender, data):
        """Handle incoming BLE notifications"""
        if not self.is_collecting:
            return
            
        data_string = data.decode('utf-8')
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
            
            # Print progress
            if self.tick_index % 10 == 0:
                print(f"  Collected {self.tick_index} samples for {self.pose_name}...")
    
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
            
            print("\n🔴 COLLECTING - Hold the pose!")
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
        
        print(f"\n✓ Saved: {filename}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Duration: {self.samples[-1]['timestamp']:.2f}s")
        print(f"  Avg rate: {output_data['metadata']['sampling_rate_hz']:.1f} Hz")
        
        return filename


class PoseScriptRunner:
    """Run a sequence of poses with aligned collection"""
    
    def __init__(self, session_id, pose_duration=3):
        self.session_id = session_id
        self.pose_duration = pose_duration
        
        # Define pose sequence (same for both sensor and camera)
        self.poses = [
            'flat_hand',
            'fist',
            'pointing',
            'thumbs_up',
            'peace_sign',
            'ok_sign',
            'pinch'
        ]
    
    async def run_collection(self):
        """Run complete pose sequence"""
        print(f"\n{'#'*60}")
        print(f"SENSOR COLLECTION SESSION: {self.session_id}")
        print(f"{'#'*60}")
        print(f"\nPoses to collect: {len(self.poses)}")
        print(f"Duration per pose: {self.pose_duration}s")
        print(f"\nPose sequence:")
        for i, pose in enumerate(self.poses, 1):
            print(f"  {i}. {pose}")
        
        input("\nPress ENTER when ready to start...")
        
        results = []
        for i, pose_name in enumerate(self.poses, 1):
            print(f"\n\n{'='*60}")
            print(f"POSE {i}/{len(self.poses)}: {pose_name.upper()}")
            print(f"{'='*60}")
            
            # Create collector for this pose
            collector = FlexDataCollector(pose_name, self.session_id)
            
            # Collect data
            success = await collector.connect_and_collect(self.pose_duration)
            
            if success:
                filename = collector.save_data()
                results.append({
                    'pose': pose_name,
                    'samples': len(collector.samples),
                    'file': filename
                })
            
            # Wait between poses
            if i < len(self.poses):
                print(f"\n⏸  Rest for 5 seconds before next pose...")
                await asyncio.sleep(5)
        
        # Save session summary
        self._save_session_summary(results)
        
        print(f"\n\n{'#'*60}")
        print(f"SESSION COMPLETE: {self.session_id}")
        print(f"{'#'*60}")
        print(f"Total poses collected: {len(results)}")
        for r in results:
            print(f"  {r['pose']}: {r['samples']} samples")
    
    def _save_session_summary(self, results):
        """Save summary of collection session"""
        summary = {
            'session_id': self.session_id,
            'collection_time': datetime.now().isoformat(),
            'pose_duration': self.pose_duration,
            'total_poses': len(results),
            'results': results
        }
        
        filename = f"data/sensor_recordings/{self.session_id}/session_summary.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Session summary saved: {filename}")


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python ble_flex_collector.py <session_id> [pose_duration]")
        print("\nExample:")
        print("  python ble_flex_collector.py session_001 3")
        print("\nThis will collect all poses in sequence.")
        return
    
    session_id = sys.argv[1]
    pose_duration = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    runner = PoseScriptRunner(session_id, pose_duration)
    await runner.run_collection()


if __name__ == "__main__":
    asyncio.run(main())