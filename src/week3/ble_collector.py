"""
NEURA GLOVE - BLE Sensor Data Collector
Collects flex sensor + IMU data at 10Hz and saves to JSON

Usage:
    python ble_collector.py --duration 60 --output sensor_data.json
"""

import asyncio
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional

from bleak import BleakClient, BleakScanner


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BLEConfig:
    """BLE collection configuration"""
    BLE_SERVICE_UUID: str = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    BLE_CHARACTERISTIC_UUID: str = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
    BLE_DEVICE_NAME: str = "ESP32-BLE"
    TARGET_FPS: int = 10
    FRAME_INTERVAL: float = 0.1  # 100ms


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SensorFrame:
    """Single sensor reading from ESP32"""
    frame_number: int
    flex_sensors: List[float]      # 5 values [0-3.3V]
    imu_orientation: List[float]   # 4 quaternion values (w,x,y,z)
    imu_accel: List[float]         # 3 acceleration values
    imu_gyro: List[float]          # 3 gyroscope values


# ============================================================================
# BLE COLLECTOR
# ============================================================================

class BLESensorCollector:
    """Collects sensor data from ESP32 via BLE at 10Hz"""
    
    def __init__(self, config: BLEConfig = None):
        self.config = config or BLEConfig()
        self.client: Optional[BleakClient] = None
        self.is_collecting = False
        self.frames: List[SensorFrame] = []
        self.frame_count = 0
        self.last_sample_time = 0
        self.raw_buffer = deque(maxlen=100)
        
    async def find_device(self) -> str:
        """Scan for ESP32 BLE device"""
        print(f"Scanning for {self.config.BLE_DEVICE_NAME}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.config.BLE_DEVICE_NAME in device.name:
                print(f"✓ Found device: {device.name} at {device.address}")
                return device.address
        
        raise Exception(f"Device {self.config.BLE_DEVICE_NAME} not found")
    
    def parse_sensor_data(self, data: bytearray) -> Optional[SensorFrame]:
        """Parse BLE notification data from ESP32"""
        try:
            # ESP32 sends: flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
            data_str = data.decode('utf-8').strip()
            values = [float(x) for x in data_str.split(',')]
            
            if len(values) != 15:
                print(f"⚠ Warning: Expected 15 values, got {len(values)}")
                return None
            
            return SensorFrame(
                frame_number=self.frame_count,
                flex_sensors=values[0:5],
                imu_orientation=values[5:9],   # qw, qx, qy, qz
                imu_accel=values[9:12],
                imu_gyro=values[12:15]
            )
        except Exception as e:
            print(f"✗ Error parsing sensor data: {e}")
            return None
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications with 10Hz rate limiting"""
        current_time = time.time()
        
        # Rate limit to exactly 10Hz
        if current_time - self.last_sample_time < self.config.FRAME_INTERVAL:
            return
        
        sensor_frame = self.parse_sensor_data(data)
        if sensor_frame:
            self.frames.append(sensor_frame)
            self.frame_count += 1
            self.last_sample_time = current_time
            
            # Print progress
            if self.frame_count % 10 == 0:
                print(f"  Frame {self.frame_count:4d} | "
                      f"Flex: [{sensor_frame.flex_sensors[0]:.2f}, {sensor_frame.flex_sensors[1]:.2f}, ...] | "
                      f"Quat: [{sensor_frame.imu_orientation[0]:.3f}, ...]")
    
    async def connect(self):
        """Connect to ESP32 via BLE"""
        address = await self.find_device()
        self.client = BleakClient(address)
        await self.client.connect()
        print(f"✓ Connected to {address}")
        
        # Subscribe to notifications
        await self.client.start_notify(
            self.config.BLE_CHARACTERISTIC_UUID,
            self.notification_handler
        )
        self.is_collecting = True
        print(f"✓ BLE notifications started at {self.config.TARGET_FPS}Hz\n")
    
    async def disconnect(self):
        """Disconnect from ESP32"""
        if self.client and self.client.is_connected:
            await self.client.stop_notify(self.config.BLE_CHARACTERISTIC_UUID)
            await self.client.disconnect()
            self.is_collecting = False
            print("\n✓ Disconnected from BLE device")
    
    async def collect(self, duration_seconds: int):
        """
        Collect sensor data for specified duration
        
        Args:
            duration_seconds: How long to collect data
            
        Returns:
            List of SensorFrame objects
        """
        print("="*60)
        print("BLE SENSOR DATA COLLECTION")
        print("="*60)
        print(f"Duration: {duration_seconds} seconds")
        print(f"Target rate: {self.config.TARGET_FPS}Hz")
        print(f"Expected frames: ~{duration_seconds * self.config.TARGET_FPS}")
        print("="*60)
        print()
        
        await self.connect()
        
        start_time = time.time()
        
        try:
            print("Collecting data... Press Ctrl+C to stop early\n")
            
            while True:
                elapsed = time.time() - start_time
                
                if elapsed >= duration_seconds:
                    break
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Collection stopped by user")
        
        finally:
            await self.disconnect()
        
        print()
        print("="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Total frames collected: {len(self.frames)}")
        print(f"Actual duration: {time.time() - start_time:.2f}s")
        print(f"Actual rate: {len(self.frames) / (time.time() - start_time):.2f}Hz")
        print("="*60)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[SensorFrame], output_path: str):
        """Save collected sensor frames to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        dataset = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_fps': self.config.TARGET_FPS,
                'data_type': 'BLE_SENSOR',
                'format': 'frame_by_frame',
                'description': 'ESP32 flex sensors + BNO085 IMU data at 10Hz'
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Dataset saved to: {output_file}")
        print(f"  Total frames: {len(frames)}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='Collect BLE sensor data from NEURA GLOVE at 10Hz'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Collection duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sensor_data.json',
        help='Output JSON file path (default: sensor_data.json)'
    )
    parser.add_argument(
        '--device', '-dev',
        type=str,
        default='ESP32-BLE',
        help='BLE device name (default: ESP32-BLE)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = BLEConfig()
    config.BLE_DEVICE_NAME = args.device
    
    # Create collector
    collector = BLESensorCollector(config)
    
    # Collect data
    frames = await collector.collect(args.duration)
    
    # Save dataset
    if frames:
        collector.save_dataset(frames, args.output)
        print(f"\n✓ Success! Collected {len(frames)} frames")
        print(f"  Next step: Run camera collector to get matching MediaPipe data")
    else:
        print("\n✗ No frames collected!")


if __name__ == "__main__":
    asyncio.run(main())