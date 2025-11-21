"""
NEURA GLOVE - BLE Collector (FIXED for reliable 10Hz)
Collects exactly 10 frames per pose at 10Hz using polling instead of passive notification

Usage:
    python fixed_ble_collector.py --pose fist --output data/fist/sensor_data.json
"""

import asyncio
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

from bleak import BleakClient, BleakScanner


# ============================================================================
# CONFIGURATION
# ============================================================================

FRAMES_PER_POSE = 300
TARGET_FPS = 300
FRAME_INTERVAL = 0.1  # 100ms


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SensorFrame:
    """Single sensor reading"""
    frame_number: int
    flex_sensors: List[float]
    imu_orientation: List[float]
    imu_accel: List[float]
    imu_gyro: List[float]


# ============================================================================
# FIXED BLE COLLECTOR
# ============================================================================

class FixedBLECollector:
    """Uses active polling for reliable 10Hz collection"""
    
    def __init__(self, device_name: str = "ESP32-BLE"):
        self.device_name = device_name
        self.service_uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.char_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        
        self.client: Optional[BleakClient] = None
        self.frames: List[SensorFrame] = []
        self.latest_data: Optional[bytearray] = None
        
    async def find_device(self) -> str:
        """Scan for ESP32"""
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.device_name in device.name:
                print(f"✓ Found: {device.name} at {device.address}")
                return device.address
        
        raise Exception(f"Device {self.device_name} not found")
    
    def notification_handler(self, sender, data):
        """Just store latest data, don't process yet"""
        self.latest_data = data
    
    def parse_sensor_data(self, data: bytearray) -> Optional[SensorFrame]:
        """Parse sensor data"""
        try:
            data_str = data.decode('utf-8').strip()
            values = [float(x) for x in data_str.split(',')]
            
            if len(values) != 15:
                return None
            
            return SensorFrame(
                frame_number=len(self.frames),
                flex_sensors=values[0:5],
                imu_orientation=values[5:9],
                imu_accel=values[9:12],
                imu_gyro=values[12:15]
            )
        except:
            return None
    
    async def connect(self):
        """Connect to ESP32"""
        address = await self.find_device()
        self.client = BleakClient(address)
        await self.client.connect()
        print(f"✓ Connected\n")
        
        # Start notifications (just to receive data)
        await self.client.start_notify(self.char_uuid, self.notification_handler)
    
    async def disconnect(self):
        """Disconnect"""
        if self.client and self.client.is_connected:
            await self.client.stop_notify(self.char_uuid)
            await self.client.disconnect()
            print("\n✓ Disconnected\n")
    
    async def collect_pose(self, pose_name: str):
        """
        Collect exactly 10 frames at 10Hz using ACTIVE POLLING
        """
        print("="*60)
        print(f"BLE COLLECTION - POSE: {pose_name.upper()}")
        print("="*60)
        print(f"Collecting {FRAMES_PER_POSE} frames at {TARGET_FPS}Hz")
        print("="*60)
        print()
        
        await self.connect()
        
        # Wait for initial data
        print("Waiting for initial data...")
        await asyncio.sleep(0.5)
        
        if self.latest_data is None:
            print("✗ No data received from glove!")
            await self.disconnect()
            return []
        
        print(f"✓ Data flowing from glove")
        print(f"\nHold '{pose_name}' pose steady...")
        print("Collecting frames...\n")
        
        start_time = time.time()
        
        # ACTIVE POLLING: Sample exactly every 100ms
        for i in range(FRAMES_PER_POSE):
            frame_start = time.time()
            
            # Get latest data
            if self.latest_data:
                sensor_frame = self.parse_sensor_data(self.latest_data)
                
                if sensor_frame:
                    self.frames.append(sensor_frame)
                    print(f"  Frame {i+1}/{FRAMES_PER_POSE} | "
                          f"Flex: [{sensor_frame.flex_sensors[0]:.2f}, {sensor_frame.flex_sensors[1]:.2f}, ...] | "
                          f"Time: {time.time() - start_time:.2f}s")
                else:
                    print(f"  Frame {i+1}/{FRAMES_PER_POSE} | Parse error, retrying...")
            
            # Sleep until next 100ms interval
            elapsed = time.time() - frame_start
            sleep_time = max(0, FRAME_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)
        
        await self.disconnect()
        
        duration = time.time() - start_time
        actual_fps = len(self.frames) / duration if duration > 0 else 0
        
        print()
        print("="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Frames collected: {len(self.frames)}/{FRAMES_PER_POSE}")
        print(f"Duration: {duration:.2f}s")
        print(f"Target FPS: {TARGET_FPS}")
        print(f"Actual FPS: {actual_fps:.2f}")
        print("="*60)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[SensorFrame], pose_name: str, output_path: str):
        """Save to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'pose_name': pose_name,
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_frames': FRAMES_PER_POSE,
                'target_fps': TARGET_FPS,
                'data_type': 'BLE_SENSOR',
                'format': 'pose_based'
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Frames: {len(frames)}")


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', '-p', required=True)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--device', default='ESP32-BLE')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"data/{args.pose}/sensor_data.json"
    
    collector = FixedBLECollector(args.device)
    frames = await collector.collect_pose(args.pose)
    
    if len(frames) == FRAMES_PER_POSE:
        collector.save_dataset(frames, args.pose, args.output)
        print(f"\n✓ SUCCESS! Collected {FRAMES_PER_POSE} frames for '{args.pose}'")
        print(f"  Next step: Run camera collector to get matching MediaPipe data")
    else:
        print(f"\n✗ Only got {len(frames)}/{FRAMES_PER_POSE} frames - try again")


if __name__ == "__main__":
    asyncio.run(main())