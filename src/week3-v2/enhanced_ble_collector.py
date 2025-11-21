"""
NEURA GLOVE - Enhanced BLE Collector
Supports multiple dataset collection per pose with 500 frames each

Usage:
    # Collect dataset 1 for fist pose
    python enhanced_ble_collector.py --pose fist --dataset-num 1
    
    # Collect dataset 2 for fist pose
    python enhanced_ble_collector.py --pose fist --dataset-num 2
    
    # Collect with custom frame count
    python enhanced_ble_collector.py --pose open --dataset-num 1 --frames 600
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

DEFAULT_FRAMES_PER_POSE = 500
TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 100ms for 10Hz


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SensorFrame:
    """Single sensor reading"""
    frame_number: int
    timestamp: float
    flex_sensors: List[float]
    imu_orientation: List[float]
    imu_accel: List[float]
    imu_gyro: List[float]


# ============================================================================
# ENHANCED BLE COLLECTOR
# ============================================================================

class EnhancedBLECollector:
    """Collects multiple datasets per pose with configurable frame count"""
    
    def __init__(self, device_name: str = "ESP32-BLE"):
        self.device_name = device_name
        self.service_uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.char_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        
        self.client: Optional[BleakClient] = None
        self.frames: List[SensorFrame] = []
        self.latest_data: Optional[bytearray] = None
        self.collection_start_time: float = 0
        
    async def find_device(self) -> str:
        """Scan for ESP32"""
        print(f"🔍 Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.device_name in device.name:
                print(f"✓ Found: {device.name} at {device.address}")
                return device.address
        
        raise Exception(f"Device {self.device_name} not found")
    
    def notification_handler(self, sender, data):
        """Store latest data from BLE notifications"""
        self.latest_data = data
    
    def parse_sensor_data(self, data: bytearray, frame_num: int) -> Optional[SensorFrame]:
        """Parse sensor data from ESP32"""
        try:
            data_str = data.decode('utf-8').strip()
            values = [float(x) for x in data_str.split(',')]
            
            if len(values) != 15:
                return None
            
            return SensorFrame(
                frame_number=frame_num,
                timestamp=time.time() - self.collection_start_time,
                flex_sensors=values[0:5],
                imu_orientation=values[5:9],
                imu_accel=values[9:12],
                imu_gyro=values[12:15]
            )
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    async def connect(self):
        """Connect to ESP32"""
        address = await self.find_device()
        self.client = BleakClient(address)
        await self.client.connect()
        print(f"✓ Connected to glove\n")
        
        # Start notifications
        await self.client.start_notify(self.char_uuid, self.notification_handler)
    
    async def disconnect(self):
        """Disconnect from ESP32"""
        if self.client and self.client.is_connected:
            await self.client.stop_notify(self.char_uuid)
            await self.client.disconnect()
            print("\n✓ Disconnected from glove\n")
    
    async def collect_dataset(self, pose_name: str, dataset_num: int, 
                             num_frames: int = DEFAULT_FRAMES_PER_POSE):
        """
        Collect a single dataset for a pose
        
        Args:
            pose_name: Name of the pose (e.g., 'fist', 'open')
            dataset_num: Dataset number for this pose (1, 2, 3, etc.)
            num_frames: Number of frames to collect (default 500)
        """
        print("=" * 70)
        print(f"BLE SENSOR COLLECTION")
        print("=" * 70)
        print(f"  Pose: {pose_name.upper()}")
        print(f"  Dataset: #{dataset_num}")
        print(f"  Target Frames: {num_frames}")
        print(f"  Sample Rate: {TARGET_FPS}Hz")
        print(f"  Estimated Duration: {num_frames / TARGET_FPS:.1f}s")
        print("=" * 70)
        print()
        
        await self.connect()
        
        # Wait for initial data
        print("⏳ Waiting for initial sensor data...")
        await asyncio.sleep(0.5)
        
        if self.latest_data is None:
            print("✗ No data received from glove!")
            await self.disconnect()
            return []
        
        print(f"✓ Receiving data from glove")
        print()
        print(f"📌 Instructions:")
        print(f"   1. Position your hand in the '{pose_name}' pose")
        print(f"   2. Hold the pose STEADY throughout collection")
        print(f"   3. Vary the pose slightly (natural hand movements)")
        print(f"   4. Collection will start in 3 seconds...")
        print()
        
        await asyncio.sleep(3)
        
        print("🎬 COLLECTION STARTED")
        print("-" * 70)
        
        self.collection_start_time = time.time()
        self.frames = []
        
        # Active polling at 10Hz
        for i in range(num_frames):
            frame_start = time.time()
            
            # Parse latest sensor data
            if self.latest_data:
                sensor_frame = self.parse_sensor_data(self.latest_data, i)
                
                if sensor_frame:
                    self.frames.append(sensor_frame)
                    
                    # Progress indicator every 50 frames
                    if (i + 1) % 50 == 0 or i == 0:
                        progress = (i + 1) / num_frames * 100
                        elapsed = time.time() - self.collection_start_time
                        print(f"  Frame {i+1:3d}/{num_frames} ({progress:5.1f}%) | "
                              f"Time: {elapsed:6.2f}s | "
                              f"Flex[0]: {sensor_frame.flex_sensors[0]:.3f}V")
                else:
                    print(f"  Frame {i+1}: Parse error, retrying...")
            
            # Sleep to maintain 10Hz rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, FRAME_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)
        
        await self.disconnect()
        
        # Statistics
        duration = time.time() - self.collection_start_time
        actual_fps = len(self.frames) / duration if duration > 0 else 0
        
        print("-" * 70)
        print("🎉 COLLECTION COMPLETE")
        print("=" * 70)
        print(f"  Frames Collected: {len(self.frames)}/{num_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Target FPS: {TARGET_FPS} Hz")
        print(f"  Actual FPS: {actual_fps:.2f} Hz")
        print(f"  Success Rate: {len(self.frames)/num_frames*100:.1f}%")
        print("=" * 70)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[SensorFrame], pose_name: str, 
                    dataset_num: int, output_dir: str = "data"):
        """
        Save dataset to JSON file
        
        File structure: data/{pose_name}/sensor_data_{dataset_num}.json
        """
        pose_dir = Path(output_dir) / pose_name
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = pose_dir / f"sensor_data_{dataset_num}.json"
        
        dataset = {
            'metadata': {
                'pose_name': pose_name,
                'dataset_number': dataset_num,
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_fps': TARGET_FPS,
                'data_type': 'BLE_SENSOR',
                'sensor_config': {
                    'flex_sensors': 5,
                    'imu_dof': 9,
                    'features_per_frame': 15
                }
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 Saved to: {output_file}")
        print(f"   Frames: {len(frames)}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='Collect BLE sensor data for hand pose training'
    )
    parser.add_argument(
        '--pose', '-p',
        required=True,
        help='Pose name (e.g., fist, open, point, peace, thumbs_up)'
    )
    parser.add_argument(
        '--dataset-num', '-n',
        type=int,
        required=True,
        help='Dataset number for this pose (1, 2, 3, etc.)'
    )
    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=DEFAULT_FRAMES_PER_POSE,
        help=f'Number of frames to collect (default: {DEFAULT_FRAMES_PER_POSE})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--device',
        default='ESP32-BLE',
        help='BLE device name (default: ESP32-BLE)'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = EnhancedBLECollector(args.device)
    
    # Collect dataset
    frames = await collector.collect_dataset(
        pose_name=args.pose,
        dataset_num=args.dataset_num,
        num_frames=args.frames
    )
    
    # Save if successful
    if len(frames) >= args.frames * 0.9:  # Allow 10% tolerance
        collector.save_dataset(frames, args.pose, args.dataset_num, args.output_dir)
        print(f"\n✅ SUCCESS! Dataset {args.dataset_num} for '{args.pose}' collected")
        print(f"\n📋 Next steps:")
        print(f"   1. Collect matching camera data:")
        print(f"      python enhanced_camera_collector.py --pose {args.pose} --dataset-num {args.dataset_num}")
        print(f"   2. Or collect more datasets for this pose:")
        print(f"      python enhanced_ble_collector.py --pose {args.pose} --dataset-num {args.dataset_num + 1}")
    else:
        print(f"\n✗ FAILED: Only collected {len(frames)}/{args.frames} frames")
        print("   Try again with better BLE connection")


if __name__ == "__main__":
    asyncio.run(main())