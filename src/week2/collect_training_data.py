"""
NEURA Glove - Sensor Data Collector (BLE ONLY)
Run this FIRST - collects sensor data and saves with timestamps
NO camera, NO cv2.imshow() - pure BLE + asyncio
"""

import asyncio
import time
import numpy as np
from bleak import BleakClient, BleakScanner
import struct
import json
import os

# BLE Configuration
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
SENSOR_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class SensorCollector:
    """Collects ONLY sensor data via BLE"""
    
    def __init__(self, save_dir="sensor_data"):
        self.ble_client = None
        self.save_dir = save_dir
        self.sample_count = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def _on_sensor_data(self, sender, data: bytearray):
        """Handle incoming sensor data"""
        try:
            # Parse packet
            timestamp_us, = struct.unpack('<Q', data[0:8])
            flex = struct.unpack('<5H', data[8:18])
            accel = struct.unpack('<3f', data[18:30])
            gyro = struct.unpack('<3f', data[30:42])
            quat = struct.unpack('<4f', data[42:58])
            
            # Save immediately
            sample = {
                'timestamp': timestamp_us / 1e6,
                'flex': list(flex),
                'accel': list(accel),
                'gyro': list(gyro),
                'quat': list(quat)
            }
            
            filename = f"{self.save_dir}/sensor_{int(timestamp_us)}.json"
            with open(filename, 'w') as f:
                json.dump(sample, f)
            
            self.sample_count += 1
            
            # Print progress every 10 samples
            if self.sample_count % 10 == 0:
                print(f"Sensor samples: {self.sample_count}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    async def collect(self, duration_seconds=300):
        """Main collection loop"""
        print("\n" + "="*60)
        print("SENSOR DATA COLLECTION (BLE ONLY)")
        print("="*60)
        
        # Scan and connect
        print("Scanning for NEURA_GLOVE...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        glove = None
        for device in devices:
            if device.name and "NEURA_GLOVE" in device.name:
                glove = device
                break
        
        if not glove:
            raise Exception("Glove not found!")
        
        print(f"Connecting to {glove.address}...")
        self.ble_client = BleakClient(glove.address)
        await self.ble_client.connect()
        
        # Subscribe to notifications
        await self.ble_client.start_notify(SENSOR_CHAR_UUID, self._on_sensor_data)
        
        print(f"✓ Connected! Collecting for {duration_seconds} seconds...")
        print(f"Saving to: {self.save_dir}/\n")
        
        # Collect for specified duration
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                await asyncio.sleep(0.1)
                
                # Progress update every 30 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 30 == 0 and elapsed > 0:
                    remaining = duration_seconds - elapsed
                    print(f"Time: {elapsed:.0f}s / {duration_seconds}s | Samples: {self.sample_count} | Remaining: {remaining:.0f}s")
        
        finally:
            if self.ble_client and self.ble_client.is_connected:
                await self.ble_client.disconnect()
        
        print(f"\n✓ Collection complete!")
        print(f"  Saved {self.sample_count} sensor samples to {self.save_dir}/")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--output", type=str, default="sensor_data")
    
    args = parser.parse_args()
    
    collector = SensorCollector(save_dir=args.output)
    await collector.collect(duration_seconds=args.duration)


if __name__ == "__main__":
    asyncio.run(main())