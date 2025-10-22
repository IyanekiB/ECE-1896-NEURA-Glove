"""
NEURA Glove - Data Synchronizer
Run this THIRD (after collection) - matches timestamps and creates training data
"""

import json
import os
from glob import glob
import numpy as np

def synchronize_data(sensor_dir="sensor_data", camera_dir="camera_data", output_dir="training_data"):
    """Match sensor and camera data by timestamps"""
    
    print("\n" + "="*60)
    print("DATA SYNCHRONIZATION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all files
    print(f"Loading sensor data from {sensor_dir}/...")
    sensor_files = sorted(glob(f"{sensor_dir}/*.json"))
    sensor_data = []
    for f in sensor_files:
        with open(f, 'r') as file:
            sensor_data.append(json.load(file))

    # --- Pre-correct sensor timestamps (shift forward by 0.12s) ---
    time_offset = 0.15  # seconds
    for s in sensor_data:
        s['timestamp'] += time_offset
    print(f"Applied timestamp correction of +{time_offset}s to all sensor samples.")
    
    print(f"Loading camera data from {camera_dir}/...")
    camera_files = sorted(glob(f"{camera_dir}/*.json"))
    camera_data = []
    for f in camera_files:
        with open(f, 'r') as file:
            camera_data.append(json.load(file))
    
    print(f"\nLoaded:")
    print(f"  Sensor samples: {len(sensor_data)}")
    print(f"  Camera samples: {len(camera_data)}")
    
    if len(sensor_data) == 0 or len(camera_data) == 0:
        print("\n❌ No data to synchronize!")
        return
    
    # Match timestamps (±50ms window)
    print(f"\nMatching timestamps (±50ms window)...")
    
    synced_count = 0
    max_time_diff = 0.05  # 50ms
    
    for sensor in sensor_data:
        sensor_time = sensor['timestamp']
        
        # Find closest camera sample
        best_match = None
        min_diff = max_time_diff
        
        for camera in camera_data:
            time_diff = abs(sensor_time - camera['timestamp'])
            if time_diff < min_diff and camera['confidence'] > 0.7:
                min_diff = time_diff
                best_match = camera
        
        if best_match:
            # Create synchronized sample
            synced_sample = {
                'timestamp': sensor_time,
                'sensor': {
                    'flex': sensor['flex'],
                    'accel': sensor['accel'],
                    'gyro': sensor['gyro'],
                    'quat': sensor['quat']
                },
                'ground_truth': {
                    'landmarks': best_match['landmarks'],
                    'confidence': best_match['confidence']
                }
            }
            
            # Save
            filename = f"{output_dir}/sample_{int(sensor_time * 1000000)}.json"
            with open(filename, 'w') as f:
                json.dump(synced_sample, f)
            
            synced_count += 1
    
    sync_rate = (synced_count / len(sensor_data)) * 100
    
    print(f"\n✓ Synchronization complete!")
    print(f"  Matched pairs: {synced_count}")
    print(f"  Sync rate: {sync_rate:.1f}%")
    print(f"  Saved to: {output_dir}/")
    
    if synced_count < 300:
        print(f"\n⚠️  WARNING: Only {synced_count} samples!")
        print("  Recommend: 300+ samples for good training")
        print("  Consider collecting more data")
    else:
        print(f"\n✓ Good! {synced_count} samples is sufficient for training")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=str, default="sensor_data")
    parser.add_argument("--camera", type=str, default="camera_data")
    parser.add_argument("--output", type=str, default="training_data")
    
    args = parser.parse_args()
    
    synchronize_data(args.sensor, args.camera, args.output)


if __name__ == "__main__":
    main()