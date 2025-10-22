"""
NEURA GLOVE - Dataset Merger
Merges BLE sensor data and MediaPipe camera data frame-by-frame

This creates training dataset by aligning frames based on frame numbers,
not timestamps. Assumes both datasets have same number of frames collected
in the same session.

Usage:
    python dataset_merger.py --sensor sensor_data.json --camera camera_data.json --output training_dataset.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingSample:
    """Single training sample with aligned sensor and camera data"""
    frame_number: int
    
    # INPUT: Sensor data (15 features)
    flex_sensors: List[float]      # 5 values
    imu_orientation: List[float]   # 4 values (quaternion)
    imu_accel: List[float]         # 3 values
    imu_gyro: List[float]          # 3 values
    
    # OUTPUT: Ground truth from MediaPipe (147 values)
    joints: List[Dict]  # 21 joints × (3 pos + 4 rot) = 147 values


# ============================================================================
# DATASET MERGER
# ============================================================================

class DatasetMerger:
    """Merges sensor and camera datasets frame-by-frame"""
    
    def __init__(self):
        self.sensor_data = None
        self.camera_data = None
        self.training_samples: List[TrainingSample] = []
    
    def load_sensor_data(self, filepath: str):
        """Load BLE sensor dataset"""
        print(f"Loading sensor data from: {filepath}")
        with open(filepath, 'r') as f:
            self.sensor_data = json.load(f)
        
        print(f"  ✓ Loaded {len(self.sensor_data['frames'])} sensor frames")
        print(f"    Data type: {self.sensor_data['metadata']['data_type']}")
        print(f"    Collection date: {self.sensor_data['metadata']['collection_date']}")
    
    def load_camera_data(self, filepath: str):
        """Load MediaPipe camera dataset"""
        print(f"Loading camera data from: {filepath}")
        with open(filepath, 'r') as f:
            self.camera_data = json.load(f)
        
        print(f"  ✓ Loaded {len(self.camera_data['frames'])} camera frames")
        print(f"    Data type: {self.camera_data['metadata']['data_type']}")
        print(f"    Collection date: {self.camera_data['metadata']['collection_date']}")
    
    def align_frames(self, method: str = 'min') -> List[TrainingSample]:
        """
        Align sensor and camera frames by frame number
        
        Args:
            method: 'min' = use minimum frame count,
                   'interpolate' = interpolate to match counts
        
        Returns:
            List of aligned training samples
        """
        print("\n" + "="*60)
        print("ALIGNING FRAMES")
        print("="*60)
        
        sensor_frames = self.sensor_data['frames']
        camera_frames = self.camera_data['frames']
        
        num_sensor = len(sensor_frames)
        num_camera = len(camera_frames)
        
        print(f"Sensor frames: {num_sensor}")
        print(f"Camera frames: {num_camera}")
        
        if method == 'min':
            # Use minimum number of frames - truncate longer dataset
            num_frames = min(num_sensor, num_camera)
            print(f"Using method: 'min' - will use {num_frames} frames")
            
            if num_sensor > num_frames:
                print(f"  ⚠ Truncating {num_sensor - num_frames} sensor frames")
                sensor_frames = sensor_frames[:num_frames]
            
            if num_camera > num_frames:
                print(f"  ⚠ Truncating {num_camera - num_frames} camera frames")
                camera_frames = camera_frames[:num_frames]
            
            # Direct frame-by-frame alignment
            for i in range(num_frames):
                sample = TrainingSample(
                    frame_number=i,
                    flex_sensors=sensor_frames[i]['flex_sensors'],
                    imu_orientation=sensor_frames[i]['imu_orientation'],
                    imu_accel=sensor_frames[i]['imu_accel'],
                    imu_gyro=sensor_frames[i]['imu_gyro'],
                    joints=camera_frames[i]['joints']
                )
                self.training_samples.append(sample)
        
        elif method == 'interpolate':
            # Interpolate to match frame counts (advanced)
            print(f"Using method: 'interpolate'")
            
            if num_sensor == num_camera:
                print("  Frame counts match - no interpolation needed")
                # Direct alignment
                for i in range(num_sensor):
                    sample = TrainingSample(
                        frame_number=i,
                        flex_sensors=sensor_frames[i]['flex_sensors'],
                        imu_orientation=sensor_frames[i]['imu_orientation'],
                        imu_accel=sensor_frames[i]['imu_accel'],
                        imu_gyro=sensor_frames[i]['imu_gyro'],
                        joints=camera_frames[i]['joints']
                    )
                    self.training_samples.append(sample)
            
            elif num_sensor > num_camera:
                print(f"  More sensor frames - interpolating camera data")
                # Interpolate camera data to match sensor frames
                for i in range(num_sensor):
                    # Find corresponding camera frame index
                    camera_idx = int(i * num_camera / num_sensor)
                    camera_idx = min(camera_idx, num_camera - 1)
                    
                    sample = TrainingSample(
                        frame_number=i,
                        flex_sensors=sensor_frames[i]['flex_sensors'],
                        imu_orientation=sensor_frames[i]['imu_orientation'],
                        imu_accel=sensor_frames[i]['imu_accel'],
                        imu_gyro=sensor_frames[i]['imu_gyro'],
                        joints=camera_frames[camera_idx]['joints']
                    )
                    self.training_samples.append(sample)
            
            else:  # num_camera > num_sensor
                print(f"  More camera frames - interpolating sensor data")
                # Interpolate sensor data to match camera frames
                for i in range(num_camera):
                    # Find corresponding sensor frame index
                    sensor_idx = int(i * num_sensor / num_camera)
                    sensor_idx = min(sensor_idx, num_sensor - 1)
                    
                    sample = TrainingSample(
                        frame_number=i,
                        flex_sensors=sensor_frames[sensor_idx]['flex_sensors'],
                        imu_orientation=sensor_frames[sensor_idx]['imu_orientation'],
                        imu_accel=sensor_frames[sensor_idx]['imu_accel'],
                        imu_gyro=sensor_frames[sensor_idx]['imu_gyro'],
                        joints=camera_frames[i]['joints']
                    )
                    self.training_samples.append(sample)
        
        print(f"\n✓ Created {len(self.training_samples)} aligned training samples")
        print("="*60)
        
        return self.training_samples
    
    def validate_dataset(self):
        """Validate the merged dataset"""
        print("\n" + "="*60)
        print("DATASET VALIDATION")
        print("="*60)
        
        if not self.training_samples:
            print("✗ No training samples to validate!")
            return False
        
        # Check sample integrity
        sample = self.training_samples[0]
        
        print(f"Total samples: {len(self.training_samples)}")
        print(f"\nSample structure:")
        print(f"  Frame number: {sample.frame_number}")
        print(f"  Flex sensors: {len(sample.flex_sensors)} values")
        print(f"  IMU orientation: {len(sample.imu_orientation)} values")
        print(f"  IMU accel: {len(sample.imu_accel)} values")
        print(f"  IMU gyro: {len(sample.imu_gyro)} values")
        print(f"  Total input features: {len(sample.flex_sensors) + len(sample.imu_orientation) + len(sample.imu_accel) + len(sample.imu_gyro)}")
        print(f"  Joints: {len(sample.joints)} joints")
        print(f"  Total output values: {len(sample.joints) * 7} (21 joints × 7)")
        
        # Check for missing values
        has_errors = False
        for i, sample in enumerate(self.training_samples):
            if len(sample.flex_sensors) != 5:
                print(f"  ✗ Frame {i}: Invalid flex sensor count")
                has_errors = True
            if len(sample.imu_orientation) != 4:
                print(f"  ✗ Frame {i}: Invalid quaternion count")
                has_errors = True
            if len(sample.joints) != 21:
                print(f"  ✗ Frame {i}: Invalid joint count")
                has_errors = True
        
        if not has_errors:
            print(f"\n✓ All samples validated successfully!")
        
        # Show sample data
        print(f"\nExample sample (frame 0):")
        print(f"  Flex: {sample.flex_sensors}")
        print(f"  Quaternion: {sample.imu_orientation}")
        print(f"  First joint (wrist):")
        print(f"    Position: {sample.joints[0]['position']}")
        print(f"    Rotation: {sample.joints[0]['rotation']}")
        
        print("="*60)
        
        return not has_errors
    
    def save_training_dataset(self, output_path: str):
        """Save merged training dataset"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        dataset = {
            'metadata': {
                'total_samples': len(self.training_samples),
                'input_features': 15,  # 5 flex + 4 quat + 3 accel + 3 gyro
                'output_values': 147,  # 21 joints × (3 pos + 4 rot)
                'format': 'frame_aligned_training_data',
                'description': 'Frame-by-frame aligned sensor and MediaPipe data for LSTM training',
                'source_sensor': self.sensor_data['metadata']['collection_date'],
                'source_camera': self.camera_data['metadata']['collection_date']
            },
            'samples': [asdict(sample) for sample in self.training_samples]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✓ Training dataset saved to: {output_file}")
        print(f"  Total samples: {len(self.training_samples)}")
        print(f"  Input features: 15 per sample")
        print(f"  Output values: 147 per sample")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    def analyze_dataset(self):
        """Analyze dataset statistics"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        if not self.training_samples:
            print("No samples to analyze!")
            return
        
        # Extract all flex sensor values for statistics
        all_flex = []
        for sample in self.training_samples:
            all_flex.extend(sample.flex_sensors)
        
        all_flex = np.array(all_flex)
        
        print("Flex Sensor Statistics:")
        print(f"  Mean: {np.mean(all_flex):.3f}V")
        print(f"  Std Dev: {np.std(all_flex):.3f}V")
        print(f"  Min: {np.min(all_flex):.3f}V")
        print(f"  Max: {np.max(all_flex):.3f}V")
        print(f"  Range: {np.max(all_flex) - np.min(all_flex):.3f}V")
        
        # Check quaternion magnitudes
        all_quat_mags = []
        for sample in self.training_samples:
            q = np.array(sample.imu_orientation)
            mag = np.sqrt(np.sum(q**2))
            all_quat_mags.append(mag)
        
        print(f"\nQuaternion Magnitudes:")
        print(f"  Mean: {np.mean(all_quat_mags):.4f} (should be ~1.0)")
        print(f"  Std Dev: {np.std(all_quat_mags):.4f}")
        
        # Check for hand movement (variance in joint positions)
        wrist_positions = []
        for sample in self.training_samples:
            wrist_positions.append(sample.joints[0]['position'])
        
        wrist_positions = np.array(wrist_positions)
        
        print(f"\nHand Movement (Wrist Position):")
        print(f"  X variance: {np.var(wrist_positions[:, 0]):.6f}")
        print(f"  Y variance: {np.var(wrist_positions[:, 1]):.6f}")
        print(f"  Z variance: {np.var(wrist_positions[:, 2]):.6f}")
        
        if np.var(wrist_positions) < 0.0001:
            print(f"  ⚠ Warning: Very low variance - hand might be stationary!")
        else:
            print(f"  ✓ Good hand movement detected")
        
        print("="*60)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge BLE sensor and MediaPipe camera data frame-by-frame'
    )
    parser.add_argument(
        '--sensor', '-s',
        type=str,
        required=True,
        help='Input sensor data JSON file'
    )
    parser.add_argument(
        '--camera', '-c',
        type=str,
        required=True,
        help='Input camera data JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='training_dataset.json',
        help='Output training dataset JSON file (default: training_dataset.json)'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['min', 'interpolate'],
        default='min',
        help='Alignment method: min (truncate) or interpolate (default: min)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Perform detailed dataset analysis'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("NEURA GLOVE - DATASET MERGER")
    print("="*60)
    print()
    
    # Create merger
    merger = DatasetMerger()
    
    # Load datasets
    merger.load_sensor_data(args.sensor)
    merger.load_camera_data(args.camera)
    
    # Align frames
    merger.align_frames(method=args.method)
    
    # Validate
    if not merger.validate_dataset():
        print("\n✗ Dataset validation failed!")
        return
    
    # Analyze if requested
    if args.analyze:
        merger.analyze_dataset()
    
    # Save training dataset
    merger.save_training_dataset(args.output)
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Training dataset ready: {args.output}")
    print(f"Next step: Train model with train_model.py")
    print("="*60)
    print()


if __name__ == "__main__":
    main()