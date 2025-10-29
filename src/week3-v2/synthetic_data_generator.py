"""
SYNTHETIC DATA GENERATOR
Creates augmented camera and sensor data from existing datasets

PURPOSE:
- Generate more training data from existing camera/sensor recordings
- Add realistic noise and variations to sensor readings
- Create slight variations in MediaPipe joint positions
- Maintain temporal coherence within each synthetic dataset

USAGE:
    # Generate 5 synthetic datasets from existing fist pose data
    python synthetic_data_generator.py --pose fist --num-synthetic 5
    
    # Generate for all poses
    python synthetic_data_generator.py --all-poses --num-synthetic 3
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import copy


class SyntheticDataGenerator:
    """
    Generates synthetic sensor and camera data with realistic variations
    """
    
    def __init__(self, pose_name: str, data_dir: str = "data"):
        self.pose_name = pose_name
        self.data_dir = Path(data_dir)
        self.pose_dir = self.data_dir / pose_name
        
        # Augmentation parameters for sensor data
        self.sensor_noise_params = {
            'flex_sensors': {
                'noise_std': 0.20,       # Small noise on flex readings
                'drift_range': 0.50,     # Slow drift over time
                'scale_range': (0.95, 1.05)  # Scale variation
            },
            'imu_orientation': {
                'noise_std': 0.10,       # Quaternion noise (small)
                'drift_range': 0.20
            },
            'imu_accel': {
                'noise_std': 0.1,        # Accelerometer noise
                'bias_range': 0.10       # Bias variation
            },
            'imu_gyro': {
                'noise_std': 0.50,       # Gyroscope noise
                'bias_range': 0.30
            }
        }
        
        # Augmentation parameters for camera data
        self.camera_noise_params = {
            'position_noise_std': 0.020,  # Small position jitter
            'rotation_noise_std': 0.10    # Small rotation jitter
        }
    
    def load_existing_datasets(self) -> Tuple[List[dict], List[dict]]:
        """Load all existing sensor and camera datasets for this pose"""
        sensor_datasets = []
        camera_datasets = []
        
        # Find all sensor and camera files
        sensor_files = sorted(self.pose_dir.glob("sensor_data_*.json"))
        camera_files = sorted(self.pose_dir.glob("camera_data_*.json"))
        
        print(f"\nFound {len(sensor_files)} sensor datasets and {len(camera_files)} camera datasets")
        
        for sf, cf in zip(sensor_files, camera_files):
            try:
                with open(sf, 'r') as f:
                    sensor_data = json.load(f)
                    sensor_datasets.append(sensor_data)
                    
                    # Debug: show structure
                    if len(sensor_datasets) == 1:  # Only for first file
                        print(f"\nSensor data structure from {sf.name}:")
                        print(f"  Keys: {list(sensor_data.keys())}")
                        if 'metadata' in sensor_data:
                            print(f"  Metadata keys: {list(sensor_data['metadata'].keys())}")
                
                with open(cf, 'r') as f:
                    camera_data = json.load(f)
                    camera_datasets.append(camera_data)
                    
                    # Debug: show structure
                    if len(camera_datasets) == 1:  # Only for first file
                        print(f"\nCamera data structure from {cf.name}:")
                        print(f"  Keys: {list(camera_data.keys())}")
                        if 'metadata' in camera_data:
                            print(f"  Metadata keys: {list(camera_data['metadata'].keys())}")
            except Exception as e:
                print(f"\n⚠️  Error loading {sf.name} or {cf.name}: {e}")
                continue
        
        return sensor_datasets, camera_datasets
    
    def normalize_quaternion(self, quat: List[float]) -> List[float]:
        """Normalize quaternion to unit length"""
        q = np.array(quat)
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return [0.0, 0.0, 0.0, 1.0]
        return (q / norm).tolist()
    
    def add_sensor_noise(self, sensor_frame: dict, temporal_drift: dict) -> dict:
        """
        Add realistic noise to a sensor frame
        
        Args:
            sensor_frame: Original sensor reading
            temporal_drift: Accumulated drift values for this sequence
        
        Returns:
            Augmented sensor frame
        """
        augmented = copy.deepcopy(sensor_frame)
        
        # Flex sensors: add noise + drift + scale
        flex = np.array(augmented['flex_sensors'])
        noise = np.random.normal(0, self.sensor_noise_params['flex_sensors']['noise_std'], len(flex))
        drift = temporal_drift['flex']
        scale = np.random.uniform(*self.sensor_noise_params['flex_sensors']['scale_range'])
        flex = flex * scale + noise + drift
        flex = np.clip(flex, 0, 1023)  # Realistic ADC range
        augmented['flex_sensors'] = flex.tolist()
        
        # IMU orientation: add noise + drift, then normalize
        imu_ori = np.array(augmented['imu_orientation'])
        noise = np.random.normal(0, self.sensor_noise_params['imu_orientation']['noise_std'], 4)
        drift = temporal_drift['imu_ori']
        imu_ori = imu_ori + noise + drift
        augmented['imu_orientation'] = self.normalize_quaternion(imu_ori.tolist())
        
        # IMU acceleration: add noise + bias
        imu_accel = np.array(augmented['imu_accel'])
        noise = np.random.normal(0, self.sensor_noise_params['imu_accel']['noise_std'], 3)
        bias = temporal_drift['imu_accel']
        imu_accel = imu_accel + noise + bias
        augmented['imu_accel'] = imu_accel.tolist()
        
        # IMU gyroscope: add noise + bias
        imu_gyro = np.array(augmented['imu_gyro'])
        noise = np.random.normal(0, self.sensor_noise_params['imu_gyro']['noise_std'], 3)
        bias = temporal_drift['imu_gyro']
        imu_gyro = imu_gyro + noise + bias
        augmented['imu_gyro'] = imu_gyro.tolist()
        
        return augmented
    
    def add_camera_noise(self, camera_frame: dict) -> dict:
        """
        Add realistic noise to camera frame (MediaPipe joint data)
        
        Args:
            camera_frame: Original camera frame with joint positions/rotations
        
        Returns:
            Augmented camera frame
        """
        augmented = copy.deepcopy(camera_frame)
        
        # Check if frame has joints data
        if 'joints' not in augmented:
            # If no joints key, return frame as-is (might be different format)
            return augmented
        
        # Add noise to each joint
        for joint in augmented['joints']:
            # Position noise
            if 'position' in joint:
                pos = np.array(joint['position'])
                pos_noise = np.random.normal(0, self.camera_noise_params['position_noise_std'], 3)
                joint['position'] = (pos + pos_noise).tolist()
            
            # Rotation noise (quaternion)
            if 'rotation' in joint:
                rot = np.array(joint['rotation'])
                rot_noise = np.random.normal(0, self.camera_noise_params['rotation_noise_std'], 4)
                rot = rot + rot_noise
                joint['rotation'] = self.normalize_quaternion(rot.tolist())
        
        return augmented
    
    def generate_temporal_drift(self, num_frames: int) -> dict:
        """
        Generate slow temporal drift for an entire sequence
        Makes synthetic data more realistic by simulating sensor drift
        """
        # Generate random walk for drift
        drift = {
            'flex': np.cumsum(np.random.normal(
                0, self.sensor_noise_params['flex_sensors']['drift_range'], 
                (num_frames, 5)
            ), axis=0),
            'imu_ori': np.cumsum(np.random.normal(
                0, self.sensor_noise_params['imu_orientation']['drift_range'],
                (num_frames, 4)
            ), axis=0),
            'imu_accel': np.random.normal(
                0, self.sensor_noise_params['imu_accel']['bias_range'],
                (num_frames, 3)
            ),
            'imu_gyro': np.random.normal(
                0, self.sensor_noise_params['imu_gyro']['bias_range'],
                (num_frames, 3)
            )
        }
        
        return drift
    
    def generate_synthetic_dataset(self, source_sensor: dict, source_camera: dict,
                                   dataset_num: int) -> Tuple[dict, dict]:
        """
        Generate one synthetic dataset pair from source data
        
        Args:
            source_sensor: Source sensor dataset
            source_camera: Source camera dataset
            dataset_num: Dataset number for output
        
        Returns:
            (synthetic_sensor, synthetic_camera) datasets
        """
        print(f"  Generating synthetic dataset {dataset_num}...")
        
        # Check which key is used for frames (handles both 'frames' and 'sensor_data')
        if 'frames' in source_sensor:
            sensor_frames_key = 'frames'
        elif 'sensor_data' in source_sensor:
            sensor_frames_key = 'sensor_data'
        else:
            raise ValueError("Sensor data has neither 'frames' nor 'sensor_data' key")
        
        if 'frames' in source_camera:
            camera_frames_key = 'frames'
        elif 'camera_data' in source_camera:
            camera_frames_key = 'camera_data'
        else:
            raise ValueError("Camera data has neither 'frames' nor 'camera_data' key")
        
        num_frames = len(source_sensor[sensor_frames_key])
        
        # Generate temporal drift for this sequence
        temporal_drift_sequence = self.generate_temporal_drift(num_frames)
        
        # Create synthetic sensor dataset (use same key as source)
        # Handle metadata gracefully
        sensor_metadata = source_sensor.get('metadata', {})
        synthetic_sensor = {
            'metadata': {
                'pose_name': sensor_metadata.get('pose_name', self.pose_name),
                'dataset_number': dataset_num,
                'num_frames': num_frames,
                'synthetic': True,
                'source_dataset': sensor_metadata.get('dataset_number', 0),
                'collection_date': sensor_metadata.get('collection_date', 'unknown'),
                'data_type': sensor_metadata.get('data_type', 'BLE_SENSOR'),
                'target_fps': sensor_metadata.get('target_fps', 10)
            },
            sensor_frames_key: []  # Use same key as source
        }
        
        # Create synthetic camera dataset (use same key as source)
        camera_metadata = source_camera.get('metadata', {})
        synthetic_camera = {
            'metadata': {
                'pose_name': camera_metadata.get('pose_name', self.pose_name),
                'dataset_number': dataset_num,
                'num_frames': num_frames,
                'synthetic': True,
                'source_dataset': camera_metadata.get('dataset_number', 0),
                'collection_date': camera_metadata.get('collection_date', 'unknown'),
                'data_type': camera_metadata.get('data_type', 'MEDIAPIPE_CAMERA'),
                'target_fps': camera_metadata.get('target_fps', 10)
            },
            camera_frames_key: []  # Use same key as source
        }
        
        # Generate augmented frames
        for i in range(num_frames):
            # Get temporal drift for this frame
            drift_values = {
                'flex': temporal_drift_sequence['flex'][i],
                'imu_ori': temporal_drift_sequence['imu_ori'][i],
                'imu_accel': temporal_drift_sequence['imu_accel'][i],
                'imu_gyro': temporal_drift_sequence['imu_gyro'][i]
            }
            
            # Augment sensor frame
            sensor_frame = source_sensor[sensor_frames_key][i]
            synthetic_sensor_frame = self.add_sensor_noise(sensor_frame, drift_values)
            synthetic_sensor[sensor_frames_key].append(synthetic_sensor_frame)
            
            # Augment camera frame
            camera_frame = source_camera[camera_frames_key][i]
            synthetic_camera_frame = self.add_camera_noise(camera_frame)
            synthetic_camera[camera_frames_key].append(synthetic_camera_frame)
        
        return synthetic_sensor, synthetic_camera
    
    def generate_multiple_synthetic_datasets(self, num_synthetic: int = 3):
        """
        Generate multiple synthetic datasets for this pose
        
        Args:
            num_synthetic: Number of synthetic datasets to generate per source
        """
        print(f"\n{'='*70}")
        print(f"GENERATING SYNTHETIC DATA FOR POSE: {self.pose_name}")
        print(f"{'='*70}")
        
        # Load existing datasets
        sensor_datasets, camera_datasets = self.load_existing_datasets()
        
        if len(sensor_datasets) == 0:
            print(f"  ❌ No existing datasets found for pose '{self.pose_name}'")
            return
        
        # Find the next available dataset number
        existing_nums = []
        for f in self.pose_dir.glob("sensor_data_*.json"):
            try:
                num = int(f.stem.split('_')[-1])
                existing_nums.append(num)
            except:
                pass
        
        next_num = max(existing_nums) + 1 if existing_nums else 1
        
        # Generate synthetic datasets from each source
        generated_count = 0
        
        for idx, (sensor_src, camera_src) in enumerate(zip(sensor_datasets, camera_datasets)):
            print(f"\nUsing source dataset {idx + 1} as template...")
            
            for i in range(num_synthetic):
                synthetic_num = next_num + generated_count
                
                # Generate synthetic pair
                synth_sensor, synth_camera = self.generate_synthetic_dataset(
                    sensor_src, camera_src, synthetic_num
                )
                
                # Save synthetic sensor data
                sensor_path = self.pose_dir / f"sensor_data_{synthetic_num}.json"
                with open(sensor_path, 'w') as f:
                    json.dump(synth_sensor, f, indent=2)
                
                # Save synthetic camera data
                camera_path = self.pose_dir / f"camera_data_{synthetic_num}.json"
                with open(camera_path, 'w') as f:
                    json.dump(synth_camera, f, indent=2)
                
                print(f"    ✓ Created synthetic dataset {synthetic_num}")
                generated_count += 1
        
        print(f"\n{'='*70}")
        print(f"SYNTHETIC DATA GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Generated {generated_count} new synthetic datasets")
        print(f"  Pose '{self.pose_name}' now has {len(sensor_datasets) + generated_count} total datasets")
        print(f"\nNext steps:")
        print(f"  1. Verify data: python verify_no_leakage.py")
        print(f"  2. Merge data: python enhanced_merger.py --all")
        print(f"  3. Train model: python fixed_trainer.py --dataset data/training_dataset.json")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic camera and sensor data from existing datasets'
    )
    parser.add_argument('--pose', type=str, help='Pose name (e.g., fist, open, point)')
    parser.add_argument('--all-poses', action='store_true', 
                       help='Generate for all poses found in data directory')
    parser.add_argument('--num-synthetic', type=int, default=3,
                       help='Number of synthetic datasets to generate per source (default: 3)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    
    args = parser.parse_args()
    
    if not args.pose and not args.all_poses:
        parser.error("Must specify either --pose or --all-poses")
    
    data_dir = Path(args.data_dir)
    
    if args.all_poses:
        # Find all pose directories
        pose_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        poses = [d.name for d in pose_dirs]
        
        if len(poses) == 0:
            print(f"❌ No pose directories found in {data_dir}")
            return
        
        print(f"\nFound {len(poses)} poses: {poses}")
        print(f"Generating {args.num_synthetic} synthetic datasets per source for each pose")
        print("="*70)
        
        for pose in poses:
            generator = SyntheticDataGenerator(pose, str(data_dir))
            generator.generate_multiple_synthetic_datasets(args.num_synthetic)
    else:
        # Single pose
        if not (data_dir / args.pose).exists():
            print(f"❌ Pose directory not found: {data_dir / args.pose}")
            return
        
        generator = SyntheticDataGenerator(args.pose, str(data_dir))
        generator.generate_multiple_synthetic_datasets(args.num_synthetic)


if __name__ == "__main__":
    main()