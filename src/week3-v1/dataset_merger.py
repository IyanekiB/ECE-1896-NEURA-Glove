"""
NEURA GLOVE - Pose-Based Dataset Merger
Merges 10-frame sensor and camera data for each pose

Usage:
    python pose_dataset_merger.py --pose fist
    # Or merge all poses:
    python pose_dataset_merger.py --all --output data/training_dataset.json
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingSample:
    """Single training sample with aligned sensor and camera data"""
    frame_number: int
    pose_name: str
    
    # INPUT: Sensor data (15 features)
    flex_sensors: List[float]
    imu_orientation: List[float]
    imu_accel: List[float]
    imu_gyro: List[float]
    
    # OUTPUT: Ground truth from MediaPipe (147 values)
    joints: List[Dict]


# ============================================================================
# POSE-BASED MERGER
# ============================================================================

class PoseDatasetMerger:
    """Merges sensor and camera data for poses (10 frames each)"""
    
    def __init__(self):
        self.training_samples: List[TrainingSample] = []
    
    def merge_pose(self, pose_name: str, data_dir: str = "data"):
        """Merge sensor and camera data for a single pose"""
        data_path = Path(data_dir) / pose_name
        sensor_file = data_path / "sensor_data.json"
        camera_file = data_path / "camera_data.json"
        
        print(f"\nMerging pose: {pose_name}")
        print("-" * 60)
        
        # Check if files exist
        if not sensor_file.exists():
            print(f"  ✗ Sensor file not found: {sensor_file}")
            return False
        
        if not camera_file.exists():
            print(f"  ✗ Camera file not found: {camera_file}")
            return False
        
        # Load data
        with open(sensor_file, 'r') as f:
            sensor_data = json.load(f)
        
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        
        sensor_frames = sensor_data['frames']
        camera_frames = camera_data['frames']
        
        print(f"  Sensor frames: {len(sensor_frames)}")
        print(f"  Camera frames: {len(camera_frames)}")
        
        # Both should have exactly 10 frames
        if len(sensor_frames) != 300 or len(camera_frames) != 300:
            print(f"  ⚠ Warning: Expected 10 frames each, got {len(sensor_frames)} and {len(camera_frames)}")
        
        # Merge frame-by-frame
        num_frames = min(len(sensor_frames), len(camera_frames))
        
        for i in range(num_frames):
            sample = TrainingSample(
                frame_number=i,
                pose_name=pose_name,
                flex_sensors=sensor_frames[i]['flex_sensors'],
                imu_orientation=sensor_frames[i]['imu_orientation'],
                imu_accel=sensor_frames[i]['imu_accel'],
                imu_gyro=sensor_frames[i]['imu_gyro'],
                joints=camera_frames[i]['joints']
            )
            self.training_samples.append(sample)
        
        print(f"  ✓ Merged {num_frames} frames for pose '{pose_name}'")
        return True
    
    def merge_all_poses(self, pose_list: List[str], data_dir: str = "data"):
        """Merge all poses in the list"""
        print("="*60)
        print("MERGING ALL POSES")
        print("="*60)
        
        successful = 0
        failed = 0
        
        for pose in pose_list:
            if self.merge_pose(pose, data_dir):
                successful += 1
            else:
                failed += 1
        
        print()
        print("="*60)
        print("MERGE SUMMARY")
        print("="*60)
        print(f"  Successful poses: {successful}")
        print(f"  Failed poses: {failed}")
        print(f"  Total samples: {len(self.training_samples)}")
        print("="*60)
    
    def save_training_dataset(self, output_path: str):
        """Save merged training dataset"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'total_samples': len(self.training_samples),
                'poses': list(set(s.pose_name for s in self.training_samples)),
                'samples_per_pose': 300,
                'input_features': 15,
                'output_values': 147,
                'format': 'pose_based_training_data'
            },
            'samples': [asdict(sample) for sample in self.training_samples]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✓ Training dataset saved to: {output_file}")
        print(f"  Total samples: {len(self.training_samples)}")
        print(f"  Poses included: {dataset['metadata']['poses']}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge BLE sensor and camera data for poses'
    )
    parser.add_argument(
        '--pose', '-p',
        type=str,
        help='Single pose to merge'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Merge all available poses in data directory'
    )
    parser.add_argument(
        '--poses',
        nargs='+',
        help='List of poses to merge (e.g., --poses fist open point)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Data directory containing pose folders (default: data)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/training_dataset.json',
        help='Output training dataset file'
    )
    
    args = parser.parse_args()
    
    merger = PoseDatasetMerger()
    
    if args.pose:
        # Merge single pose
        if merger.merge_pose(args.pose, args.data_dir):
            output_file = Path(args.data_dir) / args.pose / "merged_data.json"
            merger.save_training_dataset(str(output_file))
    
    elif args.poses:
        # Merge specified poses
        merger.merge_all_poses(args.poses, args.data_dir)
        merger.save_training_dataset(args.output)
    
    elif args.all:
        # Find all pose directories
        data_path = Path(args.data_dir)
        pose_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
        
        if not pose_dirs:
            print(f"No pose directories found in {args.data_dir}")
            return
        
        print(f"Found poses: {pose_dirs}")
        merger.merge_all_poses(pose_dirs, args.data_dir)
        merger.save_training_dataset(args.output)
    
    else:
        print("Please specify --pose, --poses, or --all")
        parser.print_help()


if __name__ == "__main__":
    main()