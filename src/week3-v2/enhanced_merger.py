"""
NEURA GLOVE - Enhanced Dataset Merger
Merges multiple sensor+camera datasets per pose for robust training

Usage:
    # Merge all datasets for all poses
    python enhanced_merger.py --all
    
    # Merge specific pose's datasets
    python enhanced_merger.py --pose fist
    
    # Merge with custom output
    python enhanced_merger.py --all --output data/full_training_dataset.json
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingSample:
    """Single training sample with aligned sensor and camera data"""
    frame_number: int
    pose_name: str
    dataset_number: int
    
    # INPUT: Sensor data (15 features)
    flex_sensors: List[float]
    imu_orientation: List[float]
    imu_accel: List[float]
    imu_gyro: List[float]
    
    # OUTPUT: Ground truth from MediaPipe (147 values = 21 joints Ã— 7)
    joints: List[Dict]


# ============================================================================
# ENHANCED MERGER
# ============================================================================

class EnhancedDatasetMerger:
    """Merges multiple datasets per pose for robust ML training"""
    
    def __init__(self):
        self.training_samples: List[TrainingSample] = []
        self.pose_dataset_counts = defaultdict(int)
    
    def find_datasets_for_pose(self, pose_name: str, data_dir: str = "data") -> List[int]:
        """
        Find all dataset numbers available for a given pose
        
        Returns:
            List of dataset numbers (e.g., [1, 2, 3])
        """
        pose_path = Path(data_dir) / pose_name
        if not pose_path.exists():
            return []
        
        # Find all sensor_data_*.json files
        sensor_files = list(pose_path.glob("sensor_data_*.json"))
        
        # Extract dataset numbers
        dataset_nums = []
        for f in sensor_files:
            try:
                # Extract number from "sensor_data_3.json" -> 3
                num = int(f.stem.split('_')[-1])
                dataset_nums.append(num)
            except ValueError:
                continue
        
        return sorted(dataset_nums)
    
    def merge_single_dataset(self, pose_name: str, dataset_num: int, 
                            data_dir: str = "data") -> int:
        """
        Merge one sensor+camera dataset pair
        
        Returns:
            Number of samples merged
        """
        pose_path = Path(data_dir) / pose_name
        sensor_file = pose_path / f"sensor_data_{dataset_num}.json"
        camera_file = pose_path / f"camera_data_{dataset_num}.json"
        
        # Check if both files exist
        if not sensor_file.exists():
            print(f"    âœ— Missing sensor file: {sensor_file.name}")
            return 0
        
        if not camera_file.exists():
            print(f"    âœ— Missing camera file: {camera_file.name}")
            return 0
        
        # Load data
        with open(sensor_file, 'r') as f:
            sensor_data = json.load(f)
        
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        
        sensor_frames = sensor_data['frames']
        # Camera data has 'samples' not 'frames'
        camera_frames = camera_data.get('samples', camera_data.get('frames', []))
        
        # Merge frame-by-frame
        num_frames = min(len(sensor_frames), len(camera_frames))
        
        for i in range(num_frames):
            # Camera data has 'joint_angles' not 'joints'
            joints = camera_frames[i].get('joint_angles', camera_frames[i].get('joints'))
            
            sample = TrainingSample(
                frame_number=i,
                pose_name=pose_name,
                dataset_number=dataset_num,
                flex_sensors=sensor_frames[i]['flex_sensors'],
                imu_orientation=sensor_frames[i]['imu_orientation'],
                imu_accel=sensor_frames[i]['imu_accel'],
                imu_gyro=sensor_frames[i]['imu_gyro'],
                joints=joints
            )
            self.training_samples.append(sample)
        
        print(f"    âœ“ Dataset {dataset_num}: {num_frames} frames merged")
        return num_frames
    
    def merge_pose(self, pose_name: str, data_dir: str = "data") -> bool:
        """
        Merge ALL datasets for a single pose
        
        Returns:
            True if at least one dataset was merged
        """
        print(f"\n{'='*70}")
        print(f"Merging pose: {pose_name.upper()}")
        print(f"{'='*70}")
        
        # Find all datasets for this pose
        dataset_nums = self.find_datasets_for_pose(pose_name, data_dir)
        
        if not dataset_nums:
            print(f"  âœ— No datasets found for pose '{pose_name}'")
            return False
        
        print(f"  Found {len(dataset_nums)} dataset(s): {dataset_nums}")
        
        total_frames = 0
        successful_datasets = 0
        
        # Merge each dataset
        for dataset_num in dataset_nums:
            frames = self.merge_single_dataset(pose_name, dataset_num, data_dir)
            if frames > 0:
                total_frames += frames
                successful_datasets += 1
                self.pose_dataset_counts[pose_name] += 1
        
        print(f"\n  Summary for '{pose_name}':")
        print(f"    Datasets merged: {successful_datasets}/{len(dataset_nums)}")
        print(f"    Total frames: {total_frames}")
        
        return successful_datasets > 0
    
    def merge_all_poses(self, data_dir: str = "data"):
        """
        Merge ALL datasets for ALL poses in data directory
        """
        print("\n" + "="*70)
        print("MERGING ALL POSES AND DATASETS")
        print("="*70)
        
        # Find all pose directories
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"âœ— Data directory not found: {data_dir}")
            return
        
        pose_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
        
        if not pose_dirs:
            print(f"âœ— No pose directories found in {data_dir}")
            return
        
        print(f"\nFound {len(pose_dirs)} pose(s): {sorted(pose_dirs)}")
        
        successful_poses = 0
        failed_poses = 0
        
        # Merge each pose
        for pose in sorted(pose_dirs):
            if self.merge_pose(pose, data_dir):
                successful_poses += 1
            else:
                failed_poses += 1
        
        # Final summary
        print("\n" + "="*70)
        print("MERGE COMPLETE")
        print("="*70)
        print(f"  Successful poses: {successful_poses}")
        print(f"  Failed poses: {failed_poses}")
        print(f"  Total training samples: {len(self.training_samples)}")
        print()
        print("  Dataset breakdown:")
        for pose, count in sorted(self.pose_dataset_counts.items()):
            samples = sum(1 for s in self.training_samples if s.pose_name == pose)
            print(f"    {pose:15s}: {count} datasets, {samples:4d} samples")
        print("="*70)
    
    def save_training_dataset(self, output_path: str = "data/training_dataset.json"):
        """
        Save merged training dataset
        
        Format compatible with enhanced_trainer.py
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get unique poses
        poses = sorted(set(s.pose_name for s in self.training_samples))
        
        # Calculate statistics
        stats_per_pose = {}
        for pose in poses:
            pose_samples = [s for s in self.training_samples if s.pose_name == pose]
            datasets = set(s.dataset_number for s in pose_samples)
            stats_per_pose[pose] = {
                'num_datasets': len(datasets),
                'num_samples': len(pose_samples),
                'datasets': sorted(list(datasets))
            }
        
        dataset = {
            'metadata': {
                'total_samples': len(self.training_samples),
                'total_poses': len(poses),
                'poses': poses,
                'input_features': 15,  # 5 flex + 4 quat + 3 accel + 3 gyro
                'output_features': 147,  # 21 joints Ã— 7 (3 pos + 4 rot)
                'format': 'multi_dataset_training_data',
                'pose_statistics': stats_per_pose
            },
            'samples': [asdict(sample) for sample in self.training_samples]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nðŸ’¾ Training dataset saved!")
        print(f"   File: {output_file}")
        print(f"   Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Total samples: {len(self.training_samples)}")
        print(f"   Poses: {poses}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple BLE sensor and camera datasets for training'
    )
    parser.add_argument(
        '--pose', '-p',
        type=str,
        help='Merge specific pose only'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Merge all available poses and datasets'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Data directory (default: data)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/training_dataset.json',
        help='Output training dataset file'
    )
    
    args = parser.parse_args()
    
    merger = EnhancedDatasetMerger()
    
    if args.pose:
        # Merge single pose
        if merger.merge_pose(args.pose, args.data_dir):
            merger.save_training_dataset(args.output)
        else:
            print(f"\nâœ— Failed to merge pose '{args.pose}'")
    
    elif args.all:
        # Merge all poses
        merger.merge_all_poses(args.data_dir)
        
        if len(merger.training_samples) > 0:
            merger.save_training_dataset(args.output)
            
            print(f"\nâœ… SUCCESS!")
            print(f"\nðŸ“‹ Next step:")
            print(f"   Train the model:")
            print(f"   python enhanced_trainer.py --dataset {args.output} --epochs 150")
        else:
            print(f"\nâœ— No samples to save")
    
    else:
        print("Please specify --pose or --all")
        parser.print_help()


if __name__ == "__main__":
    main()