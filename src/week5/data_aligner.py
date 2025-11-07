"""
Data Alignment Script
Aligns sensor and camera recordings by pose and tick index
Creates paired training samples for ML model
"""

import json
import numpy as np
import os
from pathlib import Path
from scipy import interpolate


class DataAligner:
    """Align sensor and camera data for training"""
    
    def __init__(self, sensor_session_dir, camera_session_dir, output_dir="data/aligned"):
        self.sensor_session_dir = Path(sensor_session_dir)
        self.camera_session_dir = Path(camera_session_dir)
        self.output_dir = Path(output_dir)
        
        # Extract session IDs
        self.sensor_session_id = self.sensor_session_dir.name
        self.camera_session_id = self.camera_session_dir.name
        
        print(f"Aligning sessions:")
        print(f"  Sensor: {self.sensor_session_id}")
        print(f"  Camera: {self.camera_session_id}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pose_data(self, pose_name):
        """Load sensor and camera data for a specific pose"""
        sensor_file = self.sensor_session_dir / pose_name / "sensor_data.json"
        camera_file = self.camera_session_dir / pose_name / "camera_data.json"
        
        if not sensor_file.exists():
            print(f"  ⚠️  Missing sensor data: {pose_name}")
            return None, None
        
        if not camera_file.exists():
            print(f"  ⚠️  Missing camera data: {pose_name}")
            return None, None
        
        with open(sensor_file) as f:
            sensor_data = json.load(f)
        
        with open(camera_file) as f:
            camera_data = json.load(f)
        
        return sensor_data, camera_data
    
    def interpolate_camera_data(self, camera_samples, target_timestamps):
        """Interpolate camera data to match sensor timestamps
        Camera runs at ~30 Hz, sensor at ~10 Hz
        """
        # Extract camera timestamps and joint angles
        cam_times = [s['timestamp'] for s in camera_samples]
        
        # For each joint, interpolate its rotations
        joint_names = ['thumb_ip', 'index_pip', 'middle_pip', 'ring_pip', 'pinky_pip']
        interpolated_samples = []
        
        for target_time in target_timestamps:
            joint_rotations = {}
            
            for joint_name in joint_names:
                # Extract time series for this joint
                x_rots = [s['joint_rotations'][joint_name]['x_rotation'] for s in camera_samples]
                y_rots = [s['joint_rotations'][joint_name]['y_rotation'] for s in camera_samples]
                
                # Interpolate
                x_interp = np.interp(target_time, cam_times, x_rots)
                y_interp = np.interp(target_time, cam_times, y_rots)
                
                joint_rotations[joint_name] = {
                    'x_rotation': float(x_interp),
                    'y_rotation': float(y_interp)
                }
            
            interpolated_samples.append({
                'timestamp': target_time,
                'joint_rotations': joint_rotations
            })
        
        return interpolated_samples
    
    def create_training_pairs(self, sensor_samples, camera_samples_interp):
        """Create (input, output) pairs for training"""
        pairs = []
        
        for i, sensor_sample in enumerate(sensor_samples):
            # Input: flex angles (5 values)
            flex_angles = sensor_sample['data']['flex_angles']
            input_vector = [
                flex_angles['thumb'],
                flex_angles['index'],
                flex_angles['middle'],
                flex_angles['ring'],
                flex_angles['pinky']
            ]
            
            # Output: joint x,y rotations (10 values = 5 joints × 2 axes)
            # Order: thumb, index, middle, ring, pinky
            joint_rots = camera_samples_interp[i]['joint_rotations']
            output_vector = [
                joint_rots['thumb_ip']['x_rotation'],
                joint_rots['thumb_ip']['y_rotation'],
                joint_rots['index_pip']['x_rotation'],
                joint_rots['index_pip']['y_rotation'],
                joint_rots['middle_pip']['x_rotation'],
                joint_rots['middle_pip']['y_rotation'],
                joint_rots['ring_pip']['x_rotation'],
                joint_rots['ring_pip']['y_rotation'],
                joint_rots['pinky_pip']['x_rotation'],
                joint_rots['pinky_pip']['y_rotation']
            ]
            
            pairs.append({
                'tick_index': sensor_sample['tick_index'],
                'timestamp': sensor_sample['timestamp'],
                'input': input_vector,
                'output': output_vector,
                'pose_name': sensor_sample['pose_name']
            })
        
        return pairs
    
    def align_pose(self, pose_name):
        """Align data for a single pose"""
        print(f"\nAligning pose: {pose_name}")
        
        # Load data
        sensor_data, camera_data = self.load_pose_data(pose_name)
        if sensor_data is None or camera_data is None:
            return None
        
        sensor_samples = sensor_data['samples']
        camera_samples = camera_data['samples']
        
        print(f"  Sensor samples: {len(sensor_samples)}")
        print(f"  Camera samples: {len(camera_samples)}")
        
        # Interpolate camera data to sensor timestamps
        sensor_timestamps = [s['timestamp'] for s in sensor_samples]
        camera_samples_interp = self.interpolate_camera_data(camera_samples, sensor_timestamps)
        
        # Create training pairs
        pairs = self.create_training_pairs(sensor_samples, camera_samples_interp)
        
        print(f"  Created {len(pairs)} aligned pairs")
        
        return {
            'pose_name': pose_name,
            'pairs': pairs,
            'metadata': {
                'sensor_session': self.sensor_session_id,
                'camera_session': self.camera_session_id,
                'sensor_samples': len(sensor_samples),
                'camera_samples': len(camera_samples),
                'aligned_pairs': len(pairs)
            }
        }
    
    def align_all_poses(self):
        """Align all poses in the sessions"""
        print(f"\n{'='*60}")
        print("DATA ALIGNMENT")
        print(f"{'='*60}")
        
        # Get all pose directories from sensor session
        pose_dirs = [d for d in self.sensor_session_dir.iterdir() if d.is_dir()]
        pose_names = [d.name for d in pose_dirs]
        
        print(f"\nPoses to align: {len(pose_names)}")
        for pose in pose_names:
            print(f"  - {pose}")
        
        all_pairs = []
        pose_results = []
        
        for pose_name in pose_names:
            result = self.align_pose(pose_name)
            if result:
                all_pairs.extend(result['pairs'])
                pose_results.append(result['metadata'])
        
        # Save aligned dataset
        dataset = {
            'metadata': {
                'sensor_session': self.sensor_session_id,
                'camera_session': self.camera_session_id,
                'total_poses': len(pose_results),
                'total_pairs': len(all_pairs),
                'pose_breakdown': pose_results,
                'input_dimensions': 5,  # 5 flex angles
                'output_dimensions': 10,  # 5 joints × 2 axes
                'input_description': ['thumb_angle', 'index_angle', 'middle_angle', 'ring_angle', 'pinky_angle'],
                'output_description': [
                    'thumb_ip_x', 'thumb_ip_y',
                    'index_pip_x', 'index_pip_y',
                    'middle_pip_x', 'middle_pip_y',
                    'ring_pip_x', 'ring_pip_y',
                    'pinky_pip_x', 'pinky_pip_y'
                ]
            },
            'pairs': all_pairs
        }
        
        output_file = self.output_dir / f"aligned_{self.sensor_session_id}_{self.camera_session_id}.json"
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ALIGNMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Output file: {output_file}")
        print(f"Total training pairs: {len(all_pairs)}")
        print(f"\nPer-pose breakdown:")
        for result in pose_results:
            pose_name = result.get('pose_name') or 'unknown_pose'
            pairs = result.get('aligned_pairs', 0)
            print(f"  {pose_name}: {pairs} pairs")
        
        return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python data_aligner.py <sensor_session_dir> <camera_session_dir>")
        print("\nExample:")
        print("  python data_aligner.py data/sensor_recordings/session_001 data/camera_recordings/session_001")
        sys.exit(1)
    
    sensor_dir = sys.argv[1]
    camera_dir = sys.argv[2]
    
    aligner = DataAligner(sensor_dir, camera_dir)
    aligner.align_all_poses()
