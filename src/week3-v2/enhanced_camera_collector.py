"""
NEURA GLOVE - Enhanced Camera Collector
Supports multiple dataset collection per pose with 500 frames each

Usage:
    # Collect dataset 1 for fist pose
    python enhanced_camera_collector.py --pose fist --dataset-num 1
    
    # Collect with custom frame count
    python enhanced_camera_collector.py --pose open --dataset-num 2 --frames 600
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


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
class CameraFrame:
    """Single MediaPipe frame"""
    frame_number: int
    timestamp: float
    joints: List[dict]
    confidence: float


# ============================================================================
# ENHANCED CAMERA COLLECTOR
# ============================================================================

class EnhancedCameraCollector:
    """Collects multiple datasets per pose with MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.frames: List[CameraFrame] = []
        self.collection_start_time: float = 0
        
        # MediaPipe joint structure (21 joints per hand)
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
    
    def initialize(self):
        """Initialize MediaPipe with optimal settings"""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Full model for better accuracy
        )
        print("✓ MediaPipe initialized\n")
    
    def calculate_rotation(self, point1: np.ndarray, point2: np.ndarray) -> List[float]:
        """
        Calculate quaternion rotation from point1 to point2
        Returns: [x, y, z, w]
        """
        direction = point2 - point1
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        up = np.array([0, -1, 0])
        right = np.cross(up, direction)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(direction, right)
        
        # Build rotation matrix
        rotation_matrix = np.column_stack([right, up, direction])
        
        # Convert to quaternion
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (rotation_matrix[2,1] - rotation_matrix[1,2]) / s
            y = (rotation_matrix[0,2] - rotation_matrix[2,0]) / s
            z = (rotation_matrix[1,0] - rotation_matrix[0,1]) / s
        else:
            if rotation_matrix[0,0] > rotation_matrix[1,1] and rotation_matrix[0,0] > rotation_matrix[2,2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0,0] - rotation_matrix[1,1] - rotation_matrix[2,2])
                w = (rotation_matrix[2,1] - rotation_matrix[1,2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0,1] + rotation_matrix[1,0]) / s
                z = (rotation_matrix[0,2] + rotation_matrix[2,0]) / s
            elif rotation_matrix[1,1] > rotation_matrix[2,2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1,1] - rotation_matrix[0,0] - rotation_matrix[2,2])
                w = (rotation_matrix[0,2] - rotation_matrix[2,0]) / s
                x = (rotation_matrix[0,1] + rotation_matrix[1,0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1,2] + rotation_matrix[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2,2] - rotation_matrix[0,0] - rotation_matrix[1,1])
                w = (rotation_matrix[1,0] - rotation_matrix[0,1]) / s
                x = (rotation_matrix[0,2] + rotation_matrix[2,0]) / s
                y = (rotation_matrix[1,2] + rotation_matrix[2,1]) / s
                z = 0.25 * s
        
        return [float(x), float(y), float(z), float(w)]
    
    def process_landmarks(self, landmarks) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Extract joint positions and rotations from MediaPipe landmarks
        
        Returns:
            positions: (21, 3) array of 3D positions
            rotations: List of 21 quaternions [x, y, z, w]
        """
        # Extract 3D positions
        positions = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate rotations for each joint
        rotations = []
        for i in range(21):
            if i == 0:
                # Wrist: use direction to middle finger base
                rotation = self.calculate_rotation(positions[0], positions[9])
            else:
                # Find parent joint
                parent_idx = None
                for chain in self.finger_chains:
                    if i in chain:
                        chain_pos = chain.index(i)
                        if chain_pos > 0:
                            parent_idx = chain[chain_pos - 1]
                        break
                
                if parent_idx is not None:
                    rotation = self.calculate_rotation(positions[parent_idx], positions[i])
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            
            rotations.append(rotation)
        
        return positions, rotations
    
    def wait_for_hand_detection(self, cap):
        """Wait until hand is stably detected before starting collection"""
        print("👁️  Waiting for hand detection...")
        print("   (Make sure your hand is clearly visible to camera)\n")
        
        detection_count = 0
        required_detections = 5  # Need 5 consecutive detections
        
        while detection_count < required_detections:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                detection_count += 1
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                cv2.putText(
                    frame, 
                    f"Hand detected! ({detection_count}/{required_detections})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            else:
                detection_count = 0  # Reset if hand lost
                cv2.putText(
                    frame, 
                    "NO HAND DETECTED - Show your hand", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            cv2.imshow('Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        print("✓ Hand detected and stable!\n")
        cv2.destroyWindow('Hand Detection')
        return True
    
    def collect_dataset(self, pose_name: str, dataset_num: int,
                       num_frames: int = DEFAULT_FRAMES_PER_POSE, 
                       display: bool = True):
        """
        Collect a single dataset for a pose using MediaPipe
        
        Args:
            pose_name: Name of the pose
            dataset_num: Dataset number for this pose
            num_frames: Number of frames to collect
            display: Show camera feed during collection
        """
        print("=" * 70)
        print(f"CAMERA/MEDIAPIPE COLLECTION")
        print("=" * 70)
        print(f"  Pose: {pose_name.upper()}")
        print(f"  Dataset: #{dataset_num}")
        print(f"  Target Frames: {num_frames}")
        print(f"  Sample Rate: {TARGET_FPS}Hz")
        print(f"  Estimated Duration: {num_frames / TARGET_FPS:.1f}s")
        print("=" * 70)
        print()
        
        self.initialize()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        
        # Set camera to 30fps to avoid frame delays
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Camera opened")
        
        # Wait for hand detection
        if not self.wait_for_hand_detection(cap):
            print("✗ Collection cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return []
        
        print(f"📌 Instructions:")
        print(f"   1. Position hand in '{pose_name}' pose")
        print(f"   2. Hold pose STEADY (match BLE collection)")
        print(f"   3. Vary the pose slightly (natural movements)")
        print(f"   4. Collection starts in 3 seconds...")
        print()
        
        time.sleep(3)
        
        print("🎬 COLLECTION STARTED")
        print("-" * 70)
        
        self.collection_start_time = time.time()
        self.frames = []
        failed_detections = 0
        
        # Active polling at 10Hz
        for i in range(num_frames):
            frame_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"  Frame {i+1}: Camera read failed")
                continue
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                positions, rotations = self.process_landmarks(hand_landmarks)
                
                # Build joint data
                joints = []
                for j in range(21):
                    joints.append({
                        'joint_id': j,
                        'position': positions[j].tolist(),
                        'rotation': rotations[j]
                    })
                
                camera_frame = CameraFrame(
                    frame_number=len(self.frames),
                    timestamp=time.time() - self.collection_start_time,
                    joints=joints,
                    confidence=1.0
                )
                
                self.frames.append(camera_frame)
                
                # Draw landmarks
                if display:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Progress indicator every 50 frames
                if (i + 1) % 50 == 0 or i == 0:
                    progress = (i + 1) / num_frames * 100
                    elapsed = time.time() - self.collection_start_time
                    print(f"  Frame {i+1:3d}/{num_frames} ({progress:5.1f}%) ✓ | "
                          f"Time: {elapsed:6.2f}s | "
                          f"Wrist: [{positions[0][0]:.3f}, {positions[0][1]:.3f}, {positions[0][2]:.3f}]")
                
                failed_detections = 0
            else:
                failed_detections += 1
                if (i + 1) % 50 == 0:
                    print(f"  Frame {i+1:3d}/{num_frames} ✗ | Hand not detected!")
                
                if failed_detections >= 5:
                    print("\n⚠️  Lost hand detection! Make sure hand is visible")
            
            # Display
            if display:
                cv2.putText(
                    frame, f"Pose: {pose_name} | Dataset: {dataset_num}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Frame: {len(self.frames)}/{num_frames}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow(f'Collecting: {pose_name}', frame)
                cv2.waitKey(1)
            
            # Sleep to maintain 10Hz
            elapsed = time.time() - frame_start
            sleep_time = max(0, FRAME_INTERVAL - elapsed)
            time.sleep(sleep_time)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
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
    
    def save_dataset(self, frames: List[CameraFrame], pose_name: str, 
                    dataset_num: int, output_dir: str = "data"):
        """
        Save dataset to JSON file
        
        File structure: data/{pose_name}/camera_data_{dataset_num}.json
        """
        pose_dir = Path(output_dir) / pose_name
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = pose_dir / f"camera_data_{dataset_num}.json"
        
        dataset = {
            'metadata': {
                'pose_name': pose_name,
                'dataset_number': dataset_num,
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_fps': TARGET_FPS,
                'data_type': 'MEDIAPIPE_CAMERA',
                'joint_config': {
                    'num_joints': 21,
                    'features_per_joint': 7,  # 3 position + 4 rotation
                    'total_features': 147
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

def main():
    parser = argparse.ArgumentParser(
        description='Collect camera/MediaPipe data for hand pose training'
    )
    parser.add_argument(
        '--pose', '-p',
        required=True,
        help='Pose name (must match BLE collection)'
    )
    parser.add_argument(
        '--dataset-num', '-n',
        type=int,
        required=True,
        help='Dataset number (must match BLE collection)'
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
        '--no-display',
        action='store_true',
        help='Disable camera feed display'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = EnhancedCameraCollector()
    
    # Collect dataset
    frames = collector.collect_dataset(
        pose_name=args.pose,
        dataset_num=args.dataset_num,
        num_frames=args.frames,
        display=not args.no_display
    )
    
    # Save if successful
    if len(frames) >= args.frames * 0.9:  # Allow 10% tolerance
        collector.save_dataset(frames, args.pose, args.dataset_num, args.output_dir)
        print(f"\n✅ SUCCESS! Dataset {args.dataset_num} for '{args.pose}' collected")
        print(f"\n📋 Next steps:")
        print(f"   1. Collect more datasets for this pose:")
        print(f"      python enhanced_ble_collector.py --pose {args.pose} --dataset-num {args.dataset_num + 1}")
        print(f"   2. Or merge all datasets and train:")
        print(f"      python enhanced_merger.py --all")
        print(f"      python enhanced_trainer.py --dataset data/training_dataset.json")
    else:
        print(f"\n✗ FAILED: Only collected {len(frames)}/{args.frames} frames")
        print("   Tips:")
        print("   - Ensure good lighting")
        print("   - Keep hand clearly visible")
        print("   - Hold pose steady")


if __name__ == "__main__":
    main()