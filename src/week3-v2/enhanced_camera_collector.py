"""
NEURA GLOVE - Enhanced Camera Collector (MODIFIED FOR UNITY VR COMPATIBILITY)
Supports multiple dataset collection per pose with 500 frames each

MODIFICATIONS:
- Uses palm coordinate system from mp_udp_streamer.py
- Calculates rotations matching Unity VR hand controller expectations
- Generates 21 quaternions compatible with FrameConstructor
- Training data format matches inference engine output

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
    """Single MediaPipe frame with Unity-compatible format"""
    frame_number: int
    timestamp: float
    joints: List[dict]
    confidence: float
    raw_landmarks: List[List[float]]  # Store original landmarks for debugging


# ============================================================================
# ROTATION CALCULATION METHODS (from mp_udp_streamer.py)
# ============================================================================

class RotationCalculator:
    """
    Rotation calculation methods from mp_udp_streamer.py
    Ensures compatibility with Unity VR hand controller
    """
    
    @staticmethod
    def calculate_palm_coordinate_system(landmarks):
        """
        Create a coordinate system for the palm using finger MCPs.
        Returns: (palm_forward, palm_right, palm_up) as unit vectors
        """
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Palm forward: from wrist toward average of finger MCPs
        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0
        palm_forward = palm_center - wrist
        palm_forward = palm_forward / (np.linalg.norm(palm_forward) + 1e-10)
        
        # Palm right: from pinky MCP toward index MCP
        palm_right = index_mcp - pinky_mcp
        palm_right = palm_right / (np.linalg.norm(palm_right) + 1e-10)
        
        # Palm up: perpendicular to forward and right
        palm_up = np.cross(palm_forward, palm_right)
        palm_up = palm_up / (np.linalg.norm(palm_up) + 1e-10)
        
        # Re-orthogonalize palm_right to ensure perpendicularity
        palm_right = np.cross(palm_up, palm_forward)
        palm_right = palm_right / (np.linalg.norm(palm_right) + 1e-10)
                
        return palm_forward, palm_right, palm_up
    
    @staticmethod
    def transform_to_palm_local(landmarks, palm_forward, palm_right, palm_up):
        """
        Transform landmarks to palm-local coordinate system
        """
        # Build rotation matrix
        rotation_matrix = np.column_stack([palm_right, palm_up, palm_forward])
        
        # Transform all landmarks
        local_landmarks = []
        for lm in landmarks:
            local_lm = rotation_matrix.T @ lm
            local_landmarks.append(local_lm)
        
        return np.array(local_landmarks)
    
    @staticmethod
    def calculate_rotation_from_vectors(v1, v2, angle_scale=1.0):
        """
        Calculate quaternion rotation from one vector to another.
        
        Args:
            v1: Starting vector (normalized)
            v2: Target vector (normalized)
            angle_scale: Optional scaling factor for the rotation angle
        
        Returns: [x, y, z, w] quaternion
        """
        axis = np.cross(v1, v2)
        axis_len = np.linalg.norm(axis)
        
        if axis_len > 1e-6:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle = angle * angle_scale
            
            half_angle = angle / 2.0
            s = np.sin(half_angle)
            c = np.cos(half_angle)
            
            return [
                float(axis[0] * s),
                float(axis[1] * s),
                float(axis[2] * s),
                float(c)
            ]
        else:
            # Vectors are parallel - no rotation needed
            return [0.0, 0.0, 0.0, 1.0]
    
    @staticmethod
    def project_vector_onto_plane(v, plane_normal):
        """
        Project a vector onto a plane defined by its normal.
        """
        parallel_component = np.dot(v, plane_normal) * plane_normal
        projected = v - parallel_component
        return projected
    
    @staticmethod
    def euler_to_quaternion(euler_x, euler_y, euler_z):
        """
        Convert Unity Euler angles (in degrees, XYZ order) to quaternion.
        """
        # Convert to radians
        ex = np.radians(euler_x)
        ey = np.radians(euler_y)
        ez = np.radians(euler_z)
        
        # Calculate quaternion components
        cy = np.cos(ey * 0.5)
        sy = np.sin(ey * 0.5)
        cp = np.cos(ex * 0.5)
        sp = np.sin(ex * 0.5)
        cr = np.cos(ez * 0.5)
        sr = np.sin(ez * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [float(x), float(y), float(z), float(w)]
    
    @staticmethod
    def calculate_constrained_rotation(v1, v2, hinge_axis):
        """
        Calculate rotation constrained to single axis (hinge joint).
        """
        # Normalize hinge axis
        hinge_axis = hinge_axis / (np.linalg.norm(hinge_axis) + 1e-10)
        
        # Project vectors onto plane perpendicular to hinge axis
        v1_proj = RotationCalculator.project_vector_onto_plane(v1, hinge_axis)
        v2_proj = RotationCalculator.project_vector_onto_plane(v2, hinge_axis)
        
        # Normalize projected vectors
        v1_proj = v1_proj / (np.linalg.norm(v1_proj) + 1e-10)
        v2_proj = v2_proj / (np.linalg.norm(v2_proj) + 1e-10)
        
        # Calculate angle between projected vectors
        dot_product = np.clip(np.dot(v1_proj, v2_proj), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # Determine rotation direction using cross product
        cross = np.cross(v1_proj, v2_proj)
        if np.dot(cross, hinge_axis) < 0:
            angle = -angle
        
        # Convert to quaternion
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        c = np.cos(half_angle)
        
        return [
            float(hinge_axis[0] * s),
            float(hinge_axis[1] * s),
            float(hinge_axis[2] * s),
            float(c)
        ]


# ============================================================================
# ENHANCED CAMERA COLLECTOR
# ============================================================================

class EnhancedCameraCollector:
    """Collects multiple datasets per pose with MediaPipe using Unity-compatible rotations"""
    
    def __init__(self, mirror_image=True):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.frames: List[CameraFrame] = []
        self.collection_start_time: float = 0
        self.mirror_image = mirror_image
        
        # MediaPipe joint structure (21 joints per hand)
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        # Hinge axes (from mp_udp_streamer.py)
        self.hinge_axis_fingers = np.array([1.0, 0.0, 0.0])  # X-axis for fingers
        # Thumb has special axis
        thumb_axis = np.array([1.0, -0.5, 0.0])
        self.hinge_axis_thumb = thumb_axis / np.linalg.norm(thumb_axis)
    
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
    
    def remap_landmarks_to_unity(self, landmarks):
        """
        Convert MediaPipe coordinates to Unity coordinates.
        """
        remapped = np.copy(landmarks)
        remapped[:, 1] = -remapped[:, 1]  # Flip Y
        remapped[:, 2] = -remapped[:, 2]  # Flip Z
        
        if self.mirror_image:
            remapped[:, 0] = -remapped[:, 0]
        
        # Center at wrist
        wrist = remapped[0].copy()
        remapped = remapped - wrist
        return remapped
    
    def calculate_hand_rotations(self, unity_landmarks):
        """
        Calculate 21 quaternion rotations for hand joints.
        Uses palm coordinate system and constrained rotation for realistic hand poses.
        """
        # Calculate palm coordinate system
        palm_forward, palm_right, palm_up = \
            RotationCalculator.calculate_palm_coordinate_system(unity_landmarks)
        
        # Transform to palm-local space
        local_landmarks = RotationCalculator.transform_to_palm_local(
            unity_landmarks, palm_forward, palm_right, palm_up
        )
        
        rotations = []
        
        # Wrist rotation (currently identity)
        rotations.append([0.0, 0.0, 0.0, 1.0])
        
        # Process each finger
        for chain in self.finger_chains:
            is_thumb = (chain[0] == 0 and chain[1] == 1)
            hinge_axis = self.hinge_axis_thumb if is_thumb else self.hinge_axis_fingers
            
            for i in range(1, len(chain)):
                current_idx = chain[i]
                parent_idx = chain[i-1]
                
                # SPECIAL: Thumb metacarpal (CMC joint) - set fixed orientation
                if is_thumb and i == 1:
                    thumb_metacarpal_rot = RotationCalculator.euler_to_quaternion(
                        21.194, 43.526, -69.284
                    )
                    rotations.append(thumb_metacarpal_rot)
                    continue
                
                # Check if this joint has a child (not a fingertip)
                if i + 1 < len(chain):
                    child_idx = chain[i+1]
                    
                    # Calculate vectors between joints in palm-local space
                    v1 = local_landmarks[current_idx] - local_landmarks[parent_idx]
                    v2 = local_landmarks[child_idx] - local_landmarks[current_idx]
                    
                    # Normalize
                    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                    v2 = v2 / (np.linalg.norm(v2) + 1e-10)
                    
                    # Calculate constrained rotation (single-axis hinge)
                    rotation = RotationCalculator.calculate_constrained_rotation(
                        v1, v2, hinge_axis
                    )
                    
                    rotations.append(rotation)
                else:
                    # Fingertip - no rotation
                    rotations.append([0.0, 0.0, 0.0, 1.0])
        
        return rotations
    
    def process_landmarks(self, landmarks) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Extract joint positions and rotations from MediaPipe landmarks.
        Uses Unity-compatible rotation calculation.
        
        Returns:
            positions: (21, 3) array of 3D positions
            rotations: List of 21 quaternions [x, y, z, w]
        """
        # Extract 3D positions (raw MediaPipe format)
        positions = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Remap to Unity coordinate space
        unity_landmarks = self.remap_landmarks_to_unity(positions)
        
        # Calculate rotations using palm coordinate system
        rotations = self.calculate_hand_rotations(unity_landmarks)
        
        return positions, rotations
    
    def wait_for_hand_detection(self, cap):
        """Wait until hand is stably detected before starting collection"""
        print("👁️  Waiting for hand detection...")
        print("   (Make sure your hand is clearly visible to camera)\n")
        
        detection_count = 0
        required_detections = 5
        
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
                detection_count = 0
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
                       display: bool = True) -> List[CameraFrame]:
        """
        Collect dataset for a specific pose.
        
        Args:
            pose_name: Name of the pose (e.g., "fist", "open")
            dataset_num: Dataset number for this pose
            num_frames: Number of frames to collect
            display: Whether to show camera feed
        
        Returns:
            List of CameraFrame objects
        """
        print("\n" + "="*70)
        print(f"COLLECTING DATASET {dataset_num} FOR POSE: {pose_name.upper()}")
        print("="*70)
        print(f"  Target frames: {num_frames}")
        print(f"  Target FPS: {TARGET_FPS} Hz")
        print(f"  Rotation method: Palm coordinate system (Unity compatible)")
        print("="*70)
        
        # Initialize MediaPipe
        self.initialize()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return []
        
        print("\nCamera opened successfully")
        
        # Wait for stable hand detection
        if not self.wait_for_hand_detection(cap):
            cap.release()
            cv2.destroyAllWindows()
            return []
        
        # Countdown
        print("\nStarting collection in:")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        print(" RECORDING!\n")
        
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
                
                # Build joint data (Unity-compatible format)
                joints = []
                for j in range(21):
                    joints.append({
                        'joint_id': j,
                        'position': [0.0, 0.0, 0.0],  # Position set to origin (like Unity VR code)
                        'rotation': rotations[j]  # Quaternion [x, y, z, w]
                    })
                
                camera_frame = CameraFrame(
                    frame_number=len(self.frames),
                    timestamp=time.time() - self.collection_start_time,
                    joints=joints,
                    confidence=1.0,
                    raw_landmarks=positions.tolist()  # Store raw landmarks for debugging
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
                    # Show first rotation as sample
                    sample_rot = rotations[5]  # Index MCP rotation
                    print(f"  Frame {i+1:3d}/{num_frames} ({progress:5.1f}%) ✓ | "
                          f"Time: {elapsed:6.2f}s | "
                          f"Sample Quat: [{sample_rot[0]:.3f}, {sample_rot[1]:.3f}, "
                          f"{sample_rot[2]:.3f}, {sample_rot[3]:.3f}]")
                
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
                cv2.putText(
                    frame, "Using Unity-compatible rotations", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
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
        print(f"  Rotation Method: Palm coordinate system (Unity VR compatible)")
        print("=" * 70)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[CameraFrame], pose_name: str, 
                    dataset_num: int, output_dir: str = "data"):
        """
        Save dataset to JSON file with Unity-compatible format.
        
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
                'data_type': 'MEDIAPIPE_CAMERA_UNITY_COMPATIBLE',
                'rotation_method': 'palm_coordinate_system',
                'mirror_image': self.mirror_image,
                'joint_config': {
                    'num_joints': 21,
                    'features_per_joint': 7,  # 3 position + 4 rotation quaternion
                    'total_features': 147,
                    'position_format': '[x, y, z] (always [0,0,0] in this format)',
                    'rotation_format': '[x, y, z, w] quaternion (Unity compatible)'
                }
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 Saved to: {output_file}")
        print(f"   Frames: {len(frames)}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
        print(f"   Format: Unity VR compatible (FrameConstructor ready)")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect camera/MediaPipe data for hand pose training (Unity VR compatible)'
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
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='Disable X-axis mirroring'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = EnhancedCameraCollector(mirror_image=not args.no_mirror)
    
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
        print(f"      python enhanced_camera_collector.py --pose {args.pose} --dataset-num {args.dataset_num + 1}")
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