"""
NEURA GLOVE - MediaPipe Camera Data Collector
Collects hand landmark data at 10Hz and saves to JSON

Usage:
    python camera_collector.py --duration 60 --output camera_data.json
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

@dataclass
class CameraConfig:
    """Camera collection configuration"""
    TARGET_FPS: int = 10
    FRAME_INTERVAL: float = 0.1  # 100ms
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class JointData:
    """Single hand joint with position and rotation"""
    joint_id: int
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [qx, qy, qz, qw] quaternion


@dataclass
class CameraFrame:
    """Single MediaPipe frame"""
    frame_number: int
    joints: List[dict]  # 21 joints
    confidence: float


# ============================================================================
# MEDIAPIPE COLLECTOR
# ============================================================================

class MediaPipeCameraCollector:
    """Collects hand landmark data at 10Hz using MediaPipe"""
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.frames: List[CameraFrame] = []
        self.frame_count = 0
        self.last_sample_time = 0
        
        # Hand landmark chains for rotation calculation
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
    
    def initialize(self):
        """Initialize MediaPipe Hands"""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        print("✓ MediaPipe Hands initialized")
    
    def calculate_rotation(self, point1: np.ndarray, point2: np.ndarray) -> List[float]:
        """Calculate rotation quaternion between two points"""
        direction = point2 - point1
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Default up vector
        up = np.array([0, -1, 0])
        right = np.cross(up, direction)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(direction, right)
        
        # Rotation matrix to quaternion
        rotation_matrix = np.column_stack([right, up, direction])
        
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s
        
        return [float(x), float(y), float(z), float(w)]
    
    def process_landmarks(self, landmarks) -> Tuple[np.ndarray, List[List[float]]]:
        """Extract positions and calculate rotations for all 21 joints"""
        # Extract 3D positions
        positions = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks.landmark
        ])
        
        # Calculate rotations for each joint
        rotations = []
        for i in range(21):
            if i == 0:
                # Wrist - use direction to middle finger MCP
                rotation = self.calculate_rotation(positions[0], positions[9])
            else:
                # Find parent joint in finger chain
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
    
    def process_frame(self, frame: np.ndarray) -> Optional[CameraFrame]:
        """Process single video frame with 10Hz rate limiting"""
        current_time = time.time()
        
        # Rate limit to exactly 10Hz
        if current_time - self.last_sample_time < self.config.FRAME_INTERVAL:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            positions, rotations = self.process_landmarks(hand_landmarks)
            
            # Build joints array
            joints = []
            for i in range(21):
                joint = {
                    'joint_id': i,
                    'position': positions[i].tolist(),
                    'rotation': rotations[i]
                }
                joints.append(joint)
            
            camera_frame = CameraFrame(
                frame_number=self.frame_count,
                joints=joints,
                confidence=1.0
            )
            
            self.frames.append(camera_frame)
            self.frame_count += 1
            self.last_sample_time = current_time
            
            # Draw landmarks on frame
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            return camera_frame
        
        return None
    
    def collect(self, duration_seconds: int, display: bool = True):
        """
        Collect camera data for specified duration
        
        Args:
            duration_seconds: How long to collect data
            display: Whether to show video feed
            
        Returns:
            List of CameraFrame objects
        """
        print("="*60)
        print("MEDIAPIPE CAMERA DATA COLLECTION")
        print("="*60)
        print(f"Duration: {duration_seconds} seconds")
        print(f"Target rate: {self.config.TARGET_FPS}Hz")
        print(f"Expected frames: ~{duration_seconds * self.config.TARGET_FPS}")
        print("="*60)
        print()
        
        self.initialize()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        
        print("✓ Camera opened")
        print("Collecting data... Press 'q' to stop early\n")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to grab frame")
                    break
                
                # Process frame (rate-limited to 10Hz)
                camera_frame = self.process_frame(frame)
                
                # Display info on frame
                elapsed = time.time() - start_time
                remaining = max(0, duration_seconds - elapsed)
                
                cv2.putText(frame, f"Frames: {self.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {int(remaining)}s", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if camera_frame is None:
                    cv2.putText(frame, "NO HAND DETECTED", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Print progress every 10 frames
                    if self.frame_count % 10 == 0:
                        wrist = camera_frame.joints[0]
                        print(f"  Frame {self.frame_count:4d} | "
                              f"Wrist pos: [{wrist['position'][0]:.3f}, {wrist['position'][1]:.3f}, {wrist['position'][2]:.3f}]")
                
                if display:
                    cv2.imshow('MediaPipe Hand Tracking - Press Q to stop', frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or elapsed >= duration_seconds:
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠ Collection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
        
        print()
        print("="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Total frames collected: {len(self.frames)}")
        print(f"Actual duration: {time.time() - start_time:.2f}s")
        print(f"Actual rate: {len(self.frames) / (time.time() - start_time):.2f}Hz")
        print("="*60)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[CameraFrame], output_path: str):
        """Save collected camera frames to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        dataset = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_fps': self.config.TARGET_FPS,
                'data_type': 'MEDIAPIPE_CAMERA',
                'format': 'frame_by_frame',
                'description': 'MediaPipe hand landmarks (21 joints with position and rotation) at 10Hz'
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Dataset saved to: {output_file}")
        print(f"  Total frames: {len(frames)}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect MediaPipe camera data from hand at 10Hz'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Collection duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='camera_data.json',
        help='Output JSON file path (default: camera_data.json)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display window'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = MediaPipeCameraCollector()
    
    # Collect data
    frames = collector.collect(args.duration, display=not args.no_display)
    
    # Save dataset
    if frames:
        collector.save_dataset(frames, args.output)
        print(f"\n✓ Success! Collected {len(frames)} frames")
        print(f"  Next step: Merge with sensor data using dataset_merger.py")
    else:
        print("\n✗ No frames collected! Make sure your hand is visible to the camera")


if __name__ == "__main__":
    main()