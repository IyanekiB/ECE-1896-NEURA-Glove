"""
NEURA GLOVE - Camera Collector (FIXED for reliable 10Hz and hand detection)
Uses active polling and requires hand to be detected before starting

Usage:
    python fixed_camera_collector.py --pose fist --output data/fist/camera_data.json
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

FRAMES_PER_POSE = 300
TARGET_FPS = 10
FRAME_INTERVAL = 0.1  # 100ms


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CameraFrame:
    """Single MediaPipe frame"""
    frame_number: int
    joints: List[dict]
    confidence: float


# ============================================================================
# FIXED CAMERA COLLECTOR
# ============================================================================

class FixedCameraCollector:
    """Uses active polling and ensures hand is detected"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.frames: List[CameraFrame] = []
        
        self.finger_chains = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
    
    def initialize(self):
        """Initialize MediaPipe with better settings"""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower threshold
            min_tracking_confidence=0.5,    # Lower threshold  
            model_complexity=1  # Use full model
        )
        print("✓ MediaPipe initialized\n")
    
    def calculate_rotation(self, point1: np.ndarray, point2: np.ndarray) -> List[float]:
        """Calculate quaternion"""
        direction = point2 - point1
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        up = np.array([0, -1, 0])
        right = np.cross(up, direction)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(direction, right)
        
        rotation_matrix = np.column_stack([right, up, direction])
        
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w, x, y, z = 0.25 * s, (rotation_matrix[2,1] - rotation_matrix[1,2]) / s, (rotation_matrix[0,2] - rotation_matrix[2,0]) / s, (rotation_matrix[1,0] - rotation_matrix[0,1]) / s
        else:
            if rotation_matrix[0,0] > rotation_matrix[1,1] and rotation_matrix[0,0] > rotation_matrix[2,2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0,0] - rotation_matrix[1,1] - rotation_matrix[2,2])
                w, x, y, z = (rotation_matrix[2,1] - rotation_matrix[1,2]) / s, 0.25 * s, (rotation_matrix[0,1] + rotation_matrix[1,0]) / s, (rotation_matrix[0,2] + rotation_matrix[2,0]) / s
            elif rotation_matrix[1,1] > rotation_matrix[2,2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1,1] - rotation_matrix[0,0] - rotation_matrix[2,2])
                w, x, y, z = (rotation_matrix[0,2] - rotation_matrix[2,0]) / s, (rotation_matrix[0,1] + rotation_matrix[1,0]) / s, 0.25 * s, (rotation_matrix[1,2] + rotation_matrix[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2,2] - rotation_matrix[0,0] - rotation_matrix[1,1])
                w, x, y, z = (rotation_matrix[1,0] - rotation_matrix[0,1]) / s, (rotation_matrix[0,2] + rotation_matrix[2,0]) / s, (rotation_matrix[1,2] + rotation_matrix[2,1]) / s, 0.25 * s
        
        return [float(x), float(y), float(z), float(w)]
    
    def process_landmarks(self, landmarks) -> Tuple[np.ndarray, List[List[float]]]:
        """Extract positions and rotations"""
        positions = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        rotations = []
        for i in range(21):
            if i == 0:
                rotation = self.calculate_rotation(positions[0], positions[9])
            else:
                parent_idx = None
                for chain in self.finger_chains:
                    if i in chain:
                        chain_pos = chain.index(i)
                        if chain_pos > 0:
                            parent_idx = chain[chain_pos - 1]
                        break
                
                rotation = self.calculate_rotation(positions[parent_idx], positions[i]) if parent_idx is not None else [0.0, 0.0, 0.0, 1.0]
            
            rotations.append(rotation)
        
        return positions, rotations
    
    def wait_for_hand_detection(self, cap):
        """Wait until hand is stably detected"""
        print("Waiting for hand detection...")
        print("(Make sure your hand is clearly visible to the camera)\n")
        
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
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(frame, f"Hand detected! ({detection_count}/{required_detections})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                detection_count = 0  # Reset if hand lost
                cv2.putText(frame, "NO HAND DETECTED - Show your hand", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        print("✓ Hand detected and stable!\n")
        cv2.destroyWindow('Hand Detection')
        return True
    
    def collect_pose(self, pose_name: str, display: bool = True):
        """
        Collect exactly 10 frames at 10Hz using ACTIVE POLLING
        """
        print("="*60)
        print(f"CAMERA COLLECTION - POSE: {pose_name.upper()}")
        print("="*60)
        print(f"Collecting {FRAMES_PER_POSE} frames at {TARGET_FPS}Hz")
        print("="*60)
        print()
        
        self.initialize()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        
        # Set camera to 30fps to avoid frame delays
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened")
        
        # Wait for hand to be detected
        if not self.wait_for_hand_detection(cap):
            print("✗ Collection cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return []
        
        print(f"Hold '{pose_name}' pose STEADY...")
        print("Starting collection in 3 seconds...\n")
        time.sleep(3)
        
        print("Collecting frames...\n")
        start_time = time.time()
        failed_detections = 0
        
        # ACTIVE POLLING: Sample exactly every 100ms
        for i in range(FRAMES_PER_POSE):
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
                
                joints = []
                for j in range(21):
                    joints.append({
                        'joint_id': j,
                        'position': positions[j].tolist(),
                        'rotation': rotations[j]
                    })
                
                camera_frame = CameraFrame(
                    frame_number=len(self.frames),
                    joints=joints,
                    confidence=1.0
                )
                
                self.frames.append(camera_frame)
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                print(f"  Frame {i+1}/{FRAMES_PER_POSE} ✓ | Wrist: [{positions[0][0]:.3f}, {positions[0][1]:.3f}, {positions[0][2]:.3f}]")
                failed_detections = 0
            else:
                failed_detections += 1
                print(f"  Frame {i+1}/{FRAMES_PER_POSE} ✗ | Hand not detected!")
                
                if failed_detections >= 3:
                    print("\n⚠ Lost hand detection! Make sure hand is visible")
            
            # Display
            if display:
                cv2.putText(frame, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {len(self.frames)}/{FRAMES_PER_POSE}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f'Collecting: {pose_name}', frame)
                cv2.waitKey(1)
            
            # Sleep until next 100ms interval
            elapsed = time.time() - frame_start
            sleep_time = max(0, FRAME_INTERVAL - elapsed)
            time.sleep(sleep_time)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        duration = time.time() - start_time
        actual_fps = len(self.frames) / duration if duration > 0 else 0
        
        print()
        print("="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Frames collected: {len(self.frames)}/{FRAMES_PER_POSE}")
        print(f"Duration: {duration:.2f}s")
        print(f"Target FPS: {TARGET_FPS}")
        print(f"Actual FPS: {actual_fps:.2f}")
        print("="*60)
        print()
        
        return self.frames
    
    def save_dataset(self, frames: List[CameraFrame], pose_name: str, output_path: str):
        """Save to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'pose_name': pose_name,
                'collection_date': datetime.now().isoformat(),
                'total_frames': len(frames),
                'target_frames': FRAMES_PER_POSE,
                'target_fps': TARGET_FPS,
                'data_type': 'MEDIAPIPE_CAMERA',
                'format': 'pose_based'
            },
            'frames': [asdict(frame) for frame in frames]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Frames: {len(frames)}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', '-p', required=True)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"data/{args.pose}/camera_data.json"
    
    collector = FixedCameraCollector()
    frames = collector.collect_pose(args.pose, display=not args.no_display)
    
    if len(frames) == FRAMES_PER_POSE:
        collector.save_dataset(frames, args.pose, args.output)
        print(f"\n✓ SUCCESS! Collected {FRAMES_PER_POSE} frames for '{args.pose}'")
        print(f"  Next step: Merge with sensor data using dataset_merger.py")
    else:
        print(f"\n✗ Only got {len(frames)}/{FRAMES_PER_POSE} frames - try again")
        print("Tips:")
        print("  - Make sure hand is clearly visible")
        print("  - Good lighting is important")
        print("  - Hold pose steady during collection")


if __name__ == "__main__":
    main()