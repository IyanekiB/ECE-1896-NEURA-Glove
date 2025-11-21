"""
MediaPipe Camera Data Collector - Single Pose Version
Collects hand joint rotations from camera using MediaPipe for ONE pose with custom duration
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
import os


class MediaPipeCollector:
    def __init__(self, pose_name, session_id, output_dir="data/camera_recordings"):
        self.pose_name = pose_name
        self.session_id = session_id
        self.output_dir = output_dir
        self.samples = []
        self.tick_index = 0
        
        # Create output directory structure
        os.makedirs(f"{output_dir}/{session_id}/{pose_name}", exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Target joints for Unity (only these 5 need Y-axis rotation)
        self.target_joints = {
            3: 'thumb_ip',      # THUMB_IP
            6: 'index_pip',     # INDEX_FINGER_PIP
            10: 'middle_pip',   # MIDDLE_FINGER_PIP
            14: 'ring_pip',     # RING_FINGER_PIP
            18: 'pinky_pip'     # PINKY_PIP
        }
        
        self.start_time = None
        
    def calculate_bend_angle(self, joint_idx, landmarks):
        """Calculate bend angle for a finger joint
        Returns angle in degrees (0 = straight, 90 = fully bent)
        """
        # Define parent-current-child triplets for each joint
        joint_chains = {
            3: (2, 3, 4),      # Thumb IP
            6: (5, 6, 7),      # Index PIP
            10: (9, 10, 11),   # Middle PIP
            14: (13, 14, 15),  # Ring PIP
            18: (17, 18, 19)   # Pinky PIP
        }
        
        if joint_idx not in joint_chains:
            return 0.0
        
        parent_idx, current_idx, child_idx = joint_chains[joint_idx]
        
        # Get 3D positions
        p1 = np.array([landmarks[parent_idx].x, landmarks[parent_idx].y, landmarks[parent_idx].z])
        p2 = np.array([landmarks[current_idx].x, landmarks[current_idx].y, landmarks[current_idx].z])
        p3 = np.array([landmarks[child_idx].x, landmarks[child_idx].y, landmarks[child_idx].z])
        
        # Calculate vectors
        v1 = p1 - p2  # Parent to current
        v2 = p3 - p2  # Current to child
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Convert to bend angle (0-90 degrees)
        # Straight finger = 180 deg between vectors = 0 bend
        # Bent finger = smaller angle between vectors = higher bend
        bend_angle = 90.0 - np.degrees(angle_rad / 2.0)
        bend_angle = np.clip(bend_angle, 0.0, 90.0)
        
        return float(bend_angle)
    
    def calculate_x_rotation(self, joint_idx, landmarks):
        """Calculate X-axis rotation (side-to-side waggle)
        For now, return 0 as we're focusing on Y-axis bending
        """
        return 0.0
    
    def process_hand_landmarks(self, landmarks):
        """Extract rotation data for target joints"""
        joint_rotations = {}
        
        for joint_idx, joint_name in self.target_joints.items():
            # Calculate Y-axis rotation (bend angle)
            y_rotation = self.calculate_bend_angle(joint_idx, landmarks)
            
            # X-axis rotation (minimal for now)
            x_rotation = self.calculate_x_rotation(joint_idx, landmarks)
            
            joint_rotations[joint_name] = {
                'joint_index': joint_idx,
                'x_rotation': x_rotation,  # Side waggle
                'y_rotation': y_rotation   # Finger bend (primary)
            }
        
        return joint_rotations
    
    def collect_pose(self, duration_seconds, show_video=True):
        """Collect data for specified duration"""
        print(f"\n{'='*60}")
        print(f"Collecting pose: {self.pose_name}")
        print(f"Session: {self.session_id}")
        print(f"Duration: {duration_seconds}s")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return False
        
        # Countdown
        print("\nStarting collection in:")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        print(f"\n🎥 RECORDING - Hold the '{self.pose_name}' pose for {duration_seconds} seconds!")
        self.start_time = time.time()
        is_collecting = True
        
        while is_collecting:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract joint rotations
                joint_rotations = self.process_hand_landmarks(hand_landmarks.landmark)
                
                # Store sample
                sample = {
                    'tick_index': self.tick_index,
                    'timestamp': time.time() - self.start_time,
                    'pose_name': self.pose_name,
                    'session_id': self.session_id,
                    'joint_rotations': joint_rotations
                }
                self.samples.append(sample)
                self.tick_index += 1
                
                # Draw landmarks
                if show_video:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display status
            if show_video:
                elapsed = time.time() - self.start_time
                remaining = max(0, duration_seconds - elapsed)
                
                cv2.putText(frame, f"Pose: {self.pose_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {self.tick_index}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if results.multi_hand_landmarks is None:
                    cv2.putText(frame, "NO HAND DETECTED", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('MediaPipe Collection', frame)
            
            # Check if duration complete
            if time.time() - self.start_time >= duration_seconds:
                is_collecting = False
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCollection stopped by user")
                is_collecting = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Collection complete! Collected {len(self.samples)} samples")
        return True
    
    def save_data(self):
        """Save collected data to JSON"""
        if not self.samples:
            print("No data to save!")
            return None
        
        output_data = {
            'metadata': {
                'pose_name': self.pose_name,
                'session_id': self.session_id,
                'total_samples': len(self.samples),
                'duration_seconds': self.samples[-1]['timestamp'],
                'collection_time': datetime.now().isoformat(),
                'sampling_rate_hz': len(self.samples) / self.samples[-1]['timestamp'],
                'target_joints': self.target_joints,
                'hand': 'left',  # Unity expects left hand
                'recording_hand': 'right'  # Actually recorded right hand
            },
            'samples': self.samples
        }
        
        filename = f"{self.output_dir}/{self.session_id}/{self.pose_name}/camera_data.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved: {filename}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Duration: {self.samples[-1]['timestamp']:.2f}s")
        print(f"  Avg rate: {output_data['metadata']['sampling_rate_hz']:.1f} Hz")
        
        return filename


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("\nUsage:")
        print("  python camera_collector_single.py <session_id> <pose_name> <duration_seconds>")
        print("\nExample:")
        print("  python camera_collector_single.py session_001 fist 60")
        print("\nThis will collect 'fist' pose camera data for 60 seconds.")
        print("\nCommon pose names: flat_hand, fist, grab, pointing, peace_sign, ok_sign")
        print("\n⚠️  IMPORTANT: Remove the glove before camera collection!")
        print("    (Camera needs clear view of hand)")
        sys.exit(1)
    
    session_id = sys.argv[1]
    pose_name = sys.argv[2]
    duration_seconds = int(sys.argv[3])
    
    if duration_seconds < 1 or duration_seconds > 300:
        print("ERROR: Duration must be between 1 and 300 seconds")
        sys.exit(1)
    
    # Create collector
    collector = MediaPipeCollector(pose_name, session_id)
    
    # Collect data
    success = collector.collect_pose(duration_seconds, show_video=True)
    
    if success:
        collector.save_data()
        print("\n✓ Camera data collection complete!")
    else:
        print("\n✗ Camera data collection failed!")