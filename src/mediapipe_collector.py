import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
import os


class MediaPipeCollector:
    """Collects hand landmark data from camera using MediaPipe"""
    
    def __init__(self, save_dir='datasets'):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # MediaPipe hand topology
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        # Create save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Data storage
        self.collected_frames = []
        self.frame_count = 0
        
    def get_hand_landmarks(self, frame):
        """
        Extract 21 hand landmarks from frame
        Returns: numpy array of shape (21, 3) or None if no hand detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract 21 landmarks as [x, y, z]
            landmarks = np.zeros((21, 3))
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks[idx] = [landmark.x, landmark.y, landmark.z]
            
            return landmarks, hand_landmarks
        
        return None, None
    
    def calculate_rotation(self, parent_pos, child_pos):
        """Calculate quaternion rotation from parent to child joint"""
        forward = child_pos - parent_pos
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        
        forward = forward / forward_norm
        reference_up = np.array([0.0, 1.0, 0.0])
        
        right = np.cross(reference_up, forward)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            right = np.cross(np.array([1.0, 0.0, 0.0]), forward)
            right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        
        right = right / right_norm
        up = np.cross(forward, right)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([right, up, forward])
        
        # Convert to quaternion
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
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
    
    def calculate_hand_rotations(self, landmarks):
        """Calculate rotation quaternions for all 21 joints"""
        rotations = []
        
        for i in range(21):
            if i == 0:
                # Wrist
                middle_mcp = landmarks[9]
                rotation = self.calculate_rotation(landmarks[0], middle_mcp)
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
                    rotation = self.calculate_rotation(landmarks[parent_idx], landmarks[i])
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            
            rotations.append(rotation)
        
        return rotations
    
    def collect_session(self, duration_seconds=60, display=True):
        """
        Collect hand tracking data for specified duration
        
        Args:
            duration_seconds: How long to collect data
            display: Whether to show video feed
        """
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        
        print(f"Starting data collection for {duration_seconds} seconds")
        print("Move your hand naturally - make various gestures and poses")
        print("Press 'q' to stop early, 's' to save current progress")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get landmarks
            landmarks, hand_landmarks = self.get_hand_landmarks(frame)
            
            if landmarks is not None:
                # Calculate rotations
                rotations = self.calculate_hand_rotations(landmarks)
                
                # Build joints array in VR Module format
                joints = []
                for i in range(21):
                    joint = {
                        'joint_id': i,
                        'position': [float(landmarks[i][0]), float(landmarks[i][1]), float(landmarks[i][2])],
                        'rotation': rotations[i]
                    }
                    joints.append(joint)
                
                # Store data with VR Module format
                data_point = {
                    'Timestamp': int(time.time() * 1000),
                    'frame_number': self.frame_count,
                    'joints': joints
                }
                self.collected_frames.append(data_point)
                self.frame_count += 1
                
                # Draw landmarks on frame
                if display and hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display stats
            elapsed = time.time() - start_time
            remaining = max(0, duration_seconds - elapsed)
            
            cv2.putText(frame, f"Frames: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {int(remaining)}s", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if landmarks is None:
                cv2.putText(frame, "NO HAND DETECTED", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if display:
                cv2.imshow('MediaPipe Hand Tracking', frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping collection early...")
                break
            elif key == ord('s'):
                self.save_dataset()
                print(f"Progress saved: {self.frame_count} frames")
            
            # Check time limit
            if elapsed >= duration_seconds:
                print("Time limit reached")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCollection complete: {self.frame_count} frames collected")
        return self.collected_frames
    
    def save_dataset(self, filename=None):
        """Save collected data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hand_data_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        dataset = {
            'metadata': {
                'total_frames': self.frame_count,
                'collection_date': datetime.now().isoformat(),
                'joint_count': 21,
                'format': 'VR Module Design Concept 2 - joints with position and rotation'
            },
            'frames': self.collected_frames
        }
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to: {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    collector = MediaPipeCollector(save_dir='datasets')
    
    # Collect 60 seconds of data
    data = collector.collect_session(duration_seconds=60, display=True)
    
    # Save to file
    collector.save_dataset()