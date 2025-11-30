"""
MediaPipe Hand Angle Extractor - Snapshot Mode
Captures hand landmarks from webcam and calculates individual joint angles
Press SPACEBAR to capture a snapshot
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from collections import deque


class HandAngleExtractor:
    """Extract individual finger joint angles from MediaPipe hand landmarks"""
    
    def __init__(self, smoothing_window=5):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Smoothing buffers for each joint of each finger
        self.smoothing_window = smoothing_window
        self.angle_buffers = {
            'index': {'mcp': deque(maxlen=smoothing_window), 'pip': deque(maxlen=smoothing_window), 'dip': deque(maxlen=smoothing_window)},
            'middle': {'mcp': deque(maxlen=smoothing_window), 'pip': deque(maxlen=smoothing_window), 'dip': deque(maxlen=smoothing_window)},
            'ring': {'mcp': deque(maxlen=smoothing_window), 'pip': deque(maxlen=smoothing_window), 'dip': deque(maxlen=smoothing_window)},
            'pinky': {'mcp': deque(maxlen=smoothing_window), 'pip': deque(maxlen=smoothing_window), 'dip': deque(maxlen=smoothing_window)}
        }
        
        # Snapshot data
        self.snapshots = []
        self.last_snapshot = None
        self.snapshot_count = 0
        self.current_angles = None
        
        # MediaPipe landmark indices
        self.LANDMARKS = {
            'wrist': 0,
            'thumb': {
                'cmc': 1,      # Carpometacarpal
                'mcp': 2,      # Metacarpophalangeal
                'ip': 3,       # Interphalangeal
                'tip': 4
            },
            'index': {
                'mcp': 5,
                'pip': 6,      # Proximal interphalangeal
                'dip': 7,      # Distal interphalangeal
                'tip': 8
            },
            'middle': {
                'mcp': 9,
                'pip': 10,
                'dip': 11,
                'tip': 12
            },
            'ring': {
                'mcp': 13,
                'pip': 14,
                'dip': 15,
                'tip': 16
            },
            'pinky': {
                'mcp': 17,
                'pip': 18,
                'dip': 19,
                'tip': 20
            }
        }
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3
        
        Args:
            p1, p2, p3: 3D points [x, y, z]
        
        Returns:
            Angle in degrees (0-180)
        """
        # Vectors
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # Normalize
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Angle
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        return np.degrees(angle)
    
    def calculate_bend_angle(self, angle_at_joint):
        """Convert joint angle to bend angle
        
        MediaPipe gives the angle between bones.
        We want the bend angle: 0° = straight, 90° = bent
        """
        # When finger is straight, angle is ~180°
        # When bent, angle is smaller
        bend = 180.0 - angle_at_joint
        return max(0, min(180, bend))
    
    def get_finger_joint_angles(self, landmarks):
        """Extract individual joint angles for each finger (excluding thumb)
        
        Returns:
            dict: {finger_name: {'mcp': angle, 'pip': angle, 'dip': angle}}
        """
        joint_angles = {}
        
        # Helper to get landmark position
        def get_pos(idx):
            lm = landmarks[idx]
            return [lm.x, lm.y, lm.z]
        
        wrist = get_pos(self.LANDMARKS['wrist'])
        
        # For each finger (excluding thumb)
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            finger_landmarks = self.LANDMARKS[finger_name]
            
            # Get all joint positions
            mcp = get_pos(finger_landmarks['mcp'])
            pip = get_pos(finger_landmarks['pip'])
            dip = get_pos(finger_landmarks['dip'])
            tip = get_pos(finger_landmarks['tip'])
            
            # Calculate angle at each joint
            # MCP: angle between wrist->mcp->pip
            mcp_angle = self.calculate_angle(wrist, mcp, pip)
            
            # PIP: angle between mcp->pip->dip
            pip_angle = self.calculate_angle(mcp, pip, dip)
            
            # DIP: angle between pip->dip->tip
            dip_angle = self.calculate_angle(pip, dip, tip)
            
            # Convert to bend angles
            joint_angles[finger_name] = {
                'mcp': self.calculate_bend_angle(mcp_angle),
                'pip': self.calculate_bend_angle(pip_angle),
                'dip': self.calculate_bend_angle(dip_angle)
            }
        
        return joint_angles
    
    def smooth_angles(self, joint_angles):
        """Apply temporal smoothing to joint angles"""
        smoothed = {}
        
        for finger in joint_angles:
            smoothed[finger] = {}
            for joint in joint_angles[finger]:
                self.angle_buffers[finger][joint].append(joint_angles[finger][joint])
                smoothed[finger][joint] = np.mean(self.angle_buffers[finger][joint])
        
        return smoothed
    
    def create_angle_matrix(self, joint_angles):
        """Create a numpy matrix of joint angles
        
        Returns:
            np.ndarray: 4x3 matrix [index, middle, ring, pinky] x [mcp, pip, dip]
        """
        fingers = ['index', 'middle', 'ring', 'pinky']
        joints = ['mcp', 'pip', 'dip']
        
        matrix = np.zeros((4, 3))
        for i, finger in enumerate(fingers):
            for j, joint in enumerate(joints):
                matrix[i, j] = joint_angles[finger][joint]
        
        return matrix
    
    def format_matrix_string(self, matrix):
        """Format matrix as a pretty string"""
        fingers = ['Index ', 'Middle', 'Ring  ', 'Pinky ']
        
        lines = []
        lines.append("        MCP     PIP     DIP")
        lines.append("-" * 35)
        
        for i, finger in enumerate(fingers):
            row = f"{finger}  {matrix[i, 0]:6.1f}° {matrix[i, 1]:6.1f}° {matrix[i, 2]:6.1f}°"
            lines.append(row)
        
        return "\n".join(lines)
    
    def capture_snapshot(self, joint_angles):
        """Capture a snapshot of current joint angles"""
        matrix = self.create_angle_matrix(joint_angles)
        
        snapshot = {
            'snapshot_id': self.snapshot_count,
            'timestamp': time.time(),
            'joint_angles': joint_angles,
            'matrix': matrix.tolist()
        }
        
        self.snapshots.append(snapshot)
        self.last_snapshot = snapshot
        self.snapshot_count += 1
        
        # Print to console
        print(f"\n📸 Snapshot {self.snapshot_count} captured:")
        print(self.format_matrix_string(matrix))
        print()
        
        return snapshot
    
    def draw_angles(self, image, joint_angles, hand_landmarks):
        """Draw angle information on image"""
        h, w, _ = image.shape
        
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Draw angle matrix as table
        y_offset = 30
        cv2.putText(image, "Joint Angles:", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        # Header
        cv2.putText(image, "        MCP   PIP   DIP", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 20
        
        # Data rows
        for finger in ['index', 'middle', 'ring', 'pinky']:
            angles = joint_angles[finger]
            text = f"{finger[:3].upper()}  {angles['mcp']:4.0f}° {angles['pip']:4.0f}° {angles['dip']:4.0f}°"
            cv2.putText(image, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        # Snapshot counter
        cv2.putText(image, f"Snapshots: {self.snapshot_count}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Press SPACEBAR to capture snapshot",
            "Press 'S' to save snapshots",
            "Press 'Q' to quit"
        ]
        y_offset = h - 60
        for instruction in instructions:
            cv2.putText(image, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 18
        
        return image
    
    def save_snapshots(self, filename="mediapipe_snapshots.json"):
        """Save snapshot data to JSON file"""
        if not self.snapshots:
            print("⚠ No snapshots to save")
            return
        
        data = {
            'metadata': {
                'total_snapshots': len(self.snapshots),
                'timestamp': time.time(),
                'smoothing_window': self.smoothing_window
            },
            'snapshots': self.snapshots
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.snapshots)} snapshots to {filename}")
        
        # Print all snapshot matrices
        if len(self.snapshots) > 0:
            print("\n📊 All Snapshots:")
            for snapshot in self.snapshots:
                matrix = np.array(snapshot['matrix'])
                print(f"\nSnapshot {snapshot['snapshot_id']}:")
                print(self.format_matrix_string(matrix))
    
    def run(self):
        """Main loop - snapshot mode"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*60)
        print("MEDIAPIPE HAND ANGLE EXTRACTOR - SNAPSHOT MODE")
        print("="*60)
        print("Controls:")
        print("  SPACEBAR - Capture snapshot")
        print("  S - Save all snapshots to JSON")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue
            
            # Flip for mirror view
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            results = self.hands.process(image_rgb)
            
            # Draw
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate individual joint angles
                    joint_angles = self.get_finger_joint_angles(hand_landmarks.landmark)
                    
                    # Smooth angles
                    smoothed_angles = self.smooth_angles(joint_angles)
                    
                    # Draw on image
                    image = self.draw_angles(image, smoothed_angles, hand_landmarks)
                    
                    # Store for snapshot capture
                    self.current_angles = smoothed_angles
            else:
                # No hand detected
                cv2.putText(image, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.current_angles = None
            
            # Display
            cv2.imshow('MediaPipe Hand Angles - Snapshot Mode', image)
            
            # Keyboard controls
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == 32:  # SPACEBAR
                if self.current_angles is not None:
                    self.capture_snapshot(self.current_angles)
                    # Flash effect
                    flash = image.copy()
                    cv2.rectangle(flash, (0, 0), (image.shape[1], image.shape[0]), 
                                  (255, 255, 255), -1)
                    cv2.imshow('MediaPipe Hand Angles - Snapshot Mode', flash)
                    cv2.waitKey(100)
                else:
                    print("⚠ No hand detected - cannot capture snapshot")
            elif key == ord('s') or key == ord('S'):
                self.save_snapshots()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n✓ Shutdown complete")


def main():
    extractor = HandAngleExtractor(smoothing_window=5)
    extractor.run()


if __name__ == "__main__":
    main()

