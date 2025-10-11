import cv2
import mediapipe as mp
import numpy as np
import json
import socket
import time


class HandDataStreamer:
    """Streams hand landmark data to Unity VR via UDP"""
    
    def __init__(self, unity_ip='127.0.0.1', unity_port=5555):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = (unity_ip, unity_port)
        
        # MediaPipe hand topology (parent joint for each joint)
        # Used for calculating rotations
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        print(f"Streaming to Unity at {unity_ip}:{unity_port}")
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_hand_landmarks(self, frame):
        """
        Extract 21 hand landmarks from frame
        Returns: numpy array of shape (21, 3) or None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def calculate_rotation(self, parent_pos, child_pos):
        """
        Calculate quaternion rotation from parent to child joint
        Returns: [x, y, z, w] quaternion
        """
        # Calculate forward direction (parent to child)
        forward = child_pos - parent_pos
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            # Joints too close, return identity quaternion
            return [0.0, 0.0, 0.0, 1.0]
        
        forward = forward / forward_norm
        
        # Reference up vector
        reference_up = np.array([0.0, 1.0, 0.0])
        
        # Calculate right vector (perpendicular to forward and up)
        right = np.cross(reference_up, forward)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # Forward parallel to up, use alternative
            right = np.cross(np.array([1.0, 0.0, 0.0]), forward)
            right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # Still parallel, return identity
            return [0.0, 0.0, 0.0, 1.0]
        
        right = right / right_norm
        
        # Calculate actual up vector
        up = np.cross(forward, right)
        
        # Create rotation matrix from basis vectors
        rotation_matrix = np.column_stack([right, up, forward])
        
        # Convert rotation matrix to quaternion
        # Using standard conversion formula
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
        """
        Calculate rotation quaternions for all 21 joints
        Returns: list of 21 quaternions [x, y, z, w]
        """
        rotations = []
        
        for i in range(21):
            if i == 0:
                # Wrist: calculate based on palm direction
                middle_mcp = landmarks[9]
                rotation = self.calculate_rotation(landmarks[0], middle_mcp)
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
                    rotation = self.calculate_rotation(landmarks[parent_idx], landmarks[i])
                else:
                    rotation = [0.0, 0.0, 0.0, 1.0]
            
            rotations.append(rotation)
        
        return rotations
    
    def create_data_packet(self, landmarks):
        """
        Create data packet for Unity VR Module
        Format matches VR Module Design Concept 2:
        {
            "Timestamp": 12345678,
            "joints": [
                {
                    "joint_id": 0,
                    "position": [x, y, z],
                    "rotation": [x, y, z, w]
                },
                ...
            ]
        }
        """
        # Calculate rotations for all joints
        rotations = self.calculate_hand_rotations(landmarks)
        
        # Build joints array
        joints = []
        for i in range(21):
            joint = {
                "joint_id": i,
                "position": [float(landmarks[i][0]), float(landmarks[i][1]), float(landmarks[i][2])],
                "rotation": rotations[i]
            }
            joints.append(joint)
        
        # Create packet with exact format from VR Module spec
        packet = {
            "Timestamp": int(time.time() * 1000),  # Milliseconds
            "joints": joints
        }
        
        return json.dumps(packet)
    
    def send_to_unity(self, packet_json):
        """Send JSON packet to Unity via UDP"""
        try:
            self.sock.sendto(packet_json.encode('utf-8'), self.unity_address)
        except Exception as e:
            print(f"Error sending to Unity: {e}")
    
    def stream(self, show_video=True):
        """
        Main streaming loop - captures camera and sends to Unity
        """
        cap = cv2.VideoCapture(0)
        
        print("Starting hand tracking stream...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get hand landmarks
            landmarks, hand_landmarks = self.get_hand_landmarks(frame)
            
            if landmarks is not None:
                # Create and send packet
                packet = self.create_data_packet(landmarks)
                self.send_to_unity(packet)
                
                self.frame_count += 1
                
                # Draw on frame
                if show_video and hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display stats
            if show_video:
                fps = self.frame_count / (time.time() - self.start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {self.frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Streaming to {self.unity_address[0]}:{self.unity_address[1]}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if landmarks is None:
                    cv2.putText(frame, "NO HAND DETECTED", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Hand Tracking - Streaming to Unity', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.sock.close()
        
        print(f"\nStream ended. Total frames: {self.frame_count}")


if __name__ == "__main__":
    # Create streamer (change IP if Unity is on different machine)
    streamer = HandDataStreamer(unity_ip='127.0.0.1', unity_port=5555)
    
    # Start streaming
    streamer.stream(show_video=True)