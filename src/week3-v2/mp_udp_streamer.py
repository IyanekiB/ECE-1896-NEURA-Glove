import cv2
import mediapipe as mp
import numpy as np
import json
import socket
import time
from FrameConstructor import FrameConstructor


class HandDataStreamer:
    """Streams hand landmark data to Unity VR via UDP"""
    
    def __init__(self, unity_ip='127.0.0.1', unity_port=5555, mirror_image=True, save_training_data=False):
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
        self.mirror_image = mirror_image

        # MediaPipe hand topology (parent joint for each joint)
        # Used for calculating rotations
        self.finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]

        print(f"Streaming to Unity at {unity_ip}:{unity_port} (mirror_x={self.mirror_image})")

        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        # Logging
        self.log_interval_seconds = 0.5
        self._next_frame_log_time = self.start_time + self.log_interval_seconds

        # AI Training Data Collection
        self.save_training_data = save_training_data
        if self.save_training_data:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self.training_data_file = f"training_data_{timestamp_str}.json"
            self.training_data_samples = []
            print(f"Training data will be saved to: {self.training_data_file}")
    
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
    
    def remap_landmarks_to_unity(self, landmarks):
        """
        Convert MediaPipe coordinates (x right, y down, z toward camera) to Unity (x right, y up, z forward).
        Also center landmarks at wrist to improve numerical stability for rotations.
        """
        # Flip Y (down->up) and Z (toward camera -> forward)
        remapped = np.copy(landmarks)
        remapped[:, 1] = -remapped[:, 1]
        remapped[:, 2] = -remapped[:, 2]

        # Optional mirror around camera vertical axis to counter mirrored input feeds
        if self.mirror_image:
            remapped[:, 0] = -remapped[:, 0]

        # Center at wrist to work in local hand space
        wrist = remapped[0].copy()
        remapped = remapped - wrist
        return remapped


    def calculate_palm_coordinate_system(self, landmarks):
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


    # def calculate_rotation_from_vectors(self, v1, v2, angle_scale=1.0):
    #     """
    #     Calculate quaternion rotation from one vector to another.
        
    #     Args:
    #         v1: Starting vector (normalized)
    #         v2: Target vector (normalized)
    #         angle_scale: Optional scaling factor for the rotation angle
        
    #     Returns: [x, y, z, w] quaternion
    #     """
    #     axis = np.cross(v1, v2)
    #     axis_len = np.linalg.norm(axis)
        
    #     if axis_len > 1e-6:
    #         axis = axis / axis_len
    #         angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    #         angle = angle * angle_scale 
            
    #         half_angle = angle / 2.0
    #         s = np.sin(half_angle)
    #         c = np.cos(half_angle)
            
    #         return [
    #             float(axis[0] * s),
    #             float(axis[1] * s),
    #             float(axis[2] * s),
    #             float(c)
    #         ]
    #     else:
    #         # Vectors are parallel - no rotation needed
    #         return [0.0, 0.0, 0.0, 1.0]

    def calculate_rotation_from_vectors(self, v1, v2, angle_scale=1.0):
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

    def project_vector_onto_plane(self, v, plane_normal):
        """
        Project a vector onto a plane defined by its normal.

        Args:
            v: Vector to project
            plane_normal: Normal vector of the plane (must be normalized)

        Returns: Projected vector
        """
        # Remove component parallel to normal
        parallel_component = np.dot(v, plane_normal) * plane_normal
        projected = v - parallel_component
        return projected

    def euler_to_quaternion(self, euler_x, euler_y, euler_z):
        """
        Convert Unity Euler angles (in degrees, XYZ order) to quaternion.

        Args:
            euler_x: Rotation around X axis in degrees
            euler_y: Rotation around Y axis in degrees
            euler_z: Rotation around Z axis in degrees

        Returns: [x, y, z, w] quaternion
        """
        # Convert degrees to radians
        x = np.radians(euler_x)
        y = np.radians(euler_y)
        z = np.radians(euler_z)

        # Unity uses ZXY rotation order
        # Calculate half angles
        cx = np.cos(x * 0.5)
        sx = np.sin(x * 0.5)
        cy = np.cos(y * 0.5)
        sy = np.sin(y * 0.5)
        cz = np.cos(z * 0.5)
        sz = np.sin(z * 0.5)

        # ZXY order quaternion multiplication
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz

        return [float(qx), float(qy), float(qz), float(qw)]

    def calculate_constrained_rotation(self, v1, v2, hinge_axis):
        """
        Calculate quaternion rotation from v1 to v2, constrained to rotate around a single axis.
        This creates hinge-joint behavior - only bending, no side-to-side wagging.

        Args:
            v1: Starting vector (normalized)
            v2: Target vector (normalized)
            hinge_axis: Axis to rotate around (normalized) - e.g., palm_right for fingers

        Returns: [x, y, z, w] quaternion representing rotation only around hinge_axis
        """
        # Project both vectors onto the plane perpendicular to hinge axis
        v1_proj = self.project_vector_onto_plane(v1, hinge_axis)
        v2_proj = self.project_vector_onto_plane(v2, hinge_axis)

        # Normalize projected vectors
        v1_len = np.linalg.norm(v1_proj)
        v2_len = np.linalg.norm(v2_proj)

        if v1_len < 1e-6 or v2_len < 1e-6:
            # Vectors are parallel to hinge axis - no rotation in the plane
            return [0.0, 0.0, 0.0, 1.0]

        v1_proj = v1_proj / v1_len
        v2_proj = v2_proj / v2_len

        # Calculate angle between projected vectors
        dot_product = np.clip(np.dot(v1_proj, v2_proj), -1.0, 1.0)
        angle = np.arccos(dot_product)

        # Determine sign of rotation using cross product
        # If cross product points in same direction as hinge axis, angle is positive
        cross = np.cross(v1_proj, v2_proj)
        if np.dot(cross, hinge_axis) < 0:
            angle = -angle

        # Create quaternion from axis-angle
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        c = np.cos(half_angle)

        return [
            float(hinge_axis[0] * s),
            float(hinge_axis[1] * s),
            float(hinge_axis[2] * s),
            float(c)
        ]


    def calculate_thumb_cmc_rotation(self, landmarks, palm_forward, palm_right):
        """
        Calculate rotation for thumb CMC (metacarpal base) joint.
        This joint needs special handling as it rotates relative to the palm plane.
        
        Args:
            landmarks: All hand landmarks
            palm_forward: Forward direction of palm
            palm_right: Right direction of palm
        
        Returns: [x, y, z, w] quaternion
        """
        thumb_cmc = landmarks[1]  # CMC joint
        thumb_mcp = landmarks[2]  # MCP joint
        
        # Direction from CMC to MCP
        thumb_bone = thumb_mcp - thumb_cmc
        thumb_bone = thumb_bone / (np.linalg.norm(thumb_bone) + 1e-10)
        
        # Ideal thumb direction: mix of palm forward and palm right
        # Thumb typically rests at about 45 degrees from palm
        ideal_thumb_dir = palm_forward * 0.7 + palm_right * 0.7
        ideal_thumb_dir = ideal_thumb_dir / (np.linalg.norm(ideal_thumb_dir) + 1e-10)
        
        # Calculate rotation from ideal to actual thumb position
        return self.calculate_rotation_from_vectors(ideal_thumb_dir, thumb_bone)


    def calculate_standard_joint_rotation(self, landmarks, parent_idx, current_idx, child_idx, is_thumb=False):
        """
        Calculate rotation for a standard finger joint.
        Measures the bend angle between parent-current and current-child segments.
        
        Args:
            landmarks: All hand landmarks
            parent_idx: Index of parent joint
            current_idx: Index of current joint
            child_idx: Index of child joint
            is_thumb: Whether this is a thumb joint (affects angle scaling)
        
        Returns: [x, y, z, w] quaternion
        """
        # Vector from parent to current joint
        v1 = landmarks[current_idx] - landmarks[parent_idx]
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        
        # Vector from current to child joint
        v2 = landmarks[child_idx] - landmarks[current_idx]
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Apply angle correction for natural hand pose
        # Fingers naturally curl, so reduce extension angles slightly
        angle_scale = 1.0 if is_thumb else 0.9
        
        return self.calculate_rotation_from_vectors(v1, v2, angle_scale)


    # def calculate_hand_rotations(self, landmarks):
    #     """
    #     Calculate rotations for all hand joints.
    #     Returns list of quaternions [x, y, z, w] for each joint.
    #     """
    #     rotations = []
    #     wrist = landmarks[0]

    #     # Calculate wrist rotation
    #     wrist_rotation, wrist_inverse = self.calculate_wrist_rotation(landmarks)
    #     rotations.append([0.0, 0.0, 0.0, 1.0])  # Wrist gets identity quaternion

    #     # Transform all landmarks to wrist-local space
    #     local_landmarks = np.zeros_like(landmarks)
    #     for i in range(len(landmarks)):
    #         world_pos = landmarks[i] - wrist
    #         local_landmarks[i] = wrist_inverse @ world_pos
        
    #     # Calculate palm coordinate system for thumb reference
    #     palm_forward, palm_right, palm_up = self.calculate_palm_coordinate_system(local_landmarks)
        
    #     palm_rotation_matrix = np.column_stack([palm_right, palm_up, palm_forward])
    #     palm_to_world = palm_rotation_matrix
    #     world_to_palm = palm_rotation_matrix.T

    #     # Process each finger
    #     for finger_idx, chain in enumerate(self.finger_chains):
    #         is_thumb = (finger_idx == 0)
            
    #         for i in range(1, len(chain)):
    #             current_idx = chain[i]
    #             parent_idx = chain[i-1]
                
    #             # Check if this joint has a child (not a fingertip)
    #             if i + 1 < len(chain):
    #                 child_idx = chain[i+1]

    #                 parent_pos = local_landmarks[parent_idx]
    #                 current_pos = local_landmarks[current_idx]
    #                 child_pos = local_landmarks[child_idx]
                    
    #                 # SPECIAL CASE: Thumb CMC (first joint after wrist)
    #                 if is_thumb and i == 1:
    #                     # For thumb metacarpal, measure against palm right direction
    #                     bone_dir = current_pos - parent_pos
    #                     bone_dir = bone_dir / (np.linalg.norm(bone_dir) + 1e-10)
                        
    #                     # Expected thumb direction: angled between forward and right
    #                     expected_dir = (palm_forward * 0.5 + palm_right * 0.866)  # ~30° angle
    #                     expected_dir = expected_dir / (np.linalg.norm(expected_dir) + 1e-10)
                        
    #                     rotation = self.calculate_rotation_from_vectors(expected_dir, bone_dir, 1.0)
    #                     rotations.append(rotation)
    #                     continue
                    
    #                 # Standard joint calculation in palm space
    #                 v1 = current_pos - parent_pos
    #                 v2 = child_pos - current_pos
                    
    #                 v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    #                 v2 = v2 / (np.linalg.norm(v2) + 1e-10)
                    
    #                 # For thumb, be more sensitive to rotation
    #                 angle_scale = 1.0 if is_thumb else 0.9
                    
    #                 rotation = self.calculate_rotation_from_vectors(v1, v2, angle_scale)
    #                 rotations.append(rotation)

    #             else:
    #                 # Fingertip - no rotation
    #                 rotations.append([0.0, 0.0, 0.0, 1.0])
        
    #     return rotations

    def quaternion_to_angle(self, quat):
        """
        Extract rotation angle (in degrees) from a quaternion.

        Args:
            quat: Quaternion [x, y, z, w]

        Returns: Rotation angle in degrees
        """
        # angle = 2 * arccos(w), where w is the scalar part
        w = quat[3]
        angle_rad = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        return np.degrees(angle_rad)

    def extract_training_angles(self, rotations):
        """
        Extract bend angles from rotation quaternions for AI training.
        Returns a structured dict with angles for each finger joint.

        Args:
            rotations: List of 21 quaternion rotations

        Returns: Dict with finger bend angles
        """
        angles = {
            "wrist": 0.0,  # Wrist is identity
            "thumb": {
                "metacarpal": self.quaternion_to_angle(rotations[1]),
                "proximal": self.quaternion_to_angle(rotations[2]),
                "distal": self.quaternion_to_angle(rotations[3]),
                "tip": 0.0
            },
            "index": {
                "metacarpal": self.quaternion_to_angle(rotations[5]),
                "proximal": self.quaternion_to_angle(rotations[6]),
                "intermediate": self.quaternion_to_angle(rotations[7]),
                "distal": self.quaternion_to_angle(rotations[8])
            },
            "middle": {
                "metacarpal": self.quaternion_to_angle(rotations[9]),
                "proximal": self.quaternion_to_angle(rotations[10]),
                "intermediate": self.quaternion_to_angle(rotations[11]),
                "distal": self.quaternion_to_angle(rotations[12])
            },
            "ring": {
                "metacarpal": self.quaternion_to_angle(rotations[13]),
                "proximal": self.quaternion_to_angle(rotations[14]),
                "intermediate": self.quaternion_to_angle(rotations[15]),
                "distal": self.quaternion_to_angle(rotations[16])
            },
            "pinky": {
                "metacarpal": self.quaternion_to_angle(rotations[17]),
                "proximal": self.quaternion_to_angle(rotations[18]),
                "intermediate": self.quaternion_to_angle(rotations[19]),
                "distal": self.quaternion_to_angle(rotations[20])
            }
        }
        return angles

    def calculate_hand_rotations(self, landmarks):
        """
        Calculate finger rotations relative to the palm orientation.
        Uses single-axis constraints for hinge-joint behavior (no finger wagging).
        Thumb metacarpal is fixed; thumb joints use single-axis bending.
        This ensures fingers always bend toward the palm regardless of hand orientation.
        Returns list of quaternions [x, y, z, w] for each joint.
        """
        rotations = []

        # Wrist gets identity rotation (we're ignoring wrist orientation)
        rotations.append([0.0, 0.0, 0.0, 1.0])

        # Calculate palm coordinate system
        palm_forward, palm_right, palm_up = self.calculate_palm_coordinate_system(landmarks)

        # Create transformation matrix from world space to palm-local space
        # Column vectors are the palm's local axes expressed in world coordinates
        palm_basis = np.column_stack([palm_right, palm_up, palm_forward])
        world_to_palm = palm_basis.T  # Inverse of orthogonal matrix is its transpose

        # Transform all landmarks to palm-local space
        wrist = landmarks[0]
        local_landmarks = np.zeros_like(landmarks)
        for i in range(len(landmarks)):
            world_pos = landmarks[i] - wrist  # Position relative to wrist
            local_landmarks[i] = world_to_palm @ world_pos  # Transform to palm space

        # In palm-local space, define hinge axes:
        # - palm_right = X-axis (left-right across knuckles)
        # - palm_up = Y-axis (perpendicular to palm surface)
        # - palm_forward = Z-axis (toward fingers)

        # Fingers rotate around palm_right axis (like door hinges)
        hinge_axis_fingers = np.array([1.0, 0.0, 0.0])  # palm_right in local space

        # Thumb: Sticks out perpendicular to side of hand, rotates around custom axis
        # This allows thumb to curl inward toward palm (not toward back of hand)
        thumb_axis_raw = np.array([1.0, -0.5, 0.0])
        hinge_axis_thumb = thumb_axis_raw / np.linalg.norm(thumb_axis_raw)  # Normalize

        # Process each finger in palm-local space
        for finger_idx, chain in enumerate(self.finger_chains):
            is_thumb = (finger_idx == 0)
            hinge_axis = hinge_axis_thumb if is_thumb else hinge_axis_fingers

            for i in range(1, len(chain)):
                current_idx = chain[i]
                parent_idx = chain[i-1]

                # SPECIAL: Thumb metacarpal (CMC joint) - set fixed orientation from Unity rest pose
                if is_thumb and i == 1:
                    # Use the Unity rest rotation: (21.194, 43.526, -69.284)
                    thumb_metacarpal_rot = self.euler_to_quaternion(21.194, 43.526, -69.284)
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
                    rotation = self.calculate_constrained_rotation(v1, v2, hinge_axis)

                    rotations.append(rotation)
                else:
                    # Fingertip - no rotation
                    rotations.append([0.0, 0.0, 0.0, 1.0])

        return rotations

    def calculate_wrist_rotation(self, landmarks):
        """
        ########## Only yaw rotation is calculated for now ##########
        Calculate wrist rotation from landmarks
        Returns: [x, y, z, w] quaternion
        """

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        knuckle_avg = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0
        palm_forward = knuckle_avg - wrist
        palm_forward = palm_forward / (np.linalg.norm(palm_forward) + 1e-10)        
        palm_forward_flat = np.array([palm_forward[0], 0.0, palm_forward[2]])
        palm_forward_flat = palm_forward_flat / (np.linalg.norm(palm_forward_flat) + 1e-10)

        # Calculate yaw angle (rotation around Y axis)
        # Calculate angle using atan2 for full 360° range
        yaw = -1*np.arctan2(palm_forward_flat[0], palm_forward_flat[2])
        
        # Convert yaw angle to quaternion (rotation around Y axis)
        half_yaw = yaw / 2.0
        wrist_quat = [
            0.0,                    # x
            float(np.sin(half_yaw)), # y (rotation axis)
            0.0,                    # z
            float(np.cos(half_yaw))  # w
        ]
        
        # ===== TRANSFORM LANDMARKS TO WRIST-LOCAL SPACE =====
        # Build simple rotation matrix for yaw only
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Rotation matrix around Y axis
        rotation_matrix = np.array([
            [cos_yaw,  0.0, sin_yaw],
            [0.0,      1.0, 0.0    ],
            [-sin_yaw, 0.0, cos_yaw]
        ])
        
        wrist_inverse = rotation_matrix.T

        return [0.0, 0.0, 0.0, 1.0], wrist_inverse
    
    def create_data_packet(self, landmarks):
        """
        Create a frame matching HandDataGenerator.py format using calculated rotations.
        Also extracts training data if enabled.
        """
        # Remap to Unity coordinate space before rotation calculation
        unity_landmarks = self.remap_landmarks_to_unity(landmarks)
        rotations = self.calculate_hand_rotations(unity_landmarks)

        # Extract training angles if enabled
        if self.save_training_data:
            training_angles = self.extract_training_angles(rotations)
            training_sample = {
                "timestamp": time.time(),
                "frame_number": self.frame_count,
                "raw_landmarks": landmarks.tolist(),  # Original MediaPipe landmarks
                "joint_angles": training_angles  # Processed single-axis bend angles
            }
            self.training_data_samples.append(training_sample)

        frame = FrameConstructor.build_frame_from_rotations(rotations, hand="left")
        return json.dumps(frame)
    
    def send_to_unity(self, packet_json):
        """Send JSON packet to Unity via UDP"""
        try:
            self.sock.sendto(packet_json.encode('utf-8'), self.unity_address)
        except Exception as e:
            print(f"Error sending to Unity: {e}")

    def save_training_data_to_file(self):
        """Save collected training data to JSON file"""
        if not self.save_training_data or len(self.training_data_samples) == 0:
            return

        output = {
            "metadata": {
                "total_samples": len(self.training_data_samples),
                "start_time": self.start_time,
                "end_time": time.time(),
                "duration_seconds": time.time() - self.start_time,
                "description": "Hand tracking training data with single-axis joint bend angles"
            },
            "samples": self.training_data_samples
        }

        try:
            with open(self.training_data_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n✓ Training data saved: {self.training_data_file}")
            print(f"  Total samples: {len(self.training_data_samples)}")
        except Exception as e:
            print(f"\n✗ Error saving training data: {e}")
    
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

        # Save training data if enabled
        if self.save_training_data:
            self.save_training_data_to_file()


if __name__ == "__main__":
    # Configuration
    SAVE_TRAINING_DATA = False  # Set to True to save joint angle training data to JSON

    # Create streamer (change IP if Unity is on different machine)
    streamer = HandDataStreamer(
        unity_ip='127.0.0.1',
        unity_port=5555,
        save_training_data=SAVE_TRAINING_DATA
    )

    # Start streaming
    streamer.stream(show_video=True)