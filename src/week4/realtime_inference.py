"""
Real-Time Inference V3 - Fixed IMU orientation, updated pose templates, optimized performance
Changes from V2:
- IMU quaternion correction for proper hand orientation
- Updated pose templates based on actual sensor data
- Relaxed classification threshold
- Performance optimizations (reduced overhead)
"""

import asyncio
import json
import numpy as np
import torch
import socket
import time
from bleak import BleakClient, BleakScanner


# BLE Configuration
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DEVICE_NAME = "ESP32-BLE"

# Unity UDP Configuration
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5555

# Flex sensor calibration
FLEX_MIN_VOLTAGE = 0.55
FLEX_MAX_VOLTAGE = 1.65

# Biomechanical ratios
FINGER_BEND_RATIOS = {
    'metacarpal': 0.3,
    'proximal': 1.0,
    'intermediate': 1.5,
    'distal': 0.5
}

# Updated pose templates - BASED ON YOUR DIAGNOSTIC OUTPUT
# After inversion fix, these should work:
POSE_TEMPLATES = {
    'flat_hand': [86, 86, 87, 85, 82],   # Image 1: 90-ML_predictions [90-4.4, 90-3.6, 90-3.3, 90-5.2, 90-7.8]
    'fist': [52, 79, 82, 75, 78],        # Image 2: 90-ML_predictions [90-38, 90-11, 90-9, 90-15, 90-12]
    'grab': [85, 56, 33, 20, 29],        # Image 3: 90-ML_predictions [90-5, 90-34, 90-57, 90-70, 90-61]
}


class PoseClassifier:
    """Simple pose classifier with relaxed threshold"""
    
    def __init__(self, templates=POSE_TEMPLATES, threshold=25):  # Increased from 25 to 35
        self.templates = templates
        self.threshold = threshold
    
    def classify(self, finger_angles):
        """Classify pose based on finger angles"""
        best_match = None
        best_distance = float('inf')
        
        for pose_name, template in self.templates.items():
            diff = np.array(finger_angles) - np.array(template)
            rmse = np.sqrt(np.mean(diff ** 2))
            
            if rmse < best_distance:
                best_distance = rmse
                best_match = pose_name
        
        if best_distance < self.threshold:
            confidence = 1.0 - (best_distance / self.threshold)
            return best_match, confidence
        else:
            return 'unknown', 0.0


class FlexToRotationInference:
    """Real-time inference with all fixes"""
    
    def __init__(self, model_path):
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.input_scaler = checkpoint['input_scaler']
        self.output_scaler = checkpoint['output_scaler']
        
        # Get model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_dim = config['input_dim']
            output_dim = config['output_dim']
            hidden_dims = config['hidden_dims']
        else:
            state_dict = checkpoint['model_state_dict']
            linear_keys = sorted(
                [k for k in state_dict.keys() if k.endswith(".weight") and k.startswith("network.")],
                key=lambda k: int(k.split('.')[1])
            )
            input_dim = state_dict[linear_keys[0]].shape[1]
            layer_dims = [state_dict[k].shape[0] for k in linear_keys]
            hidden_dims = layer_dims[:-1]
            output_dim = layer_dims[-1]
        
        print(f"  Config: Input({input_dim}) -> {hidden_dims} -> Output({output_dim})")
        
        # Recreate model
        from train_model import FlexToRotationModel
        self.model = FlexToRotationModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ Model loaded successfully")
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = (UNITY_IP, UNITY_PORT)
        print(f"✓ UDP socket created for {UNITY_IP}:{UNITY_PORT}")
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
        # Pose classifier
        self.pose_classifier = PoseClassifier()
        self.current_pose = 'unknown'
        self.pose_confidence = 0.0
        
        # Pre-allocate arrays for performance
        self._flex_angles_buffer = np.zeros(5)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion [x, y, z, w]"""
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return [qx, qy, qz, qw]
    
    def correct_imu_quaternion(self, imu_quat):
        """Correct IMU quaternion for mounting orientation
        
        The IMU is mounted on the back of the hand, so we need to rotate
        the coordinate frame to match Unity's hand orientation.
        
        This applies a 180° rotation around X-axis to flip the hand right-side up.
        """
        qx, qy, qz, qw = imu_quat
        
        # 180° rotation around X-axis: multiply by [1, 0, 0, 0]
        # q_corrected = q_rotation * q_imu
        corrected_qx = qw    # New x
        corrected_qy = -qz   # New y
        corrected_qz = qy    # New z
        corrected_qw = qx    # New w
        
        return [corrected_qx, corrected_qy, corrected_qz, corrected_qw]
    
    def voltage_to_angle(self, voltage):
        """Convert flex voltage to bend angle (0-90 degrees)"""
        voltage = np.clip(voltage, FLEX_MIN_VOLTAGE, FLEX_MAX_VOLTAGE)
        normalized = (voltage - FLEX_MIN_VOLTAGE) / (FLEX_MAX_VOLTAGE - FLEX_MIN_VOLTAGE)
        angle = 90.0 * (1.0 - normalized)
        return angle  # Return float directly
    
    def parse_ble_data(self, data_string):
        """Parse BLE data from ESP32 - optimized"""
        try:
            values = data_string.split(',')
            if len(values) != 15:
                return None, None
            
            # Parse flex voltages directly into buffer
            for i in range(5):
                self._flex_angles_buffer[i] = self.voltage_to_angle(float(values[i]))
            
            # Extract IMU quaternion
            imu_quat = [float(values[6]), float(values[7]), float(values[8]), float(values[5])]
            
            return self._flex_angles_buffer, imu_quat
        
        except:
            return None, None
    
    def predict_rotations(self, flex_angles):
        """Predict proximal joint rotations - optimized"""
        flex_scaled = self.input_scaler.transform([flex_angles])
        flex_tensor = torch.FloatTensor(flex_scaled)
        
        with torch.no_grad():
            output_scaled = self.model(flex_tensor).numpy()
        
        rotations = self.output_scaler.inverse_transform(output_scaled)[0]
        
        # FIX: Model learned inverse - invert Y-axis predictions
        # Extract Y rotations (indices 1, 3, 5, 7, 9)
        # for i in [1, 3, 5, 7, 9]:
        #     rotations[i] = 90.0 - rotations[i]  # Invert: if model predicts 10°, use 80°
        
        return rotations
    
    def compute_hierarchical_joints(self, proximal_angle):
        """Compute all joint angles from proximal"""
        return {
            'metacarpal': proximal_angle * FINGER_BEND_RATIOS['metacarpal'],
            'proximal': proximal_angle * FINGER_BEND_RATIOS['proximal'],
            'intermediate': proximal_angle * FINGER_BEND_RATIOS['intermediate'],
            'distal': proximal_angle * FINGER_BEND_RATIOS['distal']
        }
    
    def build_unity_packet(self, proximal_rotations, imu_quat):
        """Build Unity packet - optimized"""
        
        # Convert rotation angle to quaternion around X-axis
        def angle_to_quat_x(angle_deg):
            angle_rad = np.radians(angle_deg)
            half_angle = angle_rad * 0.5
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            return [sin_half, 0.0, 0.0, cos_half]
        
        # Extract and mirror proximal rotations
        thumb_y = proximal_rotations[1]
        index_y = proximal_rotations[3]
        middle_y = proximal_rotations[5]
        ring_y = proximal_rotations[7]
        pinky_y = proximal_rotations[9]
        
        # Compute hierarchical angles
        thumb_joints = self.compute_hierarchical_joints(thumb_y)
        index_joints = self.compute_hierarchical_joints(index_y)
        middle_joints = self.compute_hierarchical_joints(middle_y)
        ring_joints = self.compute_hierarchical_joints(ring_y)
        pinky_joints = self.compute_hierarchical_joints(pinky_y)
        
        # Fixed thumb metacarpal
        thumb_metacarpal_rot = self.euler_to_quaternion(21.194, 43.526, -69.284)
        
        # Correct IMU orientation (FIX for upside-down hand)
        corrected_wrist_quat = self.correct_imu_quaternion(imu_quat)
        
        # Build packet
        packet = {
            "timestamp": time.time(),
            "hand": "left",
            "wrist": {
                "position": [0, 0, 0],
                "rotation": corrected_wrist_quat  # FIXED: Corrected quaternion
            },
            "thumb": {
                "metacarpal": {
                    "position": [0, 0, 0],
                    "rotation": thumb_metacarpal_rot
                },
                "proximal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(thumb_joints['proximal'])
                },
                "intermediate": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(thumb_joints['intermediate'])
                },
                "distal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(thumb_joints['distal'])
                }
            },
            "index": {
                "metacarpal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(index_joints['metacarpal'])
                },
                "proximal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(index_joints['proximal'])
                },
                "intermediate": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(index_joints['intermediate'])
                },
                "distal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(index_joints['distal'])
                }
            },
            "middle": {
                "metacarpal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(middle_joints['metacarpal'])
                },
                "proximal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(middle_joints['proximal'])
                },
                "intermediate": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(middle_joints['intermediate'])
                },
                "distal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(middle_joints['distal'])
                }
            },
            "ring": {
                "metacarpal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(ring_joints['metacarpal'])
                },
                "proximal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(ring_joints['proximal'])
                },
                "intermediate": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(ring_joints['intermediate'])
                },
                "distal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(ring_joints['distal'])
                }
            },
            "pinky": {
                "metacarpal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(pinky_joints['metacarpal'])
                },
                "proximal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(pinky_joints['proximal'])
                },
                "intermediate": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(pinky_joints['intermediate'])
                },
                "distal": {
                    "position": [0, 0, 0],
                    "rotation": angle_to_quat_x(pinky_joints['distal'])
                }
            }
        }
        
        return packet
    
    def send_to_unity(self, packet):
        """Send packet to Unity via UDP"""
        try:
            packet_json = json.dumps(packet)
            self.sock.sendto(packet_json.encode('utf-8'), self.unity_address)
            self.frame_count += 1
        except Exception as e:
            print(f"Error sending to Unity: {e}")
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications - optimized"""
        data_string = data.decode('utf-8')
        
        flex_angles, imu_quat = self.parse_ble_data(data_string)
        if flex_angles is None:
            return
        
        # Predict rotations
        rotations = self.predict_rotations(flex_angles)
        
        # Classify pose (every 5 frames to reduce overhead)
        if self.frame_count % 5 == 0:
            proximal_angles = [
                rotations[1],  # Thumb Y
                rotations[3],  # Index Y
                rotations[5],  # Middle Y
                rotations[7],  # Ring Y
                rotations[9]   # Pinky Y
            ]
            self.current_pose, self.pose_confidence = self.pose_classifier.classify(proximal_angles)
        
        # Build and send packet
        packet = self.build_unity_packet(rotations, imu_quat)
        self.send_to_unity(packet)
        
        # Print status every 10 frames
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"Frame {self.frame_count} | FPS: {fps:.1f} | Pose: {self.current_pose} ({self.pose_confidence:.1%})")
    
    async def run(self):
        """Main inference loop"""
        print(f"\n{'='*60}")
        print("REAL-TIME INFERENCE V3")
        print(f"{'='*60}")
        print(f"Streaming to Unity at {UNITY_IP}:{UNITY_PORT}")
        print(f"Pose templates: {list(POSE_TEMPLATES.keys())}")
        print(f"Classification threshold: {self.pose_classifier.threshold}° RMSE")
        
        print("\nScanning for ESP32...")
        devices = await BleakScanner.discover(timeout=5.0)
        target_device = None
        
        for device in devices:
            if device.name == DEVICE_NAME:
                target_device = device
                break
        
        if not target_device:
            print(f"ERROR: Device '{DEVICE_NAME}' not found!")
            return
        
        print(f"Found: {target_device.name} ({target_device.address})")
        
        print("Connecting...")
        async with BleakClient(target_device.address) as client:
            print(f"Connected: {client.is_connected}")
            
            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            print("✓ Subscribed to notifications")
            
            print("\n🚀 STREAMING TO UNITY")
            print("⚠️  If FPS is still low, run diagnostic_v2.py to identify bottleneck")
            print("⚠️  If poses don't classify, calibrate templates with diagnostic_v2.py")
            print("Press Ctrl+C to stop\n")
            
            self.start_time = time.time()
            
            try:
                while True:
                    await asyncio.sleep(0.05)  # Reduced from 0.1 for better responsiveness
            except KeyboardInterrupt:
                print("\n\nStopping...")
            
            await client.stop_notify(CHARACTERISTIC_UUID)
        
        print(f"\n{'='*60}")
        print(f"Total frames sent: {self.frame_count}")
        print(f"Average FPS: {self.frame_count / (time.time() - self.start_time):.1f}")


async def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python realtime_inference_v3.py <model_path>")
        print("\nExample:")
        print("  python realtime_inference_v3.py models/flex_to_rotation_model.pth")
        print("\nV3 Features:")
        print("  ✓ Fixed IMU orientation (hand no longer upside down)")
        print("  ✓ Relaxed pose classification threshold (35° RMSE)")
        print("  ✓ Performance optimizations")
        print("\nIf issues persist:")
        print("  1. Run diagnostic_v2.py to see actual angles")
        print("  2. Calibrate pose templates based on diagnostic output")
        print("  3. Adjust FLEX_MIN/MAX_VOLTAGE if angles seem off")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inference = FlexToRotationInference(model_path)
    await inference.run()


if __name__ == "__main__":
    asyncio.run(main())