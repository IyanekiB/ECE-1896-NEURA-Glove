"""
Real-Time Inference V4 FINAL - Per-finger bend ratios + correct templates
Fixes:
- Removed model inversion (was incorrect)
- Per-finger hierarchical ratios (index/middle bend more)
- Correct pose templates based on actual ML predictions
- IMU correction
- Performance optimizations
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

# Per-finger biomechanical ratios
# Index and middle need more aggressive bending (higher multipliers)
FINGER_BEND_RATIOS = {
    'thumb': {
        'metacarpal': 0.3,
        'proximal': 1.0,
        'intermediate': 1.2,  # Thumb bends moderately
        'distal': 0.6
    },
    'index': {
        'metacarpal': 2.0,
        'proximal': 1.0,
        'intermediate': 2.0,  # INCREASED: Index bends a lot
        'distal': 1.0         # INCREASED: Tip curls more
    },
    'middle': {
        'metacarpal': 0.8,
        'proximal': 1.0,
        'intermediate': 2.0,  # INCREASED: Middle bends a lot
        'distal': 1.0         # INCREASED: Tip curls more
    },
    'ring': {
        'metacarpal': 0.8,
        'proximal': 1.0,
        'intermediate': 1.5,  # Ring bends normally
        'distal': 0.7
    },
    'pinky': {
        'metacarpal': 0.8,
        'proximal': 1.0,
        'intermediate': 1.5,  # Pinky bends normally
        'distal': 0.7
    }
}

# CORRECT templates based on ML predictions WITHOUT inversion
# From your diagnostic images:
# flat_hand: [4.4, 3.6, 3.3, 5.2, 7.8]
# fist: [38.3, 11.1, 8.5, 14.6, 12.2]
# grab: [5.1, 34.5, 57.0, 69.7, 60.8]
POSE_TEMPLATES = {
    'flat_hand': [4, 4, 3, 5, 8],       # Nearly straight (low angles)
    'fist': [38, 11, 9, 15, 12],        # All curled (moderate angles)
    'grab': [5, 35, 57, 70, 61],        # Thumb straight, fingers vary
}


class PoseClassifier:
    """Pose classifier with relaxed threshold"""
    
    def __init__(self, templates=POSE_TEMPLATES, threshold=30):
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
    """Real-time inference with per-finger ratios"""
    
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
        
        # Pre-allocate arrays
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
        """Correct IMU quaternion for mounting orientation"""
        qx, qy, qz, qw = imu_quat
        
        # 180° rotation around X-axis
        corrected_qx = qw
        corrected_qy = -qz
        corrected_qz = qy
        corrected_qw = qx
        
        return [corrected_qx, corrected_qy, corrected_qz, corrected_qw]
    
    def voltage_to_angle(self, voltage):
        """Convert flex voltage to bend angle"""
        voltage = np.clip(voltage, FLEX_MIN_VOLTAGE, FLEX_MAX_VOLTAGE)
        normalized = (voltage - FLEX_MIN_VOLTAGE) / (FLEX_MAX_VOLTAGE - FLEX_MIN_VOLTAGE)
        angle = 90.0 * (1.0 - normalized)
        return angle
    
    def parse_ble_data(self, data_string):
        """Parse BLE data from ESP32"""
        try:
            values = data_string.split(',')
            if len(values) != 15:
                return None, None
            
            # Parse flex voltages
            for i in range(5):
                self._flex_angles_buffer[i] = self.voltage_to_angle(float(values[i]))
            
            # Extract IMU quaternion
            imu_quat = [float(values[6]), float(values[7]), float(values[8]), float(values[5])]
            
            return self._flex_angles_buffer, imu_quat
        
        except:
            return None, None
    
    def predict_rotations(self, flex_angles):
        """Predict proximal joint rotations"""
        flex_scaled = self.input_scaler.transform([flex_angles])
        flex_tensor = torch.FloatTensor(flex_scaled)
        
        with torch.no_grad():
            output_scaled = self.model(flex_tensor).numpy()
        
        rotations = self.output_scaler.inverse_transform(output_scaled)[0]
        
        # NO INVERSION - model output is correct as-is
        return rotations
    
    def compute_hierarchical_joints(self, proximal_angle, finger_name):
        """Compute all joint angles from proximal using per-finger ratios"""
        ratios = FINGER_BEND_RATIOS[finger_name]
        
        return {
            'metacarpal': proximal_angle * ratios['metacarpal'],
            'proximal': proximal_angle * ratios['proximal'],
            'intermediate': proximal_angle * ratios['intermediate'],
            'distal': proximal_angle * ratios['distal']
        }
    
    def build_unity_packet(self, proximal_rotations, imu_quat):
        """Build Unity packet with per-finger ratios"""
        
        # Convert rotation angle to quaternion around X-axis
        def angle_to_quat_x(angle_deg):
            angle_rad = np.radians(angle_deg)
            half_angle = angle_rad * 0.5
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            return [sin_half, 0.0, 0.0, cos_half]
        
        # Extract proximal Y rotations
        thumb_y = proximal_rotations[1]
        index_y = proximal_rotations[3]
        middle_y = proximal_rotations[5]
        ring_y = proximal_rotations[7]
        pinky_y = proximal_rotations[9]
        
        # Compute hierarchical angles with per-finger ratios
        thumb_joints = self.compute_hierarchical_joints(thumb_y, 'thumb')
        index_joints = self.compute_hierarchical_joints(index_y, 'index')
        middle_joints = self.compute_hierarchical_joints(middle_y, 'middle')
        ring_joints = self.compute_hierarchical_joints(ring_y, 'ring')
        pinky_joints = self.compute_hierarchical_joints(pinky_y, 'pinky')
        
        # Fixed thumb metacarpal
        thumb_metacarpal_rot = self.euler_to_quaternion(21.194, 43.526, -69.284)
        
        # Correct IMU orientation
        corrected_wrist_quat = self.correct_imu_quaternion(imu_quat)
        
        # Build packet
        packet = {
            "timestamp": time.time(),
            "hand": "left",
            "wrist": {
                "position": [0, 0, 0],
                "rotation": corrected_wrist_quat
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
        """Handle BLE notifications"""
        data_string = data.decode('utf-8')
        
        flex_angles, imu_quat = self.parse_ble_data(data_string)
        if flex_angles is None:
            return
        
        # Predict rotations
        rotations = self.predict_rotations(flex_angles)
        
        # Classify pose (every 5 frames)
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
        print("REAL-TIME INFERENCE V4 FINAL")
        print(f"{'='*60}")
        print(f"Streaming to Unity at {UNITY_IP}:{UNITY_PORT}")
        print(f"\nPer-finger bend ratios:")
        print(f"  Index/Middle intermediate: 2.0× (aggressive bending)")
        print(f"  Thumb/Ring/Pinky intermediate: 1.2-1.5× (normal)")
        print(f"\nPose templates: {list(POSE_TEMPLATES.keys())}")
        
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
            print("Press Ctrl+C to stop\n")
            
            self.start_time = time.time()
            
            try:
                while True:
                    await asyncio.sleep(0.05)
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
        print("  python realtime_inference_v4_final.py <model_path>")
        print("\nExample:")
        print("  python realtime_inference_v4_final.py models/flex_to_rotation_model.pth")
        print("\nV4 Final Features:")
        print("  ✓ Per-finger bend ratios (index/middle bend more)")
        print("  ✓ No model inversion (was incorrect)")
        print("  ✓ Correct pose templates")
        print("  ✓ Fixed IMU orientation")
        print("  ✓ Performance optimizations")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inference = FlexToRotationInference(model_path)
    await inference.run()


if __name__ == "__main__":
    asyncio.run(main())