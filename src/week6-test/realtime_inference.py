"""
Real-Time Inference - FINAL FIX
Critical Fixes:
1. Proper Ctrl+C handling to save predictions_log.json
2. Live IMU wrist orientation (not hardcoded)
3. Pinky angle clamping
4. Performance optimization
"""

import asyncio
import json
import numpy as np
import torch
import socket
import time
import signal
import sys
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

# OPTIMIZED: More aggressive bend ratios for realistic fist
FINGER_BEND_RATIOS = {
    'thumb': {
        'metacarpal': 0.4,   # Increased from 0.3
        'proximal': 1.2,     # Increased from 1.0
        'intermediate': 1.5, # Increased from 1.2
        'distal': 0.8        # Increased from 0.6
    },
    'index': {
        'metacarpal': 0.7,   # Increased from 0.5
        'proximal': 1.3,     # Increased from 1.0
        'intermediate': 2.2, # Increased from 1.8
        'distal': 1.2        # Increased from 0.9
    },
    'middle': {
        'metacarpal': 0.7,   # Increased from 0.5
        'proximal': 1.3,     # Increased from 1.0
        'intermediate': 2.2, # Increased from 1.8
        'distal': 1.2        # Increased from 0.9
    },
    'ring': {
        'metacarpal': 0.7,   # Increased from 0.5
        'proximal': 1.3,     # Increased from 1.0
        'intermediate': 1.8, # Increased from 1.5
        'distal': 1.0        # Increased from 0.7
    },
    'pinky': {
        'metacarpal': 0.7,   # Increased from 0.5
        'proximal': 1.3,     # Increased from 1.0
        'intermediate': 1.8, # Increased from 1.5
        'distal': 1.0        # Increased from 0.7
    }
}

# Fist confidence-based scaling for aggressive curl
FIST_CONFIDENCE_THRESHOLD = 0.8  # 80% confidence minimum to trigger scaling
FIST_RATIO_MULTIPLIER = 1.8     # 80% more aggressive curl (reaches 1.8x at 100% confidence)
FIST_METACARPAL_BOOST = 2.0     # Extra multiplier for 4-finger metacarpals during fist

# Pose templates
POSE_TEMPLATES = {
    'flat_hand': [3.5, 7.5, 5.5, 6.0, 5.3],
    'fist': [48.9, 34.8, 32.5, 34.0, 30.2],
    'grab': [34.4, 33.8, 37.5, 38.6, 29.7],
}


class KalmanFilter:
    """1D Kalman filter for smoothing joint angles"""
    
    def __init__(self, process_variance=0.01, measurement_variance=0.1, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.error_covariance = 1.0
        
    def update(self, measurement):
        """Update filter with new measurement"""
        # Prediction
        predicted_estimate = self.estimate
        predicted_error_covariance = self.error_covariance + self.process_variance
        
        # Update
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self.measurement_variance)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance
        
        return self.estimate


class MultiKalmanFilter:
    """Kalman filter bank for multiple joint angles"""
    
    def __init__(self, num_joints=10, process_variance=0.01, measurement_variance=0.1):
        self.filters = [
            KalmanFilter(process_variance, measurement_variance) 
            for _ in range(num_joints)
        ]
        
    def update(self, measurements):
        """Update all filters with new measurements"""
        return np.array([
            self.filters[i].update(measurements[i]) 
            for i in range(len(measurements))
        ])


class PoseClassifier:
    """Pose classifier with configurable threshold"""
    
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
    """Real-time inference - FINAL FIX"""
    
    def __init__(self, model_path, enable_kalman=True, 
                 process_variance=0.005, measurement_variance=0.08):
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
        
        # Kalman filtering
        self.enable_kalman = enable_kalman
        if self.enable_kalman:
            self.kalman = MultiKalmanFilter(
                num_joints=output_dim,
                process_variance=process_variance,
                measurement_variance=measurement_variance
            )
            print(f"✓ Kalman filtering enabled (Q={process_variance}, R={measurement_variance})")
        else:
            print("⚠ Kalman filtering disabled")
        
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
        
        # CRITICAL FIX: Logging for evaluation
        self.predictions_log = []
        self.log_file = "predictions_log.json"
        
        # CRITICAL FIX: Flag for graceful shutdown
        self.shutdown_requested = False
    
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
    
    def voltage_to_angle(self, voltage):
        """Convert flex voltage to bend angle"""
        voltage = np.clip(voltage, FLEX_MIN_VOLTAGE, FLEX_MAX_VOLTAGE)
        normalized = (voltage - FLEX_MIN_VOLTAGE) / (FLEX_MAX_VOLTAGE - FLEX_MIN_VOLTAGE)
        angle = 90.0 * (1.0 - normalized)
        return angle
    
    def parse_ble_data(self, data_string):
        """Parse BLE data from ESP32
        Format: flex1,flex2,flex3,flex4,flex5,qw,qx,qy,qz (9 values)
        """
        try:
            values = data_string.split(',')
            if len(values) != 9:
                return None, None

            # Parse flex voltages (indices 0-4)
            self._flex_angles_buffer[0] = self.voltage_to_angle(float(values[4]))  # flex1 -> Thumb
            self._flex_angles_buffer[1] = self.voltage_to_angle(float(values[0]))  # flex2 -> Index
            self._flex_angles_buffer[2] = self.voltage_to_angle(float(values[3]))  # flex3 -> Middle
            self._flex_angles_buffer[3] = self.voltage_to_angle(float(values[2]))  # flex4 -> Ring
            self._flex_angles_buffer[4] = self.voltage_to_angle(float(values[1]))  # flex5 -> Pinky

            # Extract LIVE IMU quaternion from BLE stream
            # Format from ESP32 (indices 5-8): qw,qx,qy,qz
            imu_quat = [
                float(values[6]),  # qx
                float(values[7]),  # qy
                float(values[8]),  # qz
                float(values[5])   # qw
            ]

            return self._flex_angles_buffer.copy(), imu_quat

        except Exception as e:
            return None, None
    
    def predict_rotations(self, flex_angles):
        """Predict joint rotations with optional Kalman filtering"""
        # ML model prediction
        flex_scaled = self.input_scaler.transform([flex_angles])
        flex_tensor = torch.FloatTensor(flex_scaled)
        
        with torch.no_grad():
            output_scaled = self.model(flex_tensor).numpy()
        
        rotations_raw = self.output_scaler.inverse_transform(output_scaled)[0]
        
        # FIXED: Clamp pinky angle to prevent negative values
        rotations_raw[9] = max(0, rotations_raw[9])  # Pinky Y-axis
        
        # Apply Kalman filtering
        if self.enable_kalman:
            rotations = self.kalman.update(rotations_raw)
        else:
            rotations = rotations_raw
        
        return rotations
    
    def distribute_rotations(self, proximal_angle, ratios, current_pose='unknown', confidence=0.0, finger_name='unknown'):
        """Distribute proximal angle across finger joints

        Args:
            proximal_angle: Base angle from ML model
            ratios: Bend ratios for the finger
            current_pose: Current detected pose ('fist', 'flat_hand', 'grab', or 'unknown')
            confidence: Confidence score of the detected pose (0.0 to 1.0)
            finger_name: Name of the finger ('thumb', 'index', 'middle', 'ring', 'pinky')
        """
        # Ensure angle is positive
        proximal_angle = max(0, proximal_angle)

        # Apply confidence-based scaling for fist pose
        applied_ratios = ratios.copy()
        if current_pose == 'fist' and confidence >= FIST_CONFIDENCE_THRESHOLD:
            # Gradual scaling: at 0.8 confidence = 1.0x, at 1.0 confidence = 1.8x
            scale_factor = 1.0 + (confidence - FIST_CONFIDENCE_THRESHOLD) * FIST_RATIO_MULTIPLIER
            applied_ratios = {
                key: ratio * scale_factor
                for key, ratio in ratios.items()
            }

            # Extra boost for 4-finger metacarpals during fist (not thumb)
            if finger_name in ['index', 'middle', 'ring', 'pinky']:
                # Gradual scaling from 1.0x to 2.0x as confidence increases from 0 to 1.0
                metacarpal_scale = 1.0 + confidence * (FIST_METACARPAL_BOOST - 1.0)
                applied_ratios['metacarpal'] *= metacarpal_scale

        return {
            'metacarpal': proximal_angle * applied_ratios['metacarpal'],
            'proximal': proximal_angle * applied_ratios['proximal'],
            'intermediate': proximal_angle * applied_ratios['intermediate'],
            'distal': proximal_angle * applied_ratios['distal']
        }
    
    def build_unity_packet(self, rotations, imu_quat):
        """Build Unity UDP packet from rotations
        
        CRITICAL: Uses LIVE IMU quaternion data for wrist orientation
        """
        # Extract proximal Y-axis rotations
        thumb_y = rotations[1]
        index_y = rotations[3]
        middle_y = rotations[5]
        ring_y = rotations[7]
        pinky_y = rotations[9]
        
        # Distribute rotations across joints with confidence-based scaling for fist
        thumb_joints = self.distribute_rotations(thumb_y, FINGER_BEND_RATIOS['thumb'],
                                                self.current_pose, self.pose_confidence, 'thumb')
        index_joints = self.distribute_rotations(index_y, FINGER_BEND_RATIOS['index'],
                                                self.current_pose, self.pose_confidence, 'index')
        middle_joints = self.distribute_rotations(middle_y, FINGER_BEND_RATIOS['middle'],
                                                 self.current_pose, self.pose_confidence, 'middle')
        ring_joints = self.distribute_rotations(ring_y, FINGER_BEND_RATIOS['ring'],
                                               self.current_pose, self.pose_confidence, 'ring')
        pinky_joints = self.distribute_rotations(pinky_y, FINGER_BEND_RATIOS['pinky'],
                                                self.current_pose, self.pose_confidence, 'pinky')
        
        # Convert to quaternions (X-axis rotation for finger curl)
        def angle_to_quat_x(angle):
            theta = np.radians(max(0, angle))
            return [np.sin(theta/2), 0, 0, np.cos(theta/2)]
        
        # Fixed thumb metacarpal
        thumb_metacarpal_rot = self.euler_to_quaternion(21.194, 43.526, -69.284)
        
        # CRITICAL FIX: Use LIVE IMU quaternion for wrist (not hardcoded!)
        # The imu_quat comes directly from BLE stream each frame
        wrist_quat = imu_quat  # [qx, qy, qz, qw] format
        
        # Build packet
        packet = {
            "timestamp": time.time(),
            "hand": "left",
            "wrist": {
                "position": [0, 0, 0],
                "rotation": wrist_quat  # LIVE IMU DATA
            },
            "thumb": {
                "metacarpal": {"position": [0, 0, 0], "rotation": thumb_metacarpal_rot},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['distal'])}
            },
            "index": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['distal'])}
            },
            "middle": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['distal'])}
            },
            "ring": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['distal'])}
            },
            "pinky": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['distal'])}
            }
        }
        
        return packet
    
    def send_to_unity(self, packet):
        """Send packet to Unity via UDP - OPTIMIZED"""
        try:
            packet_json = json.dumps(packet, separators=(',', ':'))
            self.sock.sendto(packet_json.encode('utf-8'), self.unity_address)
            self.frame_count += 1
        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"Error sending to Unity: {e}")
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications"""
        if self.shutdown_requested:
            return
        
        data_string = data.decode('utf-8')
        
        flex_angles, imu_quat = self.parse_ble_data(data_string)
        if flex_angles is None:
            return
        
        # Predict rotations (with Kalman filtering if enabled)
        rotations = self.predict_rotations(flex_angles)
        
        # Classify pose (every 10 frames to reduce CPU load)
        if self.frame_count % 10 == 0:
            proximal_angles = [
                rotations[1],  # Thumb Y
                rotations[3],  # Index Y
                rotations[5],  # Middle Y
                rotations[7],  # Ring Y
                rotations[9]   # Pinky Y
            ]
            self.current_pose, self.pose_confidence = self.pose_classifier.classify(proximal_angles)
            
            # Log prediction for evaluation
            self.predictions_log.append({
                'timestamp': time.time(),
                'frame': self.frame_count,
                'pose': self.current_pose,
                'confidence': float(self.pose_confidence),
                'angles': [float(a) for a in proximal_angles],
                'imu_quat': [float(q) for q in imu_quat]  # Also log IMU data
            })
        
        # Build and send packet
        packet = self.build_unity_packet(rotations, imu_quat)
        self.send_to_unity(packet)
        
        # Print status every 30 frames
        if self.frame_count % 10 == 0 and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            kalman_status = "ON" if self.enable_kalman else "OFF"
            print(f"Frame {self.frame_count} | FPS: {fps:.1f} | Kalman: {kalman_status} | "
                  f"Pose: {self.current_pose} ({self.pose_confidence:.1%})")
    
    def save_predictions_log(self):
        """CRITICAL FIX: Save predictions log for evaluation"""
        if not self.predictions_log:
            print("No predictions to save (ran too short?)")
            return False
        
        output = {
            'metadata': {
                'total_predictions': len(self.predictions_log),
                'duration': time.time() - self.start_time if self.start_time else 0,
                'kalman_enabled': self.enable_kalman,
                'process_variance': self.kalman.filters[0].process_variance if self.enable_kalman else None,
                'measurement_variance': self.kalman.filters[0].measurement_variance if self.enable_kalman else None
            },
            'predictions': self.predictions_log
        }
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nPredictions log saved: {self.log_file}")
            print(f"  Total predictions: {len(self.predictions_log)}")
            return True
        except Exception as e:
            print(f"\n✗ Error saving predictions log: {e}")
            return False
    
    async def run(self):
        """Main inference loop with PROPER shutdown handling"""
        print(f"\n{'='*60}")
        print("REAL-TIME INFERENCE - FINAL FIX")
        print(f"{'='*60}")
        print(f"Streaming to Unity at {UNITY_IP}:{UNITY_PORT}")
        print(f"Kalman filtering: {'ENABLED' if self.enable_kalman else 'DISABLED'}")
        print(f"Pose templates: {list(POSE_TEMPLATES.keys())}")
        print(f"\nFixes applied:")
        print(f"  Live IMU wrist orientation (not hardcoded)")
        print(f"  Proper Ctrl+C handling (saves log)")
        print(f"  Pinky curl direction")
        print(f"  Performance optimization")
        
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
        
        # CRITICAL FIX: Proper exception handling
        try:
            async with BleakClient(target_device.address) as client:
                print(f"Connected: {client.is_connected}")
                
                await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
                print("✓ Subscribed to notifications")
                
                print("\nSTREAMING TO UNITY")
                print("Using LIVE IMU data for wrist orientation")
                print("Press Ctrl+C to stop and save log\n")
                
                self.start_time = time.time()
                
                # Main loop
                while not self.shutdown_requested:
                    await asyncio.sleep(0.02)
                
                # Clean shutdown
                await client.stop_notify(CHARACTERISTIC_UUID)
                
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected...")
            self.shutdown_requested = True
        except Exception as e:
            print(f"\n✗ Error during streaming: {e}")
            self.shutdown_requested = True
        finally:
            # CRITICAL: Always save log, even on error
            print(f"\n{'='*60}")
            print(f"SHUTDOWN")
            print(f"{'='*60}")
            
            if self.frame_count > 0:
                elapsed = time.time() - self.start_time
                print(f"Total frames sent: {self.frame_count}")
                print(f"Duration: {elapsed:.1f}s")
                print(f"Average FPS: {self.frame_count / elapsed:.1f}")
            
            # Save predictions log
            print("\nSaving predictions log...")
            self.save_predictions_log()


# CRITICAL FIX: Global reference for signal handler
inference_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    if inference_instance:
        print("\n\nInterrupt signal received, shutting down gracefully...")
        inference_instance.shutdown_requested = True


async def main():
    global inference_instance
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python realtime_inference_final_fix.py <model_path> [--no-kalman] [--process-var Q] [--measurement-var R]")
        print("\nExamples:")
        print("  python realtime_inference_final_fix.py models/flex_to_rotation_model.pth")
        print("  python realtime_inference_final_fix.py models/flex_to_rotation_model.pth --no-kalman")
        print("\nFINAL FIXES:")
        print("  Live IMU wrist orientation (reads from BLE stream)")
        print("  Proper Ctrl+C handling (always saves predictions_log.json)")
        print("  Pinky finger curl direction")
        print("  Performance optimization")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Parse optional arguments
    enable_kalman = '--no-kalman' not in sys.argv
    
    process_var = 0.005
    measurement_var = 0.08
    
    if '--process-var' in sys.argv:
        idx = sys.argv.index('--process-var')
        process_var = float(sys.argv[idx + 1])
    
    if '--measurement-var' in sys.argv:
        idx = sys.argv.index('--measurement-var')
        measurement_var = float(sys.argv[idx + 1])
    
    # Create inference instance
    inference_instance = FlexToRotationInference(
        model_path,
        enable_kalman=enable_kalman,
        process_variance=process_var,
        measurement_variance=measurement_var
    )
    
    # CRITICAL FIX: Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run inference
    await inference_instance.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Graceful shutdown complete")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()