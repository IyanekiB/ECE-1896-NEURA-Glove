"""
Real-Time Inference for Unity - ENHANCED VERSION
Supports all 21 landmarks or just the 5 primary joints
Receives flex sensor data via BLE, predicts joint rotations, sends to Unity via UDP
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

# Enable all joints (set to False to use only 5 primary joints)
ENABLE_ALL_JOINTS = False  # Change to True for all 21 landmarks


class FlexToRotationInference:
    """Real-time inference from flex sensors to Unity"""
    
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
            print(f"  Loaded config: Input({input_dim}) -> {hidden_dims} -> Output({output_dim})")
        else:
            # Infer from state_dict
            state_dict = checkpoint['model_state_dict']
            linear_keys = sorted(
                [k for k in state_dict.keys() if k.endswith(".weight") and k.startswith("network.")],
                key=lambda k: int(k.split('.')[1])
            )
            input_dim = state_dict[linear_keys[0]].shape[1]
            layer_dims = [state_dict[k].shape[0] for k in linear_keys]
            hidden_dims = layer_dims[:-1]
            output_dim = layer_dims[-1]
            print(f"  Inferred config: Input({input_dim}) -> {hidden_dims} -> Output({output_dim})")
        
        # Determine joint mode
        if output_dim == 10:
            self.joint_mode = "5_joints"  # 5 joints × 2 axes
            print("  Mode: 5 PRIMARY JOINTS (thumb, index, middle, ring, pinky)")
        elif output_dim == 42:
            self.joint_mode = "21_joints"  # 21 joints × 2 axes
            print("  Mode: ALL 21 JOINTS (full hand)")
            ENABLE_ALL_JOINTS = True
        else:
            raise ValueError(f"Unexpected output dimension: {output_dim}")
        
        # Recreate model
        from train_model import FlexToRotationModel
        self.model = FlexToRotationModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("[OK] Model loaded successfully")
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = (UNITY_IP, UNITY_PORT)
        print(f"[OK] UDP socket created for {UNITY_IP}:{UNITY_PORT}")
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
    
    def voltage_to_angle(self, voltage):
        """Convert flex voltage to bend angle (0-90 degrees)"""
        voltage = np.clip(voltage, FLEX_MIN_VOLTAGE, FLEX_MAX_VOLTAGE)
        normalized = (voltage - FLEX_MIN_VOLTAGE) / (FLEX_MAX_VOLTAGE - FLEX_MIN_VOLTAGE)
        angle = 90.0 * (1.0 - normalized)
        return float(angle)

    
    def parse_ble_data(self, data_string):
        """Parse BLE data from ESP32"""
        try:
            values = [float(x) for x in data_string.split(',')]
            if len(values) != 15:
                return None
            
            # Extract flex voltages and convert to angles
            flex_angles = [
                self.voltage_to_angle(values[0]),  # Thumb
                self.voltage_to_angle(values[1]),  # Index
                self.voltage_to_angle(values[2]),  # Middle
                self.voltage_to_angle(values[3]),  # Ring
                self.voltage_to_angle(values[4])   # Pinky
            ]

            imu_quat = [float(values[5]), float(values[6]), float(values[7]), float(values[8])]
            
            return np.array(flex_angles), imu_quat
        
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def predict_rotations(self, flex_angles):
        """Predict joint rotations from flex angles"""
        flex_scaled = self.input_scaler.transform([flex_angles])
        flex_tensor = torch.FloatTensor(flex_scaled)
        
        with torch.no_grad():
            output_scaled = self.model(flex_tensor).numpy()
        
        rotations = self.output_scaler.inverse_transform(output_scaled)[0]
        return rotations
    
    def build_unity_packet(self, rotations, imu_quat):
        """Build Unity packet with proper rotation axis
        
        FIXED: Using X-axis rotation for finger curl (not Y-axis)
        Right-hand to left-hand mirroring applied
        """
        
        # Convert rotation angle to quaternion around X-axis (finger curl)
        def angle_to_quat_x(angle_deg):
            """Convert X-axis rotation angle to quaternion"""
            angle_rad = np.radians(angle_deg)
            half_angle = angle_rad / 2.0
            return [np.sin(half_angle), 0.0, 0.0, np.cos(half_angle)]
        
        if self.joint_mode == "5_joints":
            # 5 primary joints mode (10 values)
            # Mirror X rotations (negate), keep Y rotations same
            thumb_x, thumb_y = -rotations[0], rotations[1]
            index_x, index_y = -rotations[2], rotations[3]
            middle_x, middle_y = -rotations[4], rotations[5]
            ring_x, ring_y = -rotations[6], rotations[7]
            pinky_x, pinky_y = -rotations[8], rotations[9]
            
            # Use Y rotation (the bend angle) for X-axis quaternion
            packet = {
                "timestamp": time.time(),
                "hand": "left",
                "wrist": {
                    "position": [0, 0, 0],
                    "rotation": imu_quat
                },
                "thumb": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": self.euler_to_quaternion(21.194, 43.526, -69.284)

                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "intermediate": {  # Joint 3
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    }
                },
                "index": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "proximal": {  # Joint 6
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    }
                },
                "middle": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "proximal": {  # Joint 10
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    }
                },
                "ring": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "proximal": {  # Joint 14
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    }
                },
                "pinky": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "proximal": {  # Joint 18
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0, 1]
                    }
                }
            }
        
        else:  # 21_joints mode (42 values)
            # Full hand with all rotations
            # rotations = [joint0_x, joint0_y, joint1_x, joint1_y, ..., joint20_x, joint20_y]
            
            def get_rot(idx):
                """Get mirrored rotations for a joint"""
                x_rot = -rotations[idx * 2]      # Mirror X
                y_rot = rotations[idx * 2 + 1]   # Keep Y
                return angle_to_quat_x(y_rot)    # Use Y for curl
            
            packet = {
                "timestamp": time.time(),
                "hand": "left",
                "wrist": {
                    "position": [0, 0, 0],
                    "rotation": get_rot(0)  # Joint 0
                },
                "thumb": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(1)  # Joint 1
                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(2)  # Joint 2
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(3)  # Joint 3
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(4)  # Joint 4
                    }
                },
                "index": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(5)  # Joint 5
                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(6)  # Joint 6
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(7)  # Joint 7
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(8)  # Joint 8
                    }
                },
                "middle": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(9)  # Joint 9
                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(10)  # Joint 10
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(11)  # Joint 11
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(12)  # Joint 12
                    }
                },
                "ring": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(13)  # Joint 13
                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(14)  # Joint 14
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(15)  # Joint 15
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(16)  # Joint 16
                    }
                },
                "pinky": {
                    "metacarpal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(17)  # Joint 17
                    },
                    "proximal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(18)  # Joint 18
                    },
                    "intermediate": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(19)  # Joint 19
                    },
                    "distal": {
                        "position": [0, 0, 0],
                        "rotation": get_rot(20)  # Joint 20
                    }
                }
            }
        
        return packet

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
        
        rotations = self.predict_rotations(flex_angles)
        packet = self.build_unity_packet(rotations, imu_quat)
        self.send_to_unity(packet)
        
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"Frame {self.frame_count} | FPS: {fps:.1f}")
    
    async def run(self):
        """Main inference loop"""
        print(f"\n{'='*60}")
        print("REAL-TIME INFERENCE")
        print(f"{'='*60}")
        print(f"Streaming to Unity at {UNITY_IP}:{UNITY_PORT}")
        
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
            print("[OK] Subscribed to notifications")
            
            print("\n>>> STREAMING TO UNITY")
            print("Press Ctrl+C to stop\n")
            
            self.start_time = time.time()
            
            try:
                while True:
                    await asyncio.sleep(0.1)
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
        print("  python realtime_inference_enhanced.py <model_path>")
        print("\nExample:")
        print("  python realtime_inference_enhanced.py models/flex_to_rotation_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inference = FlexToRotationInference(model_path)
    await inference.run()


if __name__ == "__main__":
    asyncio.run(main())