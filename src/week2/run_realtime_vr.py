"""
NEURA Glove - Real-Time VR Inference
Runs trained model on live sensor data
Sends hand pose to Unity via UDP
"""

import asyncio
import torch
import numpy as np
from bleak import BleakClient, BleakScanner
import struct
import socket
import json
import time
from collections import deque
from glove_data_types import SensorData, HandPoseOutput
from train_model import LSTMPoseModel


class RealTimeInference:
    """Real-time hand pose estimation for VR"""
    
    def __init__(self, model_path, unity_ip="127.0.0.1", unity_port=5005):
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMPoseModel().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        
        # BLE
        self.ble_client = None
        self.sensor_buffer = deque(maxlen=10)  # 10 timesteps for LSTM
        
        # UDP to Unity
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_ip = unity_ip
        self.unity_port = unity_port
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
    
    async def connect_glove(self):
        """Connect to ESP32 glove"""
        print("Scanning for NEURA_GLOVE...")
        
        devices = await BleakScanner.discover(timeout=10.0)
        glove = None
        
        for device in devices:
            if device.name and "NEURA_GLOVE" in device.name:
                glove = device
                break
        
        if not glove:
            raise Exception("Glove not found!")
        
        print(f"Connecting to {glove.address}...")
        self.ble_client = BleakClient(glove.address)
        await self.ble_client.connect()
        
        await self.ble_client.start_notify(
            "beb5483e-36e1-4688-b7f5-ea07361b26a8",
            self._on_sensor_data
        )
        
        print("✓ Glove connected\n")
        await asyncio.sleep(0.5)
    
    def _on_sensor_data(self, sender, data: bytearray):
        """Handle incoming sensor data"""
        try:
            timestamp_us, = struct.unpack('<Q', data[0:8])
            flex = struct.unpack('<5H', data[8:18])
            accel = struct.unpack('<3f', data[18:30])
            gyro = struct.unpack('<3f', data[30:42])
            quat = struct.unpack('<4f', data[42:58])
            
            sensor = SensorData(
                timestamp=timestamp_us / 1e6,
                flex_sensors=np.array(flex, dtype=np.float32),
                imu_accel=np.array(accel, dtype=np.float32),
                imu_gyro=np.array(gyro, dtype=np.float32),
                imu_quat=np.array(quat, dtype=np.float32)
            )
            
            self.sensor_buffer.append(sensor)
            
        except Exception as e:
            print(f"Parse error: {e}")
    
    def predict_pose(self):
        """Run ML inference on sensor buffer"""
        if len(self.sensor_buffer) < 10:
            return None
        
        # Prepare input sequence
        sequence = []
        for sensor in self.sensor_buffer:
            features = np.concatenate([
                sensor.flex_sensors,
                sensor.imu_accel,
                sensor.imu_gyro,
                sensor.imu_quat
            ])
            sequence.append(features)
        
        sequence = np.array(sequence)  # (10, 15)
        
        # Run model
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 10, 15)
            output = self.model(x)  # (1, 63)
            landmarks = output.cpu().numpy().reshape(21, 3)  # (21, 3)
        
        # Convert to Unity format
        joints = []
        for i in range(21):
            joints.append({
                "joint_id": i,
                "position": landmarks[i].tolist(),
                "rotation": [1, 0, 0, 0]  # Identity quaternion for now
            })
        
        return HandPoseOutput(
            timestamp=time.time(),
            joints=joints,
            confidence=0.95
        )
    
    def send_to_unity(self, pose):
        """Send hand pose to Unity via UDP"""
        try:
            message = pose.to_unity_json().encode('utf-8')
            self.udp_socket.sendto(message, (self.unity_ip, self.unity_port))
        except Exception as e:
            print(f"UDP send error: {e}")
    
    async def run(self):
        """Main inference loop"""
        print("="*60)
        print("REAL-TIME VR MODE")
        print("="*60)
        print(f"Streaming to Unity at {self.unity_ip}:{self.unity_port}")
        print("Press Ctrl+C to stop\n")
        
        self.start_time = time.time()
        
        try:
            while True:
                # Predict pose
                pose = self.predict_pose()
                
                if pose:
                    # Send to Unity
                    self.send_to_unity(pose)
                    
                    self.frame_count += 1
                    
                    # Stats every second
                    if self.frame_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        fps = self.frame_count / elapsed
                        latency = (time.time() - pose.timestamp) * 1000
                        print(f"Frames: {self.frame_count} | FPS: {fps:.1f} | Latency: {latency:.1f}ms")
                
                await asyncio.sleep(0.001)  # Small delay
        
        except KeyboardInterrupt:
            print(f"\n✓ Stopped after {self.frame_count} frames")
        
        finally:
            if self.ble_client and self.ble_client.is_connected:
                await self.ble_client.disconnect()
            self.udp_socket.close()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--unity-ip", type=str, default="127.0.0.1", help="Unity IP address")
    parser.add_argument("--unity-port", type=int, default=5005, help="Unity UDP port")
    
    args = parser.parse_args()
    
    inference = RealTimeInference(args.model, args.unity_ip, args.unity_port)
    
    # Connect to glove
    await inference.connect_glove()
    
    # Run inference
    await inference.run()


if __name__ == "__main__":
    asyncio.run(main())