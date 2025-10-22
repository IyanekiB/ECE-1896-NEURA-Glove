"""
NEURA GLOVE - Real-Time Inference Engine
Performs camera-free hand tracking using trained LSTM + Kalman model

Usage:
    python inference_engine.py --model models/best_model.pth --duration 60
"""

import asyncio
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Dict

import torch
import torch.nn as nn

from bleak import BleakClient, BleakScanner
from train_model import TrainingConfig

torch.serialization.add_safe_globals([TrainingConfig])

# ============================================================================
# KALMAN FILTER
# ============================================================================

class KalmanFilter:
    """Adaptive Kalman Filter for temporal smoothing"""
    
    def __init__(self, state_size: int, process_noise: float, measurement_noise: float):
        self.state_size = state_size
        
        # State estimate and covariance
        self.x = np.zeros(state_size)  # State
        self.P = np.eye(state_size)    # Covariance
        
        # Process and measurement noise
        self.Q = np.eye(state_size) * process_noise
        self.R = np.eye(state_size) * measurement_noise
        
        # State transition (identity - constant model)
        self.F = np.eye(state_size)
        self.H = np.eye(state_size)
    
    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """Update step with new measurement"""
        # Innovation
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.state_size) - K @ self.H) @ self.P
        
        return self.x


# ============================================================================
# LSTM MODEL (Same architecture as training)
# ============================================================================

class TemporalSmoothingLSTM(nn.Module):
    """LSTM network for temporal sequence modeling"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.OUTPUT_SIZE)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Real-time inference with Kalman filtering"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = TemporalSmoothingLSTM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Sequence length: {self.config.SEQUENCE_LENGTH}")
        print(f"  Input size: {self.config.INPUT_SIZE}")
        print(f"  Output size: {self.config.OUTPUT_SIZE}")
        
        # Kalman filter for smoothing
        self.kalman = KalmanFilter(
            state_size=self.config.OUTPUT_SIZE,
            process_noise=self.config.PROCESS_NOISE,
            measurement_noise=self.config.MEASUREMENT_NOISE
        )
        
        # Sequence buffer
        self.sequence_buffer = deque(maxlen=self.config.SEQUENCE_LENGTH)
        
        # BLE configuration
        self.ble_service_uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.ble_char_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.ble_device_name = "ESP32-BLE"
        
        # State
        self.client = None
        self.is_running = False
        self.frame_count = 0
        self.last_prediction_time = 0
        
        # Statistics
        self.latencies = []
    
    def parse_sensor_data(self, data: bytearray) -> Optional[np.ndarray]:
        """Parse BLE notification data"""
        try:
            data_str = data.decode('utf-8').strip()
            values = [float(x) for x in data_str.split(',')]
            
            if len(values) != 15:
                return None
            
            return np.array(values, dtype=np.float32)
        except Exception as e:
            print(f"✗ Parse error: {e}")
            return None
    
    def predict(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Predict joint positions/rotations from sensor data
        
        Args:
            sensor_data: (15,) array with sensor values
            
        Returns:
            (147,) array with joint positions and rotations
        """
        # Add to sequence buffer
        self.sequence_buffer.append(sensor_data)
        
        # Need full sequence for prediction
        if len(self.sequence_buffer) < self.config.SEQUENCE_LENGTH:
            return self.kalman.x  # Return previous state
        
        # Prepare input (1, sequence_length, input_size)
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # LSTM prediction
        with torch.no_grad():
            raw_prediction = self.model(sequence).cpu().numpy()[0]
        
        # Kalman filtering for smoothness
        self.kalman.predict()
        smoothed_prediction = self.kalman.update(raw_prediction)
        
        return smoothed_prediction
    
    def prediction_to_json(self, prediction: np.ndarray) -> Dict:
        """Convert prediction to JSON format for Unity"""
        joints = []
        for i in range(21):  # 21 joints
            idx = i * 7
            joints.append({
                'joint_id': i,
                'position': prediction[idx:idx+3].tolist(),
                'rotation': prediction[idx+3:idx+7].tolist()
            })
        
        return {
            'timestamp': int(time.time() * 1000),
            'frame_number': self.frame_count,
            'joints': joints
        }
    
    async def find_device(self) -> str:
        """Scan for ESP32 BLE device"""
        print(f"Scanning for {self.ble_device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.ble_device_name in device.name:
                print(f"✓ Found device: {device.name} at {device.address}")
                return device.address
        
        raise Exception(f"Device {self.ble_device_name} not found")
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications and perform inference"""
        start_time = time.time()
        
        # Parse sensor data
        sensor_data = self.parse_sensor_data(data)
        if sensor_data is None:
            return
        
        # Predict hand pose
        prediction = self.predict(sensor_data)
        
        # Convert to JSON
        vr_data = self.prediction_to_json(prediction)
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # ms
        self.latencies.append(latency)
        
        # Print progress
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            avg_latency = np.mean(self.latencies[-100:]) if self.latencies else 0
            print(f"Frame {self.frame_count:4d} | "
                  f"Latency: {latency:.2f}ms (avg: {avg_latency:.2f}ms) | "
                  f"Wrist pos: [{prediction[0]:.3f}, {prediction[1]:.3f}, {prediction[2]:.3f}]")
        
        # NOTE: Here we would send vr_data to Unity via UDP/socket
        # For now, we just process it
        self.last_prediction_time = time.time()
    
    async def connect(self):
        """Connect to ESP32 via BLE"""
        address = await self.find_device()
        self.client = BleakClient(address)
        await self.client.connect()
        print(f"✓ Connected to {address}")
        
        # Subscribe to notifications
        await self.client.start_notify(
            self.ble_char_uuid,
            self.notification_handler
        )
        self.is_running = True
        print(f"✓ Started real-time inference\n")
    
    async def disconnect(self):
        """Disconnect from ESP32"""
        if self.client and self.client.is_connected:
            await self.client.stop_notify(self.ble_char_uuid)
            await self.client.disconnect()
            self.is_running = False
            print("\n✓ Disconnected from BLE device")
    
    async def run(self, duration_seconds: int = 60):
        """
        Run real-time inference for specified duration
        
        Args:
            duration_seconds: How long to run inference
        """
        print("="*60)
        print("REAL-TIME INFERENCE MODE")
        print("="*60)
        print("Camera-free hand tracking using LSTM + Kalman Filter")
        print(f"Duration: {duration_seconds} seconds")
        print("="*60)
        print()
        
        await self.connect()
        
        start_time = time.time()
        
        try:
            print("Running inference... Press Ctrl+C to stop\n")
            
            while True:
                elapsed = time.time() - start_time
                
                if elapsed >= duration_seconds:
                    break
                
                await asyncio.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Inference stopped by user")
        
        finally:
            await self.disconnect()
        
        # Print statistics
        print()
        print("="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {time.time() - start_time:.2f}s")
        print(f"Average FPS: {self.frame_count / (time.time() - start_time):.2f}")
        
        if self.latencies:
            print(f"\nLatency Statistics:")
            print(f"  Mean: {np.mean(self.latencies):.2f}ms")
            print(f"  Median: {np.median(self.latencies):.2f}ms")
            print(f"  Min: {np.min(self.latencies):.2f}ms")
            print(f"  Max: {np.max(self.latencies):.2f}ms")
            print(f"  Std Dev: {np.std(self.latencies):.2f}ms")
        
        print("="*60)
        print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='Run real-time inference with NEURA GLOVE'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/best_model.pth',
        help='Path to trained model (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Inference duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--device', '-dev',
        type=str,
        default='ESP32-BLE',
        help='BLE device name (default: ESP32-BLE)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"✗ Model not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python train_model.py --dataset training_dataset.json")
        return
    
    # Create inference engine
    engine = InferenceEngine(args.model)
    engine.ble_device_name = args.device
    
    # Run inference
    await engine.run(args.duration)


if __name__ == "__main__":
    asyncio.run(main())