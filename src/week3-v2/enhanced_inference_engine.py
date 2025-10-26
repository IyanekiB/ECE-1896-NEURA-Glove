"""
NEURA GLOVE - Enhanced Real-time Inference Engine with UDP Streaming
Continuous hand pose estimation + UDP streaming to Unity VR

MODIFICATIONS:
- Added UDP streaming using FrameConstructor (compatible with mp_udp_streamer.py)
- Converts 147 predicted values to 21 quaternions
- Sends to Unity at same address/port as mp_udp_streamer.py
- Maintains pose detection functionality
- Compatible with Unity Hand Controller from VR module

Usage:
    # Run inference with UDP streaming to Unity
    python enhanced_inference_engine.py --model models/best_model.pth
    
    # Adjust Unity IP if on different machine
    python enhanced_inference_engine.py --model models/best_model.pth --unity-ip 192.168.1.100
    
    # Disable UDP streaming (use named pipes instead)
    python enhanced_inference_engine.py --model models/best_model.pth --no-udp
"""

import asyncio
import json
import time
import argparse
import numpy as np
import socket
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple
import struct
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from bleak import BleakClient, BleakScanner


# Import FrameConstructor from uploaded file (must be in same directory or PYTHONPATH)
# For this implementation, we'll inline a copy of FrameConstructor
class FrameConstructor:
    """
    Frame constructor compatible with Unity VR Hand Controller
    Builds frames from 21 quaternion rotations
    """
    
    @staticmethod
    def _zero_pos_rot(rotation):
        return {
            "position": [0, 0, 0],
            "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])]
        }

    @staticmethod
    def build_frame_from_rotations(rotations, hand="left"):
        """
        Build a frame matching Unity VR format from a list of 21 quaternions.

        rotations: list[21] of [x, y, z, w] quaternions corresponding to MediaPipe landmarks indices 0..20
        hand: "left" or "right"
        """

        if rotations is None or len(rotations) != 21:
            raise ValueError("rotations must be a list of length 21")

        # MediaPipe indices mapping to bones
        # 0: wrist
        # Thumb: 1(CMC),2(MCP),3(IP),4(Tip)
        # Index: 5(MCP),6(PIP),7(DIP),8(Tip)
        # Middle: 9(MCP),10(PIP),11(DIP),12(Tip)
        # Ring: 13(MCP),14(PIP),15(DIP),16(Tip)
        # Pinky: 17(MCP),18(PIP),19(DIP),20(Tip)

        frame = {
            "timestamp": time.time(),
            "hand": hand,
            "wrist": {
                "position": [0, 0, 0],
                "rotation": [
                    float(rotations[0][0]),
                    float(rotations[0][1]),
                    float(rotations[0][2]),
                    float(rotations[0][3])
                ]
            },
            "thumb": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[1]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[2]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[3]),
                "distal": FrameConstructor._zero_pos_rot(rotations[4])
            },
            "index": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[5]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[6]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[7]),
                "distal": FrameConstructor._zero_pos_rot(rotations[8])
            },
            "middle": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[9]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[10]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[11]),
                "distal": FrameConstructor._zero_pos_rot(rotations[12])
            },
            "ring": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[13]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[14]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[15]),
                "distal": FrameConstructor._zero_pos_rot(rotations[16])
            },
            "pinky": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[17]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[18]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[19]),
                "distal": FrameConstructor._zero_pos_rot(rotations[20])
            }
        }

        return frame


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration (needed for loading checkpoint)"""
    SEQUENCE_LENGTH: int = 10
    LSTM_HIDDEN_SIZE: int = 256
    LSTM_NUM_LAYERS: int = 3
    LSTM_DROPOUT: float = 0.3
    INPUT_SIZE: int = 15
    OUTPUT_SIZE: int = 147
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 150
    TRAIN_SPLIT: float = 0.85
    CALIBRATION_LEARNING_RATE: float = 0.0001
    CALIBRATION_EPOCHS: int = 20


PIPE_NAME = "/tmp/neura_glove_pipe"  # Named pipe for Unity communication (legacy)
POSE_PIPE_NAME = "/tmp/neura_glove_pose_pipe"  # Separate pipe for pose data
SEQUENCE_LENGTH = 10
INPUT_SIZE = 15
OUTPUT_SIZE = 147  # 21 joints × 7 (3 position + 4 rotation quaternion)
TARGET_FPS = 30

# UDP Configuration (matching mp_udp_streamer.py)
DEFAULT_UNITY_IP = '127.0.0.1'
DEFAULT_UNITY_PORT = 5555


# ============================================================================
# LSTM MODEL (must match training)
# ============================================================================

class HandPoseLSTM(nn.Module):
    """Hand pose LSTM model (same as training)"""
    
    def __init__(self, lstm_hidden_size, lstm_num_layers, lstm_dropout, num_poses):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        self.joint_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, OUTPUT_SIZE)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_poses)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        joint_pred = self.joint_head(last_output)
        pose_pred = self.classification_head(last_output)
        return joint_pred, pose_pred


# ============================================================================
# POSE DETECTION
# ============================================================================

class PoseDetector:
    """Handles pose classification with confidence scores"""
    
    def __init__(self, pose_names: List[str], confidence_threshold: float = 0.6,
                 smoothing_window: int = 5):
        self.pose_names = pose_names
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.pose_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
    def detect_pose(self, pose_logits: np.ndarray) -> Tuple[str, float, dict]:
        """Detect pose from classification logits"""
        pose_probs = F.softmax(torch.FloatTensor(pose_logits), dim=0).numpy()
        pred_idx = np.argmax(pose_probs)
        confidence = pose_probs[pred_idx]
        pose_name = self.pose_names[pred_idx]
        
        all_confidences = {
            self.pose_names[i]: float(pose_probs[i])
            for i in range(len(self.pose_names))
        }
        
        self.pose_history.append(pose_name)
        self.confidence_history.append(confidence)
        
        if len(self.pose_history) >= self.smoothing_window:
            from collections import Counter
            pose_counts = Counter(self.pose_history)
            smoothed_pose = pose_counts.most_common(1)[0][0]
            smoothed_confidence = np.mean(list(self.confidence_history))
            return smoothed_pose, smoothed_confidence, all_confidences
        
        return pose_name, confidence, all_confidences
    
    def is_confident(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold


# ============================================================================
# KALMAN FILTER
# ============================================================================

class KalmanFilter:
    """Kalman filter for smoothing joint predictions"""
    
    def __init__(self, dim: int, process_noise: float = 0.01, 
                 measurement_noise: float = 0.1):
        self.dim = dim
        self.state = np.zeros(dim)
        self.covariance = np.eye(dim)
        self.process_noise = process_noise * np.eye(dim)
        self.measurement_noise = measurement_noise * np.eye(dim)
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.state = measurement
            self.initialized = True
            return self.state
        
        # Prediction
        predicted_state = self.state
        predicted_covariance = self.covariance + self.process_noise
        
        # Update
        innovation = measurement - predicted_state
        innovation_covariance = predicted_covariance + self.measurement_noise
        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        
        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = (np.eye(self.dim) - kalman_gain) @ predicted_covariance
        
        return self.state


# ============================================================================
# DATA CONVERSION UTILITIES
# ============================================================================

class DataConverter:
    """Convert between 147-value format and 21-quaternion format"""
    
    @staticmethod
    def joint_data_to_rotations(joint_data: np.ndarray) -> List[List[float]]:
        """
        Convert 147-value joint data to 21 quaternions.
        
        joint_data: Array of 147 values (21 joints × 7 values each)
                    Each joint: [x_pos, y_pos, z_pos, qx, qy, qz, qw]
        
        Returns: List of 21 quaternions [[x, y, z, w], ...]
        """
        rotations = []
        
        for i in range(21):
            start_idx = i * 7
            # Extract quaternion (skip position values at indices 0-2)
            qx = float(joint_data[start_idx + 3])
            qy = float(joint_data[start_idx + 4])
            qz = float(joint_data[start_idx + 5])
            qw = float(joint_data[start_idx + 6])
            
            rotations.append([qx, qy, qz, qw])
        
        return rotations
    
    @staticmethod
    def normalize_quaternion(q: List[float]) -> List[float]:
        """Normalize a quaternion to unit length"""
        qx, qy, qz, qw = q
        magnitude = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if magnitude < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        return [qx/magnitude, qy/magnitude, qz/magnitude, qw/magnitude]


# ============================================================================
# ENHANCED INFERENCE ENGINE
# ============================================================================

class EnhancedInferenceEngine:
    """Real-time inference with UDP streaming to Unity"""
    
    def __init__(self, model_path: str, device_name: str = 'ESP32-BLE',
                 confidence_threshold: float = 0.6,
                 send_pose_to_unity: bool = False,
                 kalman_process_noise: float = 0.01,
                 unity_ip: str = DEFAULT_UNITY_IP,
                 unity_port: int = DEFAULT_UNITY_PORT,
                 use_udp: bool = True):
        
        self.model_path = model_path
        self.device_name = device_name
        self.send_pose_to_unity = send_pose_to_unity
        self.use_udp = use_udp
        
        # UDP setup
        self.unity_ip = unity_ip
        self.unity_port = unity_port
        self.udp_socket = None
        if self.use_udp:
            self.setup_udp()
        
        # BLE
        self.client: Optional[BleakClient] = None
        self.char_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.latest_data = None
        
        # Model
        self.model = None
        self.pose_names = []
        self.load_model()
        
        # Pose detection
        self.pose_detector = PoseDetector(
            self.pose_names, 
            confidence_threshold=confidence_threshold
        )
        
        # Kalman filter
        self.kalman = KalmanFilter(OUTPUT_SIZE, process_noise=kalman_process_noise)
        
        # Buffering
        self.sensor_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Named pipes (legacy support)
        self.joint_pipe = None
        self.pose_pipe = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = 0
        self.current_pose = "unknown"
        self.current_confidence = 0.0
        self.pose_stats = {pose: 0 for pose in self.pose_names}
    
    def setup_udp(self):
        """Setup UDP socket for Unity communication"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"✓ UDP socket created for Unity streaming")
            print(f"  Target: {self.unity_ip}:{self.unity_port}")
        except Exception as e:
            print(f"❌ Failed to create UDP socket: {e}")
            self.udp_socket = None
    
    def send_udp_to_unity(self, rotations: List[List[float]]):
        """
        Send joint rotations to Unity via UDP using FrameConstructor format.
        
        Args:
            rotations: List of 21 quaternions [[x, y, z, w], ...]
        """
        if not self.udp_socket:
            return
        
        try:
            # Normalize all quaternions
            normalized_rotations = [
                DataConverter.normalize_quaternion(r) for r in rotations
            ]
            
            # Build frame using FrameConstructor
            frame = FrameConstructor.build_frame_from_rotations(
                normalized_rotations, 
                hand="left"
            )
            
            # Convert to JSON and send
            json_data = json.dumps(frame)
            self.udp_socket.sendto(
                json_data.encode('utf-8'),
                (self.unity_ip, self.unity_port)
            )
            
        except Exception as e:
            print(f"❌ Error sending UDP to Unity: {e}")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"\n📂 Loading model from: {self.model_path}")
        
        # PyTorch 2.6+ compatibility: weights_only=False for custom classes
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Extract pose names
        self.pose_names = checkpoint.get('pose_names', ['fist', 'open', 'point', 'peace', 'thumbs_up'])
        num_poses = len(self.pose_names)
        
        # Load config
        config = checkpoint.get('config', TrainingConfig())
        
        # Create model
        self.model = HandPoseLSTM(
            lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            lstm_num_layers=config.LSTM_NUM_LAYERS,
            lstm_dropout=config.LSTM_DROPOUT,
            num_poses=num_poses
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Poses: {', '.join(self.pose_names)}")
        print(f"  Architecture: LSTM({config.LSTM_HIDDEN_SIZE}, {config.LSTM_NUM_LAYERS} layers)")
    
    def setup_unity_pipes(self):
        """Setup named pipes for Unity (legacy support)"""
        try:
            import os
            if not os.path.exists(PIPE_NAME):
                os.mkfifo(PIPE_NAME)
            self.joint_pipe = open(PIPE_NAME, 'w')
            
            if self.send_pose_to_unity:
                if not os.path.exists(POSE_PIPE_NAME):
                    os.mkfifo(POSE_PIPE_NAME)
                self.pose_pipe = open(POSE_PIPE_NAME, 'w')
            
            print("✓ Named pipes setup complete")
        except Exception as e:
            print(f"⚠️  Named pipe setup failed: {e}")
            self.joint_pipe = None
            self.pose_pipe = None
    
    async def connect_glove(self):
        """Connect to NEURA Glove via BLE"""
        print(f"\n🔍 Scanning for {self.device_name}...")
        
        devices = await BleakScanner.discover(timeout=5.0)
        target_device = None
        
        for device in devices:
            if device.name == self.device_name:
                target_device = device
                break
        
        if not target_device:
            raise Exception(f"Device {self.device_name} not found")
        
        print(f"✓ Found device: {target_device.name} ({target_device.address})")
        print(f"Connecting...")
        
        self.client = BleakClient(target_device.address)
        await self.client.connect()
        
        if not self.client.is_connected:
            raise Exception("Failed to connect")
        
        print("✓ Connected to glove")
        
        # Start notifications
        await self.client.start_notify(self.char_uuid, self.notification_handler)
        print("✓ Notifications enabled")
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications"""
        self.latest_data = data
    
    def parse_sensor_data(self, data: bytes) -> Optional[np.ndarray]:
        """Parse sensor data from BLE packet"""
        try:
            if len(data) < 60:
                return None
            
            # 5 flex sensors (float32)
            flex_values = struct.unpack('<5f', data[0:20])
            
            # IMU quaternion (float32 x 4)
            quat_values = struct.unpack('<4f', data[20:36])
            
            # Accelerometer (float32 x 3)
            accel_values = struct.unpack('<3f', data[36:48])
            
            # Gyroscope (float32 x 3)
            gyro_values = struct.unpack('<3f', data[48:60])
            
            # Combine into 15-element array
            sensor_data = np.array(
                list(flex_values) + list(quat_values) + 
                list(accel_values) + list(gyro_values),
                dtype=np.float32
            )
            
            return sensor_data
            
        except Exception as e:
            return None
    
    def predict(self, sensor_sequence: np.ndarray) -> Tuple[np.ndarray, str, float, dict]:
        """
        Run inference on sensor sequence.
        
        Returns:
            joint_data: 147-value array (smoothed)
            pose_name: Detected pose name
            confidence: Pose confidence
            all_confidences: All pose confidences
        """
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(sensor_sequence).unsqueeze(0)
            
            # Forward pass
            joint_pred, pose_pred = self.model(x)
            
            # Extract predictions
            joint_data = joint_pred.squeeze().numpy()
            pose_logits = pose_pred.squeeze().numpy()
            
            # Apply Kalman smoothing
            joint_data = self.kalman.update(joint_data)
            
            # Detect pose
            pose_name, confidence, all_confidences = \
                self.pose_detector.detect_pose(pose_logits)
            
            return joint_data, pose_name, confidence, all_confidences
    
    def send_joints_to_unity(self, joint_data: np.ndarray):
        """Send joint data to Unity (UDP or named pipe)"""
        if self.use_udp and self.udp_socket:
            # Convert 147 values to 21 quaternions
            rotations = DataConverter.joint_data_to_rotations(joint_data)
            # Send via UDP
            self.send_udp_to_unity(rotations)
        
        # Also send via named pipe if available (legacy support)
        if self.joint_pipe:
            try:
                packet = {
                    'timestamp': time.time(),
                    'joints': joint_data.tolist()
                }
                self.joint_pipe.write(json.dumps(packet) + '\n')
                self.joint_pipe.flush()
            except:
                pass
    
    def send_pose_to_unity_pipe(self, pose_name: str, confidence: float, 
                                 all_confidences: dict):
        """Send pose data to Unity via named pipe"""
        if self.pose_pipe:
            try:
                packet = {
                    'timestamp': time.time(),
                    'pose': pose_name,
                    'confidence': float(confidence),
                    'all_confidences': all_confidences
                }
                self.pose_pipe.write(json.dumps(packet) + '\n')
                self.pose_pipe.flush()
            except:
                pass
    
    def display_pose_info(self, pose_name: str, confidence: float, 
                         all_confidences: dict, frame_num: int):
        """Display detailed pose information"""
        print(f"\n{'='*60}")
        print(f"Frame {frame_num} - Pose Detection")
        print(f"{'='*60}")
        print(f"  Primary Pose: {pose_name}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"\n  All Pose Confidences:")
        for pose, conf in sorted(all_confidences.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(conf * 20)
            bar = '█' * bar_len
            print(f"    {pose:12s}: {conf*100:5.1f}% {bar}")
        print(f"{'='*60}\n")
    
    async def run(self, display_interval: int = 100, verbose_pose: bool = True,
                  no_unity: bool = False):
        """Main inference loop"""
        
        print("\n" + "="*70)
        print("STARTING ENHANCED INFERENCE WITH UDP STREAMING")
        print("="*70)
        print("Mode: Continuous joint prediction + UDP streaming to Unity")
        print(f"Poses: {', '.join(self.pose_names)}")
        print(f"Confidence threshold: {self.pose_detector.confidence_threshold}")
        if self.use_udp:
            print(f"UDP Target: {self.unity_ip}:{self.unity_port}")
        print("="*70)
        
        # Connect to glove
        await self.connect_glove()
        
        # Setup Unity pipes (unless disabled)
        if not no_unity and not self.use_udp:
            self.setup_unity_pipes()
        
        print("\n🎬 Running inference with UDP streaming...")
        print(f"   Target: {TARGET_FPS} Hz")
        print("   Press Ctrl+C to stop")
        print("-"*70)
        
        self.start_time = time.time()
        self.frame_count = 0
        last_pose_display_frame = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Get sensor data
                if self.latest_data:
                    sensor_data = self.parse_sensor_data(self.latest_data)
                    
                    if sensor_data is not None:
                        # Add to buffer
                        self.sensor_buffer.append(sensor_data)
                        
                        # Run inference when buffer is full
                        if len(self.sensor_buffer) == SEQUENCE_LENGTH:
                            # Convert buffer to sequence
                            sensor_sequence = np.array(list(self.sensor_buffer))
                            
                            # Predict joint angles and pose
                            joint_data, pose_name, confidence, all_confidences = \
                                self.predict(sensor_sequence)
                            
                            # Update current pose
                            self.current_pose = pose_name
                            self.current_confidence = confidence
                            
                            # Track pose statistics
                            if self.pose_detector.is_confident(confidence):
                                self.pose_stats[pose_name] += 1
                            
                            # Send to Unity
                            self.send_joints_to_unity(joint_data)
                            if self.send_pose_to_unity:
                                self.send_pose_to_unity_pipe(pose_name, confidence, 
                                                            all_confidences)
                            
                            # Statistics
                            self.frame_count += 1
                            
                            # Display pose info (every N frames or when verbose)
                            if verbose_pose and (self.frame_count - last_pose_display_frame >= 30):
                                self.display_pose_info(pose_name, confidence, 
                                                      all_confidences, self.frame_count)
                                last_pose_display_frame = self.frame_count
                            
                            # Periodic concise feedback
                            elif self.frame_count % display_interval == 0:
                                elapsed = time.time() - self.start_time
                                fps = self.frame_count / elapsed
                                
                                # Compact pose display
                                conf_emoji = "🟢" if confidence >= 0.8 else \
                                           "🟡" if confidence >= 0.6 else "🔴"
                                
                                # Show sample rotation
                                rotations = DataConverter.joint_data_to_rotations(joint_data)
                                sample_rot = rotations[5]  # Index MCP
                                
                                print(f"Frame {self.frame_count:5d} | FPS: {fps:5.1f} | "
                                      f"Pose: {pose_name:12s} {conf_emoji} {confidence*100:5.1f}% | "
                                      f"Sample Quat: [{sample_rot[0]:.2f}, {sample_rot[1]:.2f}, "
                                      f"{sample_rot[2]:.2f}, {sample_rot[3]:.2f}]")
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0/TARGET_FPS - elapsed)
                await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping inference...")
        
        finally:
            # Cleanup
            if self.client and self.client.is_connected:
                await self.client.stop_notify(self.char_uuid)
                await self.client.disconnect()
            
            if self.udp_socket:
                self.udp_socket.close()
            
            if self.joint_pipe:
                self.joint_pipe.close()
            if self.pose_pipe:
                self.pose_pipe.close()
            
            # Final statistics
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            print()
            print("="*70)
            print("INFERENCE STOPPED - FINAL STATISTICS")
            print("="*70)
            print(f"  Total frames: {self.frame_count}")
            print(f"  Duration: {elapsed:.2f}s")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Target FPS: {TARGET_FPS}")
            if self.use_udp:
                print(f"  UDP Target: {self.unity_ip}:{self.unity_port}")
            print()
            print("Pose Distribution (confident detections only):")
            total_confident = sum(self.pose_stats.values())
            for pose, count in sorted(self.pose_stats.items(), 
                                     key=lambda x: x[1], reverse=True):
                percentage = (count / total_confident * 100) if total_confident > 0 else 0
                bar_len = int(percentage / 5)
                bar = '▓' * bar_len
                print(f"  {pose:12s}: {count:5d} frames ({percentage:5.1f}%) {bar}")
            print("="*70)


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='NEURA Glove enhanced real-time inference with UDP streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with UDP streaming
  python enhanced_inference_engine.py --model models/best_model.pth
  
  # Specify Unity IP if on different machine
  python enhanced_inference_engine.py --model models/best_model.pth --unity-ip 192.168.1.100
  
  # Use named pipes instead of UDP
  python enhanced_inference_engine.py --model models/best_model.pth --no-udp
  
  # Adjust confidence threshold
  python enhanced_inference_engine.py --model models/best_model.pth --confidence-threshold 0.7
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--device', default='ESP32-BLE',
                       help='BLE device name (default: ESP32-BLE)')
    parser.add_argument('--unity-ip', default=DEFAULT_UNITY_IP,
                       help=f'Unity IP address (default: {DEFAULT_UNITY_IP})')
    parser.add_argument('--unity-port', type=int, default=DEFAULT_UNITY_PORT,
                       help=f'Unity UDP port (default: {DEFAULT_UNITY_PORT})')
    parser.add_argument('--no-udp', action='store_true',
                       help='Disable UDP streaming (use named pipes instead)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Confidence threshold for pose detection (0-1, default: 0.6)')
    parser.add_argument('--send-pose-to-unity', action='store_true',
                       help='Send pose classification data to Unity via separate pipe')
    parser.add_argument('--no-unity', action='store_true',
                       help='Skip Unity setup (useful for testing)')
    parser.add_argument('--kalman-process-noise', type=float, default=0.01,
                       help='Kalman filter process noise (default: 0.01)')
    parser.add_argument('--display-interval', type=int, default=100,
                       help='Display stats every N frames (default: 100)')
    parser.add_argument('--verbose-pose', dest='verbose_pose', action='store_true',
                       help='Show detailed pose info every 30 frames (default)')
    parser.add_argument('--no-verbose-pose', dest='verbose_pose', action='store_false',
                       help='Disable detailed pose display')
    parser.set_defaults(verbose_pose=True)
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("Confidence threshold must be between 0 and 1")
    
    # Create and run inference engine
    engine = EnhancedInferenceEngine(
        model_path=args.model,
        device_name=args.device,
        confidence_threshold=args.confidence_threshold,
        send_pose_to_unity=args.send_pose_to_unity,
        kalman_process_noise=args.kalman_process_noise,
        unity_ip=args.unity_ip,
        unity_port=args.unity_port,
        use_udp=not args.no_udp
    )
    await engine.run(
        display_interval=args.display_interval,
        verbose_pose=args.verbose_pose,
        no_unity=args.no_unity
    )


if __name__ == "__main__":
    asyncio.run(main())