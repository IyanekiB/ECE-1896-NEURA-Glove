"""
NEURA GLOVE - Enhanced Real-time Inference Engine with Pose Detection
Continuous hand pose estimation + discrete pose classification with confidence

ENHANCEMENTS:
- Added pose detection with confidence scores
- Display current pose and confidence in real-time
- Option to send pose classification data to Unity
- Configurable confidence threshold for pose detection

Usage:
    # Run inference with pose detection
    python enhanced_inference_engine.py --model models/best_model.pth
    
    # Adjust smoothing and confidence threshold
    python enhanced_inference_engine.py --model models/best_model.pth \
                                        --kalman-process-noise 0.005 \
                                        --confidence-threshold 0.7
    
    # Send pose data to Unity
    python enhanced_inference_engine.py --model models/best_model.pth --send-pose-to-unity
"""

import asyncio
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
from bleak import BleakClient, BleakScanner


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration (needed for loading checkpoint)"""
    # Sequence parameters
    SEQUENCE_LENGTH: int = 10
    
    # LSTM architecture
    LSTM_HIDDEN_SIZE: int = 256
    LSTM_NUM_LAYERS: int = 3
    LSTM_DROPOUT: float = 0.3
    
    # Input/Output
    INPUT_SIZE: int = 15
    OUTPUT_SIZE: int = 147
    
    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 150
    TRAIN_SPLIT: float = 0.85
    
    # Calibration hyperparameters
    CALIBRATION_LEARNING_RATE: float = 0.0001
    CALIBRATION_EPOCHS: int = 20


PIPE_NAME = "/tmp/neura_glove_pipe"  # Named pipe for Unity communication
POSE_PIPE_NAME = "/tmp/neura_glove_pose_pipe"  # Separate pipe for pose data
SEQUENCE_LENGTH = 10
INPUT_SIZE = 15
OUTPUT_SIZE = 147  # 21 joints × 7 (3 position + 4 rotation quaternion)
TARGET_FPS = 30  # Increased for smoother VR interaction


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
        
        # History for temporal smoothing
        self.pose_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
    def detect_pose(self, pose_logits: np.ndarray) -> Tuple[str, float, dict]:
        """
        Detect pose from classification logits
        
        Args:
            pose_logits: Raw output from classification head
            
        Returns:
            pose_name: Name of detected pose
            confidence: Confidence score (0-1)
            all_confidences: Dictionary of all pose confidences
        """
        # Convert logits to probabilities using softmax
        pose_probs = F.softmax(torch.FloatTensor(pose_logits), dim=0).numpy()
        
        # Get top prediction
        pred_idx = np.argmax(pose_probs)
        confidence = pose_probs[pred_idx]
        pose_name = self.pose_names[pred_idx]
        
        # Create confidence dict for all poses
        all_confidences = {
            self.pose_names[i]: float(pose_probs[i])
            for i in range(len(self.pose_names))
        }
        
        # Add to history for smoothing
        self.pose_history.append(pose_name)
        self.confidence_history.append(confidence)
        
        # Get smoothed pose (most common in recent history)
        if len(self.pose_history) >= self.smoothing_window:
            from collections import Counter
            pose_counts = Counter(self.pose_history)
            smoothed_pose = pose_counts.most_common(1)[0][0]
            smoothed_confidence = np.mean(list(self.confidence_history))
            
            return smoothed_pose, smoothed_confidence, all_confidences
        
        return pose_name, confidence, all_confidences
    
    def is_confident(self, confidence: float) -> bool:
        """Check if confidence exceeds threshold"""
        return confidence >= self.confidence_threshold


# ============================================================================
# KALMAN FILTER
# ============================================================================

class KalmanFilter:
    """Kalman filter for smoothing joint predictions"""
    
    def __init__(self, dim: int, process_noise: float = 0.01, 
                 measurement_noise: float = 0.1):
        self.dim = dim
        self.x = np.zeros(dim)  # State
        self.P = np.eye(dim)    # Covariance
        self.Q = np.eye(dim) * process_noise      # Process noise
        self.R = np.eye(dim) * measurement_noise  # Measurement noise
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with new measurement"""
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        
        # Predict (assuming constant state)
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        self.x = x_pred + K @ (measurement - x_pred)
        self.P = (np.eye(self.dim) - K) @ P_pred
        
        return self.x


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class EnhancedInferenceEngine:
    """Real-time inference with Unity streaming and pose detection"""
    
    def __init__(self, model_path: str, device_name: str = "ESP32-BLE",
                 confidence_threshold: float = 0.6, send_pose_to_unity: bool = False,
                 kalman_process_noise: float = 0.01):
        self.device_name = device_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.send_pose_to_unity = send_pose_to_unity
        
        # Load model
        print(f"🔥 Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config
        config = checkpoint['config']
        self.pose_names = checkpoint['pose_names']
        self.idx_to_pose = {idx: name for name, idx in checkpoint['pose_to_idx'].items()}
        
        # Initialize model
        self.model = HandPoseLSTM(
            lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            lstm_num_layers=config.LSTM_NUM_LAYERS,
            lstm_dropout=config.LSTM_DROPOUT,
            num_poses=len(self.pose_names)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"  ✓ Model loaded ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
        print(f"  ✓ Trained poses: {self.pose_names}")
        print(f"  ✓ Device: {self.device}")
        
        # Initialize pose detector
        self.pose_detector = PoseDetector(
            pose_names=self.pose_names,
            confidence_threshold=confidence_threshold,
            smoothing_window=5
        )
        print(f"  ✓ Pose detector initialized (confidence threshold: {confidence_threshold})")
        
        # Sensor sequence buffer
        self.sensor_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Kalman filters for each joint feature
        self.kalman_filters = [
            KalmanFilter(1, kalman_process_noise, 0.1) 
            for _ in range(OUTPUT_SIZE)
        ]
        
        # BLE connection
        self.service_uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.char_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.client: Optional[BleakClient] = None
        self.latest_data: Optional[bytearray] = None
        
        # Unity pipes
        self.joint_pipe = None
        self.pose_pipe = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = 0
        self.pose_stats = {pose: 0 for pose in self.pose_names}
        
        # Current pose tracking
        self.current_pose = "unknown"
        self.current_confidence = 0.0
    
    async def find_device(self) -> str:
        """Scan for ESP32"""
        print(f"🔍 Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and self.device_name in device.name:
                print(f"  ✓ Found: {device.name} at {device.address}")
                return device.address
        
        raise Exception(f"Device {self.device_name} not found")
    
    def notification_handler(self, sender, data):
        """BLE notification handler"""
        self.latest_data = data
    
    def parse_sensor_data(self, data: bytearray) -> Optional[np.ndarray]:
        """Parse sensor data from ESP32"""
        try:
            data_str = data.decode('utf-8').strip()
            values = [float(x) for x in data_str.split(',')]
            
            if len(values) != 15:
                return None
            
            return np.array(values, dtype=np.float32)
        except:
            return None
    
    async def connect_glove(self):
        """Connect to ESP32 glove"""
        address = await self.find_device()
        self.client = BleakClient(address)
        await self.client.connect()
        print("  ✓ Connected")
        
        await self.client.start_notify(self.char_uuid, self.notification_handler)
        
        # Wait for initial data
        print("  ⏳ Waiting for sensor data...")
        await asyncio.sleep(0.5)
        
        if self.latest_data is None:
            raise Exception("No data received from glove")
        
        print("  ✓ Receiving data")
    
    def setup_unity_pipes(self):
        """Setup named pipes for Unity communication (Unix/Linux/Mac only)"""
        import os
        import sys
        
        print("\n🔗 Setting up Unity communication pipes...")
        
        # Check if running on Windows
        if sys.platform == 'win32':
            print("\n⚠️  WARNING: Named pipes are not supported on Windows in this version.")
            print("   Unity integration is disabled for this session.")
            print("\n   For Windows support, you have two options:")
            print("   1. Run this script in WSL (Windows Subsystem for Linux)")
            print("   2. Use the original inference_engine.py without pose detection")
            print("\n   The script will continue WITHOUT Unity integration.")
            print("   Pose detection will still work and display in console.\n")
            
            # Disable Unity pipes on Windows
            self.joint_pipe = None
            self.pose_pipe = None
            return
        
        # Unix/Linux/Mac named pipes
        # Joint data pipe
        if os.path.exists(PIPE_NAME):
            os.remove(PIPE_NAME)
        os.mkfifo(PIPE_NAME)
        print(f"  ✓ Created joint pipe: {PIPE_NAME}")
        
        # Pose data pipe (optional)
        if self.send_pose_to_unity:
            if os.path.exists(POSE_PIPE_NAME):
                os.remove(POSE_PIPE_NAME)
            os.mkfifo(POSE_PIPE_NAME)
            print(f"  ✓ Created pose pipe: {POSE_PIPE_NAME}")
        
        print(f"  ⏳ Waiting for Unity to connect...")
        
        # Open pipes (blocks until Unity connects)
        self.joint_pipe = open(PIPE_NAME, 'wb', buffering=0)
        print(f"  ✓ Unity connected to joint pipe!")
        
        if self.send_pose_to_unity:
            self.pose_pipe = open(POSE_PIPE_NAME, 'wb', buffering=0)
            print(f"  ✓ Unity connected to pose pipe!")
    
    def send_joints_to_unity(self, joint_data: np.ndarray):
        """Send continuous joint data to Unity via pipe"""
        if self.joint_pipe is None:
            return
        
        try:
            # Convert to bytes (147 floats = 588 bytes)
            joint_bytes = joint_data.astype(np.float32).tobytes()
            
            # Create packet: size + joint_data
            packet = struct.pack('I', len(joint_bytes)) + joint_bytes
            
            self.joint_pipe.write(packet)
            self.joint_pipe.flush()
        except BrokenPipeError:
            print("\n⚠️  Unity disconnected from joint pipe")
            self.joint_pipe = None
    
    def send_pose_to_unity_pipe(self, pose_name: str, confidence: float, 
                                all_confidences: dict):
        """Send pose classification data to Unity via separate pipe"""
        if self.pose_pipe is None:
            return
        
        try:
            # Create JSON payload
            pose_data = {
                'pose': pose_name,
                'confidence': float(confidence),
                'all_confidences': all_confidences,
                'timestamp': time.time()
            }
            
            # Convert to JSON string and encode
            json_str = json.dumps(pose_data)
            json_bytes = json_str.encode('utf-8')
            
            # Create packet: size + json_data
            packet = struct.pack('I', len(json_bytes)) + json_bytes
            
            self.pose_pipe.write(packet)
            self.pose_pipe.flush()
        except BrokenPipeError:
            print("\n⚠️  Unity disconnected from pose pipe")
            self.pose_pipe = None
    
    def predict(self, sensor_sequence: np.ndarray) -> Tuple[np.ndarray, str, float, dict]:
        """
        Run inference on sensor sequence for continuous hand pose + pose classification
        
        Args:
            sensor_sequence: (10, 15) array of sensor readings
        
        Returns:
            joint_data: (147,) array of CONTINUOUS joint values
            pose_name: Detected pose name
            confidence: Confidence score for detected pose
            all_confidences: Dict of all pose confidences
        """
        # Convert to tensor
        x = torch.FloatTensor(sensor_sequence).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            joint_pred, pose_pred = self.model(x)
        
        # Get continuous joint predictions
        joint_data = joint_pred.cpu().numpy()[0]
        
        # Apply Kalman filtering for smooth motion
        filtered_joint_data = np.zeros_like(joint_data)
        for i in range(OUTPUT_SIZE):
            filtered_joint_data[i] = self.kalman_filters[i].update(
                np.array([joint_data[i]])
            )[0]
        
        # Get pose classification
        pose_logits = pose_pred.cpu().numpy()[0]
        pose_name, confidence, all_confidences = self.pose_detector.detect_pose(pose_logits)
        
        return filtered_joint_data, pose_name, confidence, all_confidences
    
    def display_pose_info(self, pose_name: str, confidence: float, 
                         all_confidences: dict, frame_num: int):
        """Display pose detection information"""
        # Confidence bar visualization
        bar_length = 20
        filled = int(bar_length * confidence)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Confidence color indicator
        if confidence >= 0.8:
            conf_indicator = "🟢"
        elif confidence >= 0.6:
            conf_indicator = "🟡"
        else:
            conf_indicator = "🔴"
        
        # Display current pose
        print(f"\n{'='*70}")
        print(f"Frame {frame_num:5d} | Detected Pose: {pose_name.upper():12s} {conf_indicator}")
        print(f"{'='*70}")
        print(f"Confidence: [{bar}] {confidence*100:.1f}%")
        print(f"\nAll Pose Confidences:")
        
        # Sort and display all confidences
        sorted_confidences = sorted(all_confidences.items(), 
                                   key=lambda x: x[1], reverse=True)
        for pose, conf in sorted_confidences:
            mini_bar_len = 15
            mini_filled = int(mini_bar_len * conf)
            mini_bar = '▓' * mini_filled + '░' * (mini_bar_len - mini_filled)
            marker = "◄" if pose == pose_name else " "
            print(f"  {pose:12s}: [{mini_bar}] {conf*100:5.1f}% {marker}")
        print(f"{'='*70}\n")
    
    async def run(self, display_interval: int = 100, verbose_pose: bool = True, no_unity: bool = False):
        """Main inference loop with pose detection"""
        print("\n" + "="*70)
        print("STARTING ENHANCED INFERENCE WITH POSE DETECTION")
        print("="*70)
        print("Mode: Continuous joint prediction + Discrete pose classification")
        print(f"Poses: {', '.join(self.pose_names)}")
        print(f"Confidence threshold: {self.pose_detector.confidence_threshold}")
        print("="*70)
        
        # Connect to glove
        await self.connect_glove()
        
        # Setup Unity pipes (unless disabled)
        if not no_unity:
            self.setup_unity_pipes()
        else:
            print("\n⚠️  Unity integration disabled (--no-unity flag)")
            self.joint_pipe = None
            self.pose_pipe = None
        
        print("\n🎬 Running inference with pose detection...")
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
                                
                                print(f"Frame {self.frame_count:5d} | FPS: {fps:5.1f} | "
                                      f"Pose: {pose_name:12s} {conf_emoji} {confidence*100:5.1f}% | "
                                      f"Joint Sample: [{joint_data[0]:.2f}, {joint_data[1]:.2f}, {joint_data[2]:.2f}]")
                
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
        description='NEURA Glove enhanced real-time inference with pose detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python enhanced_inference_engine.py --model models/best_model.pth
  
  # Adjust confidence threshold
  python enhanced_inference_engine.py --model models/best_model.pth --confidence-threshold 0.7
  
  # Send pose data to Unity
  python enhanced_inference_engine.py --model models/best_model.pth --send-pose-to-unity
  
  # Less verbose output
  python enhanced_inference_engine.py --model models/best_model.pth --display-interval 200 --no-verbose-pose
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--device', default='ESP32-BLE',
                       help='BLE device name (default: ESP32-BLE)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Confidence threshold for pose detection (0-1, default: 0.6)')
    parser.add_argument('--send-pose-to-unity', action='store_true',
                       help='Send pose classification data to Unity via separate pipe')
    parser.add_argument('--no-unity', action='store_true',
                       help='Skip Unity setup (useful for testing pose detection only)')
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
        kalman_process_noise=args.kalman_process_noise
    )
    await engine.run(
        display_interval=args.display_interval,
        verbose_pose=args.verbose_pose,
        no_unity=args.no_unity
    )


if __name__ == "__main__":
    asyncio.run(main())