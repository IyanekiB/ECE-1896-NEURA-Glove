"""
NEURA Glove - Core Data Structures
Shared data types used across all modules
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class SensorData:
    """Raw sensor data from ESP32 glove"""
    timestamp: float
    flex_sensors: np.ndarray  # 5 values
    imu_accel: np.ndarray     # 3D acceleration
    imu_gyro: np.ndarray      # 3D angular velocity  
    imu_quat: np.ndarray      # 4D quaternion (w, x, y, z)


@dataclass
class MediaPipeData:
    """Ground truth from MediaPipe camera"""
    timestamp: float
    landmarks: np.ndarray     # 21 joints × 3 coordinates
    hand_present: bool
    confidence: float


@dataclass
class SynchronizedSample:
    """Combined sensor + ground truth pair for training"""
    timestamp: float
    sensor: SensorData
    ground_truth: MediaPipeData
    
    def to_dict(self):
        """Convert to dictionary for JSON saving"""
        return {
            'timestamp': self.timestamp,
            'sensor': {
                'flex': self.sensor.flex_sensors.tolist(),
                'accel': self.sensor.imu_accel.tolist(),
                'gyro': self.sensor.imu_gyro.tolist(),
                'quat': self.sensor.imu_quat.tolist()
            },
            'ground_truth': {
                'landmarks': self.ground_truth.landmarks.tolist(),
                'confidence': float(self.ground_truth.confidence)
            }
        }


@dataclass
class HandPoseOutput:
    """Final hand pose for Unity VR"""
    timestamp: float
    joints: list  # 21 joints with position + rotation
    confidence: float
    
    def to_unity_json(self):
        """Convert to Unity JSON format"""
        return json.dumps({
            "Timestamp": int(self.timestamp * 1000000),
            "joints": self.joints
        })