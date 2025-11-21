import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_python_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_python_types(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def simulate_sensor_data_from_joints(joints):
    """
    Generate realistic sensor data based on joint positions/rotations
    This simulates what the actual flex sensors and IMU would read
    """
    # Extract key joint positions for sensor simulation
    wrist = joints[0]
    thumb_tip = joints[4]
    index_tip = joints[8]
    middle_tip = joints[12]
    ring_tip = joints[16]
    pinky_tip = joints[20]
    
    # Simulate flex sensors based on finger curl
    # Calculate distance from MCP to tip for each finger
    def calculate_flex(mcp_id, tip_id):
        mcp_pos = np.array(joints[mcp_id]['position'])
        tip_pos = np.array(joints[tip_id]['position'])
        distance = np.linalg.norm(tip_pos - mcp_pos)
        # Convert to ADC value (0-4095 for 12-bit)
        # More curl = higher value, less curl = lower value
        flex_value = int(2048 + distance * 2000)
        flex_value = np.clip(flex_value, 0, 4095)
        return flex_value
    
    flex_sensors = [
        calculate_flex(1, 4),   # Thumb
        calculate_flex(5, 8),   # Index
        calculate_flex(9, 12),  # Middle
        calculate_flex(13, 16), # Ring
        calculate_flex(17, 20)  # Pinky
    ]
    
    # Add some realistic noise and convert to Python int
    flex_sensors = [int(f + np.random.normal(0, 50)) for f in flex_sensors]
    flex_sensors = [int(np.clip(f, 0, 4095)) for f in flex_sensors]
    
    # IMU orientation (use wrist rotation as base)
    wrist_rotation = wrist['rotation']
    imu_orientation = [
        wrist_rotation[0] + np.random.normal(0, 0.01),
        wrist_rotation[1] + np.random.normal(0, 0.01),
        wrist_rotation[2] + np.random.normal(0, 0.01),
        wrist_rotation[3] + np.random.normal(0, 0.01)
    ]
    
    # Normalize quaternion and convert to Python float
    norm = np.linalg.norm(imu_orientation)
    imu_orientation = [float(q / norm) for q in imu_orientation]
    
    # Simulate acceleration (mostly gravity with some hand movement)
    imu_accel = [
        float(np.random.normal(0, 0.5)),
        float(np.random.normal(-9.8, 0.3)),  # Gravity
        float(np.random.normal(0, 0.5))
    ]
    
    # Simulate gyroscope (small rotational velocity)
    imu_gyro = [
        float(np.random.normal(0, 0.05)),
        float(np.random.normal(0, 0.05)),
        float(np.random.normal(0, 0.05))
    ]
    
    return {
        "flex_sensors": flex_sensors,
        "imu_orientation": imu_orientation,
        "imu_accel": imu_accel,
        "imu_gyro": imu_gyro
    }


def create_training_dataset(mediapipe_json_path, output_path="training_dataset.json"):
    """
    Create training dataset by adding simulated sensor data to MediaPipe ground truth
    """
    print(f"Loading MediaPipe data from {mediapipe_json_path}...")
    
    with open(mediapipe_json_path, 'r') as f:
        mediapipe_data = json.load(f)
    
    # Convert loaded data to Python native types (MediaPipe may have numpy types)
    mediapipe_data = convert_to_python_types(mediapipe_data)
    
    frames = mediapipe_data['frames']
    total_frames = len(frames)
    
    print(f"Processing {total_frames} frames...")
    
    training_samples = []
    
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{total_frames}...")
        
        # Generate sensor data based on joint positions
        sensor_data = simulate_sensor_data_from_joints(frame['joints'])
        
        # Create training sample
        sample = {
            "timestamp": int(frame['Timestamp']),
            "frame_number": int(frame['frame_number']),
            
            # INPUT: Sensor data (what ESP32 would send)
            "sensors": {
                "flex": sensor_data['flex_sensors'],  # 5 values
                "imu_orientation": sensor_data['imu_orientation'],  # 4 values (quaternion)
                "imu_accel": sensor_data['imu_accel'],  # 3 values
                "imu_gyro": sensor_data['imu_gyro']  # 3 values
                # Total: 15 input features
            },
            
            # OUTPUT: Ground truth joint positions and rotations
            "ground_truth": {
                "joints": frame['joints']  # 21 joints × 7 values = 147 outputs
            }
        }
        
        training_samples.append(sample)
    
    # Create final dataset
    dataset = {
        "metadata": {
            "total_samples": len(training_samples),
            "input_features": 15,
            "output_values": 147,  # 21 joints × (3 pos + 4 rot)
            "source": str(mediapipe_json_path),
            "description": "Simulated sensor data mapped to MediaPipe ground truth"
        },
        "samples": training_samples
    }
    
    # Convert all numpy types to Python native types
    dataset = convert_to_python_types(dataset)
    
    # Save dataset with custom encoder as backup
    print(f"\nSaving training dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Created training dataset with {len(training_samples)} samples")
    print(f"  Input features: 15 (5 flex + 10 IMU)")
    print(f"  Output values: 147 (21 joints × 7)")
    
    return dataset


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mediapipe_file = sys.argv[1]
    else:
        # Default to looking for file in datasets folder
        mediapipe_file = "datasets/hand_data_20250930_030323.json"
    
    if not Path(mediapipe_file).exists():
        print(f"Error: File not found: {mediapipe_file}")
        print("\nUsage: python create_training_dataset.py <mediapipe_json_file>")
        sys.exit(1)
    
    create_training_dataset(mediapipe_file)