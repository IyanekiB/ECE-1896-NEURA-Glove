# NEURA GLOVE ML Pipeline

Complete machine learning pipeline for mapping flex sensor signals to MediaPipe-style hand joint rotations for Unity VR hand tracking.

## System Overview

```
ESP32 (Right Hand) → BLE → Flex Angles (5) → ML Model → Joint Rotations (10) → UDP → Unity (Left Hand)
                            ↓
                     MediaPipe Camera (Ground Truth)
```

### Key Features
- **Discrete pose recognition** (no continuous motion tracking)
- **Pose-script alignment** for deterministic frame-by-frame matching
- **Right-to-left hand mirroring** for Unity compatibility
- **5 flex sensors → 5 joint rotations** (Y-axis bending only)
- **Real-time inference at ~10 Hz** streaming to Unity

### Critical Joint Mapping

Based on MediaPipe hand landmark indices and Unity left-hand model:

| Joint Index | Name | MediaPipe | Unity Bone | Rotation Axis |
|-------------|------|-----------|------------|---------------|
| 3 | Thumb IP | THUMB_IP | thumb.intermediate | Y |
| 6 | Index PIP | INDEX_FINGER_PIP | index.proximal | Y |
| 10 | Middle PIP | MIDDLE_FINGER_PIP | middle.proximal | Y |
| 14 | Ring PIP | RING_FINGER_PIP | ring.proximal | Y |
| 18 | Pinky PIP | PINKY_PIP | pinky.proximal | Y |

**Note:** Y-axis rotation = finger bending (0° = straight, 90° = fully bent)

---

## Installation

### Prerequisites
- Python 3.8+
- ESP32 with firmware (`esp32_ble_sender.ino`)
- Webcam for MediaPipe
- Unity project with UDP receiver

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
torch>=1.10.0
mediapipe>=0.9.0
opencv-python>=4.5.0
bleak>=0.19.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---

## Data Collection Workflow

### Phase 1: Sensor Data Collection (With Glove)

Collect flex sensor data from ESP32 worn on **right hand**.

```bash
python ble_flex_collector.py session_001 3
```

**Arguments:**
- `session_001`: Session identifier
- `3`: Duration per pose in seconds

**Poses collected:**
1. flat_hand
2. fist
3. pointing
4. thumbs_up
5. peace_sign
6. ok_sign
7. pinch

**Output structure:**
```
data/sensor_recordings/
└── session_001/
    ├── flat_hand/
    │   └── sensor_data.json
    ├── fist/
    │   └── sensor_data.json
    ├── ...
    └── session_summary.json
```

**Data format:**
```json
{
  "metadata": {
    "pose_name": "fist",
    "session_id": "session_001",
    "total_samples": 30,
    "sampling_rate_hz": 10.2
  },
  "samples": [
    {
      "tick_index": 0,
      "timestamp": 0.0,
      "pose_name": "fist",
      "data": {
        "flex_angles": {
          "thumb": 45.2,
          "index": 78.5,
          "middle": 82.1,
          "ring": 75.3,
          "pinky": 68.9
        }
      }
    }
  ]
}
```

---

### Phase 2: Camera Data Collection (Without Glove)

Remove glove and collect ground-truth rotations from camera on **right hand**.

```bash
python camera_collector.py session_001 3
```

**Same pose sequence** as sensor collection.

**Output structure:**
```
data/camera_recordings/
└── session_001/
    ├── flat_hand/
    │   └── camera_data.json
    ├── fist/
    │   └── camera_data.json
    ├── ...
    └── session_summary.json
```

**Data format:**
```json
{
  "metadata": {
    "pose_name": "fist",
    "session_id": "session_001",
    "total_samples": 90,
    "sampling_rate_hz": 30.1,
    "target_joints": {
      "3": "thumb_ip",
      "6": "index_pip",
      "10": "middle_pip",
      "14": "ring_pip",
      "18": "pinky_pip"
    }
  },
  "samples": [
    {
      "tick_index": 0,
      "timestamp": 0.0,
      "joint_rotations": {
        "thumb_ip": {
          "x_rotation": 0.0,
          "y_rotation": 45.3
        },
        "index_pip": {
          "x_rotation": 0.0,
          "y_rotation": 78.6
        }
      }
    }
  ]
}
```

---

### Phase 3: Data Alignment

Align sensor and camera data by pose and tick index. Camera data is interpolated to match sensor timestamps.

```bash
python data_aligner.py data/sensor_recordings/session_001 data/camera_recordings/session_001
```

**Output:**
```
data/aligned/aligned_session_001_session_001.json
```

**Aligned data format:**
```json
{
  "metadata": {
    "total_pairs": 210,
    "input_dimensions": 5,
    "output_dimensions": 10,
    "input_description": ["thumb_angle", "index_angle", "middle_angle", "ring_angle", "pinky_angle"],
    "output_description": [
      "thumb_ip_x", "thumb_ip_y",
      "index_pip_x", "index_pip_y",
      "middle_pip_x", "middle_pip_y",
      "ring_pip_x", "ring_pip_y",
      "pinky_pip_x", "pinky_pip_y"
    ]
  },
  "pairs": [
    {
      "tick_index": 0,
      "timestamp": 0.0,
      "input": [45.2, 78.5, 82.1, 75.3, 68.9],
      "output": [0.0, 45.3, 0.0, 78.6, 0.0, 82.2, 0.0, 75.4, 0.0, 69.1],
      "pose_name": "fist"
    }
  ]
}
```

---

## Model Training

### Architecture

**Input:** 5 flex sensor angles (0-90°)  
**Output:** 10 rotation values (x,y for 5 joints)

**Network:**
```
Input(5) → Dense(64) → BN → ReLU → Dropout(0.2)
        → Dense(128) → BN → ReLU → Dropout(0.2)
        → Dense(128) → BN → ReLU → Dropout(0.2)
        → Dense(64) → BN → ReLU → Dropout(0.2)
        → Dense(10)
```

### Train the Model

```bash
python train_model.py data/aligned/aligned_session_001_session_001.json
```

**Hyperparameters:**
- Epochs: 200 (with early stopping)
- Learning rate: 0.001
- Batch size: 32
- Train/val split: 80/20
- Loss function: MSE
- Optimizer: Adam with ReduceLROnPlateau

**Output:**
```
models/
├── flex_to_rotation_model.pth
└── training_history.png
```

### Multiple Sessions

For better generalization, collect multiple sessions and train on combined data:

```bash
# Collect multiple sessions
python ble_flex_collector.py session_001 3
python camera_collector.py session_001 3
python ble_flex_collector.py session_002 3
python camera_collector.py session_002 3

# Align each session
python data_aligner.py data/sensor_recordings/session_001 data/camera_recordings/session_001
python data_aligner.py data/sensor_recordings/session_002 data/camera_recordings/session_002

# Combine aligned data (manual JSON merge or script)
# Then train on combined dataset
```

---

## Real-Time Inference

Stream predictions to Unity at 127.0.0.1:5555.

```bash
python realtime_inference.py models/flex_to_rotation_model.pth
```

**Process:**
1. Connects to ESP32 via BLE
2. Receives flex sensor voltages
3. Converts to bend angles (0-90°)
4. Normalizes with trained scalers
5. Predicts joint rotations
6. Mirrors right→left hand
7. Converts to quaternions
8. Builds Unity packet (format2.json structure)
9. Sends via UDP at ~10 Hz

**Unity packet format:**
```json
{
  "timestamp": 1234567890.123,
  "hand": "left",
  "wrist": {"position": [0,0,0], "rotation": [0,0,0,1]},
  "thumb": {
    "metacarpal": {"position": [0,0,0], "rotation": [0,0,0,1]},
    "proximal": {"position": [0,0,0], "rotation": [0,0,0,1]},
    "intermediate": {"position": [0,0,0], "rotation": [0,sin(θ/2),0,cos(θ/2)]},
    "distal": {"position": [0,0,0], "rotation": [0,0,0,1]}
  },
  "index": {...},
  "middle": {...},
  "ring": {...},
  "pinky": {...}
}
```

---

## Directory Structure

```
neura-glove/
├── ble_flex_collector.py       # ESP32 sensor data collection
├── camera_collector.py          # MediaPipe camera data collection
├── data_aligner.py              # Align sensor + camera data
├── train_model.py               # ML model training
├── realtime_inference.py        # Real-time Unity streaming
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   ├── sensor_recordings/       # Raw sensor data
│   │   └── session_XXX/
│   │       └── pose_name/
│   │           └── sensor_data.json
│   ├── camera_recordings/       # Raw camera data
│   │   └── session_XXX/
│   │       └── pose_name/
│   │           └── camera_data.json
│   └── aligned/                 # Aligned training data
│       └── aligned_session_XXX_session_XXX.json
└── models/
    ├── flex_to_rotation_model.pth
    └── training_history.png
```

---

## Calibration

### Flex Sensor Calibration

Adjust voltage ranges in `ble_flex_collector.py` and `realtime_inference.py`:

```python
FLEX_MIN_VOLTAGE = 0.55  # Fully bent (90°)
FLEX_MAX_VOLTAGE = 1.65  # Flat (0°)
```

**To calibrate:**
1. Bend finger fully → record voltage → set as `FLEX_MIN_VOLTAGE`
2. Extend finger fully → record voltage → set as `FLEX_MAX_VOLTAGE`

### Per-Finger Calibration (Advanced)

Modify `FlexDataCollector.calibration` dictionary for individual sensor calibration:

```python
self.calibration = {
    'thumb': {'min': 0.60, 'max': 1.70},
    'index': {'min': 0.55, 'max': 1.65},
    'middle': {'min': 0.58, 'max': 1.68},
    'ring': {'min': 0.57, 'max': 1.67},
    'pinky': {'min': 0.56, 'max': 1.66}
}
```

---

## Right-to-Left Hand Mirroring

**Problem:** Data collected on right hand, Unity uses left-hand model.

**Solution:** Mirror X-axis rotations (negate), keep Y-axis rotations (same).

```python
# In realtime_inference.py
thumb_x, thumb_y = -rotations[0], rotations[1]  # X mirrored, Y same
index_x, index_y = -rotations[2], rotations[3]
# ... etc
```

This is handled automatically in `realtime_inference.py`.

---

## Troubleshooting

### BLE Connection Issues

**Problem:** Cannot find ESP32 device.

**Solution:**
1. Ensure ESP32 is powered and running firmware
2. Check BLE is enabled on PC
3. Verify device name matches `DEVICE_NAME = "ESP32-BLE"`
4. Try scanning manually:
   ```bash
   python -c "from bleak import BleakScanner; import asyncio; asyncio.run(BleakScanner.discover())"
   ```

### Low Sampling Rate

**Problem:** Sensor data rate < 10 Hz.

**Solution:**
1. Check ESP32 serial monitor for actual rate
2. Reduce BLE notification processing overhead
3. Verify `SAMPLE_RATE_MS = 100` in firmware

### Model Poor Performance

**Problem:** High validation loss or poor predictions.

**Solutions:**
1. Collect more diverse poses (10+ poses recommended)
2. Collect multiple sessions from different users
3. Increase pose duration (5-10 seconds)
4. Check calibration values match actual sensor behavior
5. Verify alignment is correct (timestamps match)

### Unity Not Receiving Data

**Problem:** No hand movement in Unity.

**Solution:**
1. Check Unity UDP receiver is listening on port 5555
2. Verify firewall allows UDP on port 5555
3. Test with `netcat`: `nc -u -l 5555` to see if packets arrive
4. Check Unity console for packet parsing errors

---

## Performance Metrics

**Target specifications:**
- Sampling rate: 10 Hz (100ms per frame)
- Inference latency: <10ms
- Total latency: <50ms (BLE + inference + UDP)
- Prediction accuracy: <5° RMSE per joint

**Typical results:**
- Training time: 5-10 minutes (200 epochs)
- Model size: <1 MB
- Real-time FPS: 9-11 Hz
- GPU not required (CPU inference sufficient)

---

## Future Enhancements

1. **Add IMU data** to model input for wrist orientation
2. **Continuous motion tracking** with LSTM/GRU layers
3. **All 21 joints** instead of just 5 primary joints
4. **User-specific calibration** UI in Unity
5. **Online learning** to adapt to user over time
6. **Multiple hand poses** recognition and classification

---

## Citation & References

Based on NEURA Glove Team 10 Conceptual Design Document:
- MediaPipe Hands for ground truth tracking
- ESP32-WROOM-32UE with BLE communication
- 5 flex sensors for finger bend detection
- Unity XR Hand Data for VR integration

**Key Technologies:**
- MediaPipe: https://mediapipe.dev/
- PyTorch: https://pytorch.org/
- Bleak (BLE): https://github.com/hbldh/bleak
- ESP32: https://www.espressif.com/

---

## License

MIT License - NEURA Glove Team 10

---

## Support

For issues or questions:
1. Check this README thoroughly
2. Review project files in `/mnt/project/`
3. Verify all dependencies installed
4. Test each component individually

**Component testing:**
```bash
# Test BLE connection
python ble_flex_collector.py test_session 5

# Test camera
python camera_collector.py test_session 5

# Test alignment
python data_aligner.py data/sensor_recordings/test_session data/camera_recordings/test_session

# Test model
python train_model.py data/aligned/aligned_test_session_test_session.json

# Test inference
python realtime_inference.py models/flex_to_rotation_model.pth
```

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Authors:** NEURA Glove Team 10
