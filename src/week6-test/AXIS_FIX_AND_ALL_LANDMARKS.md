# AXIS FIX & ENABLING ALL LANDMARKS

## Problem 1: Wrong Rotation Axis ✅ FIXED

**Symptom:** Fingers bend sideways instead of curling inward (as shown in your image)

**Root Cause:** Was using **Y-axis rotation** but Unity needs **X-axis rotation** for finger curl

**Solution:** Changed quaternion from `[0, sin(θ/2), 0, cos(θ/2)]` to `[sin(θ/2), 0, 0, cos(θ/2)]`

### Fixed Files

1. **[realtime_inference.py](computer:///mnt/user-data/outputs/realtime_inference.py)** - 5 primary joints, X-axis fixed
2. **[realtime_inference_enhanced.py](computer:///mnt/user-data/outputs/realtime_inference_enhanced.py)** - Supports both 5 joints and ALL 21 joints

---

## Problem 2: Enabling All Landmarks

### Quick Answer

To enable rotations for all 21 MediaPipe landmarks instead of just 5:

**Step 1:** Modify data collection
**Step 2:** Modify model output dimension
**Step 3:** Use enhanced inference script

### Detailed Instructions

#### Option 1: Modify Camera Collector (Recommended)

**File:** `camera_collector.py`

**Change line ~25:**
```python
# OLD - Only 5 primary joints
self.target_joints = {
    3: 'thumb_ip',
    6: 'index_pip',
    10: 'middle_pip',
    14: 'ring_pip',
    18: 'pinky_pip'
}
```

**NEW - All 21 joints:**
```python
# NEW - All 21 MediaPipe landmarks
self.target_joints = {
    0: 'wrist',
    1: 'thumb_cmc', 2: 'thumb_mcp', 3: 'thumb_ip', 4: 'thumb_tip',
    5: 'index_mcp', 6: 'index_pip', 7: 'index_dip', 8: 'index_tip',
    9: 'middle_mcp', 10: 'middle_pip', 11: 'middle_dip', 12: 'middle_tip',
    13: 'ring_mcp', 14: 'ring_pip', 15: 'ring_dip', 16: 'ring_tip',
    17: 'pinky_mcp', 18: 'pinky_pip', 19: 'pinky_dip', 20: 'pinky_tip'
}
```

**Change `calculate_bend_angle` function (~line 50):**

Add cases for all joints:
```python
def calculate_bend_angle(self, joint_idx, landmarks):
    """Calculate bend angle for any joint"""
    
    # Define parent-current-child triplets for ALL joints
    joint_chains = {
        # Wrist (no parent, use palm center)
        0: (0, 0, 9),  # Special case
        
        # Thumb
        1: (0, 1, 2),   # CMC
        2: (1, 2, 3),   # MCP
        3: (2, 3, 4),   # IP
        4: (3, 4, 4),   # Tip (no bend)
        
        # Index
        5: (0, 5, 6),   # MCP
        6: (5, 6, 7),   # PIP
        7: (6, 7, 8),   # DIP
        8: (7, 8, 8),   # Tip
        
        # Middle
        9: (0, 9, 10),  # MCP
        10: (9, 10, 11),  # PIP
        11: (10, 11, 12), # DIP
        12: (11, 12, 12), # Tip
        
        # Ring
        13: (0, 13, 14),  # MCP
        14: (13, 14, 15), # PIP
        15: (14, 15, 16), # DIP
        16: (15, 16, 16), # Tip
        
        # Pinky
        17: (0, 17, 18),  # MCP
        18: (17, 18, 19), # PIP
        19: (18, 19, 20), # DIP
        20: (19, 20, 20)  # Tip
    }
    
    # Rest of function stays same...
```

**Change `process_hand_landmarks` function (~line 90):**
```python
def process_hand_landmarks(self, landmarks):
    """Extract rotation data for ALL joints"""
    joint_rotations = {}
    
    for joint_idx, joint_name in self.target_joints.items():
        y_rotation = self.calculate_bend_angle(joint_idx, landmarks)
        x_rotation = self.calculate_x_rotation(joint_idx, landmarks)
        
        joint_rotations[joint_name] = {
            'joint_index': joint_idx,
            'x_rotation': x_rotation,
            'y_rotation': y_rotation
        }
    
    return joint_rotations
```

#### Option 2: Modify Training Configuration

**File:** `train_model.py` (line ~261)

**Change:**
```python
# OLD
OUTPUT_DIM = 10  # 5 joints × 2 axes

# NEW
OUTPUT_DIM = 42  # 21 joints × 2 axes
```

#### Option 3: Use Enhanced Inference (Easiest)

The **`realtime_inference_enhanced.py`** automatically detects model output dimension:
- 10 outputs → 5 joint mode
- 42 outputs → 21 joint mode

Just train a model with 42 outputs and it will work automatically!

---

## Complete Workflow for All 21 Joints

### Step 1: Collect Data (Modified)

```bash
# Modify camera_collector.py as shown above
python camera_collector.py session_full 3
```

### Step 2: Still Use Same Sensor Data

```bash
# No changes needed - still 5 flex sensors
python ble_flex_collector.py session_full 3
```

### Step 3: Align Data

```bash
python data_aligner.py data/sensor_recordings/session_full data/camera_recordings/session_full
```

**Important:** The aligner will now create pairs with:
- Input: 5 flex angles (same)
- Output: 42 rotation values (21 joints × 2 axes)

### Step 4: Train Model

```bash
# Modify train_model.py OUTPUT_DIM to 42
python train_model.py data/aligned/aligned_session_full_session_full.json
```

### Step 5: Run Enhanced Inference

```bash
python realtime_inference_enhanced.py models/flex_to_rotation_model.pth
```

Output:
```
Loading model from: models/flex_to_rotation_model.pth
  Loaded config: Input(5) -> [64, 128, 128, 64] -> Output(42)
  Mode: ALL 21 JOINTS (full hand)
✓ Model loaded successfully
```

---

## Coordinate System Reference

### Unity Hand Coordinate System

Looking at your image, Unity uses:
- **X-axis (Red):** Side to side (abduction/adduction)
- **Y-axis (Green):** Up (extends upward from wrist)
- **Z-axis (Blue):** Forward (points out from palm)

### Finger Curl Rotation

For fingers to **curl inward** (make a fist):
- **Rotate around X-axis** (the red arrow in your image)
- Positive rotation = finger curls toward palm
- Negative rotation = finger extends away

### Why It Was Wrong

**Before (Y-axis):**
- Rotating around Y (green arrow) = fingers waggle side to side ❌
- This is what you saw in your image

**After (X-axis):**
- Rotating around X (red arrow) = fingers curl inward ✓
- This matches natural finger movement

---

## Joint Indices Reference

```
MediaPipe Hand Landmarks (21 total):

0:  WRIST
1:  THUMB_CMC      5:  INDEX_MCP      9:  MIDDLE_MCP     13: RING_MCP       17: PINKY_MCP
2:  THUMB_MCP      6:  INDEX_PIP      10: MIDDLE_PIP     14: RING_PIP       18: PINKY_PIP
3:  THUMB_IP       7:  INDEX_DIP      11: MIDDLE_DIP     15: RING_DIP       19: PINKY_DIP
4:  THUMB_TIP      8:  INDEX_TIP      12: MIDDLE_TIP     16: RING_TIP       20: PINKY_TIP

Primary joints (5): 3, 6, 10, 14, 18 (the PIP/IP joints that bend most)
```

---

## Testing the Fix

### Test 1: Verify Axis

Run fixed inference and make a **fist**:
- Fingers should curl **inward toward palm** ✓
- Not waggle side to side ✗

### Test 2: Verify All Joints (if enabled)

With 21 joint model:
- All finger segments should bend
- Metacarpal joints (base of fingers) should move
- Wrist should rotate

---

## Quick Reference

| What You Want | Use This File |
|---------------|---------------|
| **Fix axis only** (5 joints) | `realtime_inference.py` |
| **Enable all 21 joints** | `realtime_inference_enhanced.py` + modified collectors |
| **Current working setup** | `realtime_inference.py` (now X-axis) |

---

## Summary of Changes

### ✅ Fixed: Rotation Axis
**File:** `realtime_inference.py`
**Change:** Y-axis → X-axis for quaternion
**Result:** Fingers now curl inward properly

### 📋 To Enable All Landmarks:
1. Modify `camera_collector.py` - add all 21 joints to `target_joints`
2. Modify `train_model.py` - change `OUTPUT_DIM = 42`
3. Recollect camera data with modified collector
4. Align and retrain
5. Use `realtime_inference_enhanced.py`

---

## Files Ready to Use

All fixed files are in `/mnt/user-data/outputs/`:

- ✅ [realtime_inference.py](computer:///mnt/user-data/outputs/realtime_inference.py) - **FIXED X-AXIS** (5 joints)
- ✅ [realtime_inference_enhanced.py](computer:///mnt/user-data/outputs/realtime_inference_enhanced.py) - Supports 5 or 21 joints
- 📝 [camera_collector.py](computer:///mnt/user-data/outputs/camera_collector.py) - Modify for 21 joints
- 📝 [train_model.py](computer:///mnt/user-data/outputs/train_model.py) - Modify OUTPUT_DIM

---

**Try the fixed version now and your fingers should curl properly!**
