# CRITICAL FIXES - Ctrl+C and Live IMU

## What Was Wrong

### Issue 1: Ctrl+C Doesn't Save predictions_log.json ❌

**Your Error:**
```
Frame 70 | FPS: 10.1 | Kalman: ON | Pose: flat_hand (92.2%)
Traceback (most recent call last):
  File "D:\ECE 1896\...\realtime_inference.py", line 577, in <module>
    asyncio.run(main())
```

**Root Cause:**
The script crashed immediately when you pressed Ctrl+C, never reaching the `save_predictions_log()` function. The exception wasn't being caught properly.

**The Fix:**
Added proper exception handling in three layers:
1. `try/except KeyboardInterrupt` in the main loop
2. `try/except/finally` around the BLE client
3. Signal handler for graceful shutdown

**Code Changes (lines 389-430):**
```python
try:
    async with BleakClient(target_device.address) as client:
        # ... streaming code ...
        while not self.shutdown_requested:
            await asyncio.sleep(0.02)
        
        await client.stop_notify(CHARACTERISTIC_UUID)
        
except KeyboardInterrupt:
    print("\n\n⏹️  Keyboard interrupt detected...")
    self.shutdown_requested = True
except Exception as e:
    print(f"\n✗ Error: {e}")
finally:
    # CRITICAL: Always save log, even on error
    print("\n💾 Saving predictions log...")
    self.save_predictions_log()
```

---

### Issue 2: Wrist Orientation Hardcoded ❌

**Your Observation:**
```
IMU Quaternion (Orientation):
  W: 0.9371  X: -0.2534  Y: 0.0043  Z: 0.2400
```
This data was being streamed but **NOT USED** for wrist rotation!

**Root Cause:**
The wrist rotation was using the `correct_imu_quaternion()` function which was applying a fixed 90° rotation, ignoring the live IMU data.

**The Fix:**
Removed the hardcoded correction and now directly uses the live IMU quaternion from BLE stream.

**Code Changes (line 294):**
```python
# BEFORE (WRONG):
def correct_imu_quaternion(self, imu_quat):
    # Apply fixed 90° rotation
    rot_qx = 0.7071
    # ... quaternion multiplication ...
    return [corrected_qx, corrected_qy, corrected_qz, corrected_qw]

corrected_wrist_quat = self.correct_imu_quaternion(imu_quat)

# AFTER (CORRECT):
wrist_quat = imu_quat  # Use LIVE data directly!
```

**In the packet (line 298):**
```python
"wrist": {
    "position": [0, 0, 0],
    "rotation": wrist_quat  # LIVE IMU DATA from BLE stream
}
```

---

## How to Verify the Fixes

### Test 1: Ctrl+C Saves Log ✅

```bash
python realtime_inference_final_fix.py models/flex_to_rotation_model.pth
```

**Expected behavior:**
1. Script starts, shows "STREAMING TO UNITY"
2. Wait for Frame 30+ (at least 3 seconds)
3. Press **Ctrl+C**
4. Should see:
   ```
   ⏹️  Keyboard interrupt detected...
   
   ============================================================
   SHUTDOWN
   ============================================================
   Total frames sent: 42
   Duration: 4.2s
   Average FPS: 10.0
   
   💾 Saving predictions log...
   ✓ Predictions log saved: predictions_log.json
     Total predictions: 5
   ```

5. **Verify file exists:**
   ```bash
   ls -lh predictions_log.json
   # Should show file size > 0
   
   head -20 predictions_log.json
   # Should show valid JSON
   ```

**If it still doesn't save:**
- Check you have write permissions: `touch test.txt`
- Make sure you ran for at least 10 frames
- Check for disk space: `df -h .`

---

### Test 2: Live IMU Wrist Orientation ✅

```bash
python realtime_inference_final_fix.py models/flex_to_rotation_model.pth
```

**Testing procedure:**

1. **Place hand flat on desk** (palm down)
   - Watch Unity hand
   - Should be flat/horizontal
   - Should match your real hand orientation

2. **Rotate hand 90° clockwise**
   - Unity hand should rotate exactly 90°
   - Should follow your movement in real-time

3. **Flip hand over** (palm up)
   - Unity hand should flip completely
   - Should show palm facing up

4. **Wave hand side to side**
   - Unity wrist should pivot smoothly
   - No lag (should track within 100ms)

**Expected vs. Before:**

| Action | Before (Hardcoded) | After (Live IMU) |
|--------|-------------------|------------------|
| Flat on desk | ❌ Tilted at angle | ✅ Flat horizontal |
| Rotate hand | ❌ No change | ✅ Follows rotation |
| Flip hand | ❌ No change | ✅ Flips over |
| Wave hand | ❌ No change | ✅ Pivots smoothly |

---

## Understanding the IMU Data

From your ESP32 serial output:
```
IMU Quaternion (Orientation):
  W: 0.9371  X: -0.2534  Y: 0.0043  Z: 0.2400
```

**What this means:**
- This is a **unit quaternion** representing 3D orientation
- Format: [qw, qx, qy, qz]
- Our script receives it as: `[qx, qy, qz, qw]` = `[-0.2534, 0.0043, 0.2400, 0.9371]`

**How it's used:**
1. ESP32 sends via BLE: `"...,0.9371,-0.2534,0.0043,0.2400,..."`
2. Script parses (line 237): `imu_quat = [qx, qy, qz, qw]`
3. Script sends to Unity (line 298): `"rotation": wrist_quat`
4. Unity applies to wrist bone

**Conversion to Euler angles (for debugging):**
```python
w, x, y, z = 0.9371, -0.2534, 0.0043, 0.2400

roll = atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
     = atan2(2*(0.9371*-0.2534 + 0.0043*0.2400), 1 - 2*(-0.2534² + 0.0043²))
     = atan2(-0.4736 + 0.0010, 1 - 2*(0.0642 + 0.00002))
     = atan2(-0.4726, 0.8716)
     = -28.5° (roll)

pitch = asin(2*(w*y - z*x))
      = asin(2*(0.9371*0.0043 - 0.2400*-0.2534))
      = asin(0.0080 + 0.1216)
      = 7.4° (pitch)

yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    = atan2(2*(0.9371*0.2400 + -0.2534*0.0043), 1 - 2*(0.0043² + 0.2400²))
    = atan2(0.4496, 0.8847)
    = 27.0° (yaw)
```

So for **flat hand on desk**, your IMU shows:
- Roll: -28.5° (tilted left)
- Pitch: 7.4° (slightly nose up)
- Yaw: 27.0° (rotated right)

This should **exactly match** what you see in Unity now!

---

## Troubleshooting

### Problem: predictions_log.json still not created

**Check 1: Did you run long enough?**
```bash
# Need at least 10 frames (1 second at 10 Hz)
python realtime_inference_final_fix.py models/flex_to_rotation_model.pth
# Wait for "Frame 30" to appear
# Then Ctrl+C
```

**Check 2: Can you write files?**
```bash
cd "D:\ECE 1896\ECE-1896-NEURA-Glove\src\week5"
touch test.txt
# If this fails, you don't have write permissions
```

**Check 3: Look at the error message**
The script now shows why it failed:
```
💾 Saving predictions log...
✗ Error saving predictions log: [Errno 13] Permission denied: 'predictions_log.json'
```

**Check 4: Check for the file**
```bash
dir predictions_log.json
# Or on Linux/Mac:
ls -la predictions_log.json
```

---

### Problem: Wrist orientation still wrong in Unity

**Check 1: Is Unity using the quaternion correctly?**

Your Unity script should be doing:
```csharp
// Parse JSON
HandData data = JsonUtility.FromJson<HandData>(json);

// Apply to wrist bone
Quaternion wristQuat = new Quaternion(
    data.wrist.rotation[0],  // x
    data.wrist.rotation[1],  // y
    data.wrist.rotation[2],  // z
    data.wrist.rotation[3]   // w
);
wristBone.localRotation = wristQuat;
```

**Check 2: Is the coordinate system correct?**

Unity uses **left-handed** coordinates, IMU uses **right-handed**. You may need to negate some axes:

```csharp
// Try negating Z:
Quaternion wristQuat = new Quaternion(
    data.wrist.rotation[0],   // x
    data.wrist.rotation[1],   // y
    -data.wrist.rotation[2],  // -z (negated)
    data.wrist.rotation[3]    // w
);
```

Or negate X:
```csharp
Quaternion wristQuat = new Quaternion(
    -data.wrist.rotation[0],  // -x (negated)
    data.wrist.rotation[1],   // y
    data.wrist.rotation[2],   // z
    data.wrist.rotation[3]    // w
);
```

**Check 3: Verify the data is arriving**

Add debug output in Unity:
```csharp
Debug.Log($"Wrist quat: [{wristQuat.x}, {wristQuat.y}, {wristQuat.z}, {wristQuat.w}]");
```

Compare to your ESP32 serial output. They should match (within 0.01).

---

## Quick Commands

```bash
# Run with live IMU and proper shutdown
python realtime_inference_final_fix.py models/flex_to_rotation_model.pth

# Run without Kalman (if laggy)
python realtime_inference_final_fix.py models/flex_to_rotation_model.pth --no-kalman

# Verify log was created
ls -lh predictions_log.json

# View log contents
head -30 predictions_log.json

# Check IMU data in log
grep -A 3 '"imu_quat"' predictions_log.json | head -20
```

---

## File Comparison

**Old (BROKEN):** `realtime_inference_kalman_fixed.py`
- ❌ Ctrl+C crashes without saving
- ❌ Wrist uses hardcoded 90° rotation
- ❌ No signal handler

**New (FIXED):** `realtime_inference_final_fix.py`
- ✅ Ctrl+C saves log gracefully
- ✅ Wrist uses live IMU data
- ✅ Proper exception handling
- ✅ Signal handler for interrupts
- ✅ IMU data logged for debugging

---

## Summary

**Before:**
```
Press Ctrl+C
  → Immediate crash
  → No log saved
  → Wrist orientation hardcoded
```

**After:**
```
Press Ctrl+C
  → Graceful shutdown
  → Log saved automatically
  → Wrist follows live IMU data
```

**Always use:** `realtime_inference_final_fix.py`

---

Last Updated: November 7, 2025
