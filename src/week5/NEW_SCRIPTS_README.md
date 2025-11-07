# UPDATED SCRIPTS DOCUMENTATION

## Overview of Changes

Three major modifications have been made to the NEURA Glove system:

1. **Single-pose data collectors** - Collect one pose for custom duration (1-300 seconds)
2. **Kalman filtering** - Smooth hand pose predictions in real-time
3. **Model evaluation tools** - Analyze classification performance with confusion matrices and confidence metrics

---

## 1. Single-Pose Data Collection

### Modified Files

#### `ble_flex_collector_single.py`
Collects flex sensor data for ONE pose with custom duration.

**Usage:**
```bash
python ble_flex_collector_single.py <session_id> <pose_name> <duration_seconds>
```

**Example:**
```bash
# Collect 'fist' pose for 60 seconds
python ble_flex_collector_single.py session_001 fist 60

# Collect 'flat_hand' for 120 seconds
python ble_flex_collector_single.py session_002 flat_hand 120
```

**Key Changes:**
- Removed multi-pose sequence loop
- Added duration parameter (1-300 seconds)
- Simplified interface for targeted data collection
- Same output format as original script

**When to use:**
- Collecting extended samples of a single pose
- Building larger datasets for specific gestures
- Testing specific hand positions

---

#### `camera_collector_single.py`
Collects MediaPipe camera data for ONE pose with custom duration.

**Usage:**
```bash
python camera_collector_single.py <session_id> <pose_name> <duration_seconds>
```

**Example:**
```bash
# Collect 'grab' pose for 60 seconds
python camera_collector_single.py session_001 grab 60

# Collect 'peace_sign' for 90 seconds  
python camera_collector_single.py session_003 peace_sign 90
```

**Key Changes:**
- Removed multi-pose sequence loop
- Added duration parameter (1-300 seconds)
- Live video feedback with countdown timer
- Press 'q' to stop early
- Same output format as original script

**Important:**
- Remove glove before camera collection
- Ensure good lighting
- Camera needs clear view of hand

---

### Workflow for Extended Data Collection

**Step 1: Collect Sensor Data (with glove)**
```bash
python ble_flex_collector_single.py session_long fist 60
```

**Step 2: Collect Camera Data (without glove)**
```bash
python camera_collector_single.py session_long fist 60
```

**Step 3: Align Data**
```bash
python data_aligner.py data/sensor_recordings/session_long data/camera_recordings/session_long
```

**Step 4: Train Model**
```bash
python train_model.py data/aligned/aligned_session_long_session_long.json
```

---

## 2. Kalman Filtering for Smooth Predictions

### New File: `realtime_inference_kalman.py`

Real-time inference with Kalman filtering to reduce jitter and smooth hand pose predictions.

### What is Kalman Filtering?

Kalman filtering is a recursive algorithm that optimally estimates the state of a system from noisy measurements. For hand tracking:

- **Problem:** Raw sensor readings and ML predictions have noise/jitter
- **Solution:** Kalman filter predicts next state and updates based on measurements
- **Result:** Smoother, more stable hand movements in Unity

### Mathematical Background

The Kalman filter maintains two key values:
1. **Estimate (x̂):** Current best guess of joint angle
2. **Error covariance (P):** Uncertainty in the estimate

**Prediction step:**
```
x̂_predicted = x̂_previous
P_predicted = P_previous + Q
```

**Update step:**
```
K = P_predicted / (P_predicted + R)  # Kalman gain
x̂ = x̂_predicted + K * (measurement - x̂_predicted)
P = (1 - K) * P_predicted
```

Where:
- **Q (process variance):** How much we expect the value to change naturally
- **R (measurement variance):** Sensor/prediction noise level
- **K (Kalman gain):** Optimal weighting between prediction and measurement

### Usage

**Basic usage (default settings):**
```bash
python realtime_inference_kalman.py models/flex_to_rotation_model.pth
```

**Disable Kalman filtering:**
```bash
python realtime_inference_kalman.py models/flex_to_rotation_model.pth --no-kalman
```

**Custom filter parameters:**
```bash
# Lower process variance = smoother but slower response
python realtime_inference_kalman.py models/flex_to_rotation_model.pth --process-var 0.001 --measurement-var 0.05

# Higher process variance = faster response but less smooth
python realtime_inference_kalman.py models/flex_to_rotation_model.pth --process-var 0.05 --measurement-var 0.2
```

### Parameter Tuning Guide

**Process Variance (Q):**
- **Lower (0.001-0.01):** More smoothing, slower response
- **Higher (0.05-0.1):** Less smoothing, faster response
- **Default:** 0.01 (balanced)

**Measurement Variance (R):**
- **Lower (0.01-0.05):** Trust measurements more (less smoothing)
- **Higher (0.1-0.5):** Trust measurements less (more smoothing)
- **Default:** 0.1 (balanced)

**Recommended settings:**

For **slow, deliberate gestures** (e.g., sign language):
```bash
--process-var 0.001 --measurement-var 0.05
```

For **fast, dynamic gestures** (e.g., gaming):
```bash
--process-var 0.05 --measurement-var 0.2
```

For **general use** (balanced):
```bash
# Use defaults (no flags needed)
```

### Features

1. **Per-joint Kalman filters:** Each of the 10 joint angles (5 joints × 2 axes) has its own filter
2. **Temporal consistency:** Eliminates sudden jumps in predictions
3. **Configurable noise parameters:** Tune Q and R for your use case
4. **Predictions logging:** Saves all predictions for evaluation
5. **Backward compatible:** Can be disabled with `--no-kalman`

### Output

The script creates `predictions_log.json` containing:
- All pose predictions with timestamps
- Confidence scores
- Joint angles
- Metadata (Kalman enabled, duration, etc.)

This log file is used by the evaluation script.

---

## 3. Model Evaluation Tools

### New File: `evaluate_model.py`

Comprehensive evaluation of pose classification performance with visualization.

### Features

1. **Class Distribution Analysis**
   - Bar chart showing prediction counts per pose
   - Percentage breakdown

2. **Confidence Score Analysis**
   - Box plots per class
   - Mean confidence with error bars
   - Statistical summary (mean, std, min, max, median)

3. **Confusion Matrix** (if ground truth provided)
   - Normalized confusion matrix heatmap
   - Classification report (precision, recall, F1-score)
   - Overall accuracy

4. **Temporal Analysis**
   - Confidence scores plotted over time
   - Color-coded by pose class

### Usage

**Basic evaluation (no ground truth):**
```bash
python evaluate_model.py predictions_log.json
```

**With ground truth labels:**
```bash
python evaluate_model.py predictions_log.json --ground-truth ground_truth.json
```

**Custom output directory:**
```bash
python evaluate_model.py predictions_log.json --output-dir my_results
```

**Create ground truth template:**
```bash
python evaluate_model.py --create-template
```

### Ground Truth Format

To evaluate with confusion matrix, create a `ground_truth.json` file:

```json
{
  "metadata": {
    "description": "Ground truth labels for pose predictions"
  },
  "samples": [
    {
      "timestamp": 1234567890.123,
      "true_pose": "fist"
    },
    {
      "timestamp": 1234567891.234,
      "true_pose": "flat_hand"
    }
  ]
}
```

**How to create ground truth:**

1. Generate template:
```bash
python evaluate_model.py --create-template
```

2. Run inference and note timestamps when you make specific poses:
```bash
python realtime_inference_kalman.py models/flex_to_rotation_model.pth
```

3. Edit `ground_truth_template.json` with actual timestamps and poses

4. Run evaluation:
```bash
python evaluate_model.py predictions_log.json --ground-truth ground_truth_template.json
```

### Output Files

The evaluation script generates:

1. **class_distribution.png** - Bar chart of pose predictions
2. **confidence_distribution.png** - Box plots and mean confidence per class
3. **confidence_timeline.png** - Confidence scores over time
4. **confusion_matrix.png** - Normalized confusion matrix (if ground truth provided)

### Interpreting Results

**Class Distribution:**
- Shows which poses are recognized most frequently
- Imbalanced distribution may indicate bias

**Confidence Scores:**
- High confidence (>0.8): Model is certain
- Medium confidence (0.5-0.8): Model is unsure
- Low confidence (<0.5): Likely misclassification or 'unknown'

**Confusion Matrix:**
- Diagonal values: Correct predictions
- Off-diagonal values: Misclassifications
- Dark blue: High proportion, light blue: Low proportion

**Example interpretation:**
```
Confusion Matrix:
           fist  flat_hand  grab
fist        0.95    0.03    0.02
flat_hand   0.05    0.90    0.05
grab        0.02    0.08    0.90
```
- Fist: 95% correctly classified, 3% confused with flat_hand
- Flat_hand: 90% correct, 5% confused with fist, 5% with grab
- Grab: 90% correct, mostly confused with flat_hand when wrong

---

## Complete Workflow Example

### 1. Extended Data Collection

```bash
# Collect 60 seconds each of 3 poses
python ble_flex_collector_single.py session_extended flat_hand 60
python camera_collector_single.py session_extended flat_hand 60

python ble_flex_collector_single.py session_extended fist 60
python camera_collector_single.py session_extended fist 60

python ble_flex_collector_single.py session_extended grab 60
python camera_collector_single.py session_extended grab 60
```

### 2. Align and Train

```bash
# Align each pose
python data_aligner.py \
    data/sensor_recordings/session_extended \
    data/camera_recordings/session_extended

# Train model
python train_model.py \
    data/aligned/aligned_session_extended_session_extended.json
```

### 3. Real-Time Inference with Kalman Filtering

```bash
# Run with Kalman filtering (default parameters)
python realtime_inference_kalman.py models/flex_to_rotation_model.pth
```

Let it run for a few minutes while making different poses. Press Ctrl+C when done.

### 4. Evaluate Performance

```bash
# Basic evaluation (no ground truth)
python evaluate_model.py predictions_log.json --output-dir results

# View generated plots
# results/class_distribution.png
# results/confidence_distribution.png
# results/confidence_timeline.png
```

### 5. Advanced Evaluation (with ground truth)

```bash
# Create template
python evaluate_model.py --create-template

# Edit ground_truth_template.json with actual timestamps and poses
# (Use timestamps from predictions_log.json)

# Run evaluation with ground truth
python evaluate_model.py predictions_log.json \
    --ground-truth ground_truth_template.json \
    --output-dir results_with_gt

# View confusion matrix
# results_with_gt/confusion_matrix.png
```

---

## Performance Tuning

### Reducing Jitter

**Symptom:** Hand movements are jerky in Unity

**Solutions:**
1. **Increase Kalman smoothing:**
   ```bash
   python realtime_inference_kalman.py models/flex_to_rotation_model.pth \
       --process-var 0.001 --measurement-var 0.05
   ```

2. **Collect more training data** (reduces model prediction noise)

3. **Verify sensor connections** (poor connections cause noise)

### Reducing Latency

**Symptom:** Hand movements lag behind actual motion

**Solutions:**
1. **Decrease Kalman smoothing:**
   ```bash
   python realtime_inference_kalman.py models/flex_to_rotation_model.pth \
       --process-var 0.05 --measurement-var 0.2
   ```

2. **Disable Kalman filtering:**
   ```bash
   python realtime_inference_kalman.py models/flex_to_rotation_model.pth --no-kalman
   ```

### Improving Classification Accuracy

**Low accuracy (<80%):**
1. Collect more training data (60+ seconds per pose)
2. Add more diverse poses
3. Ensure consistent pose performance during collection
4. Check sensor calibration
5. Verify camera data quality

**Confusion between specific poses:**
1. Collect more examples of confused poses
2. Adjust pose templates in `POSE_TEMPLATES`
3. Increase classification threshold

---

## Troubleshooting

### Issue: "No predictions in log"

**Cause:** Inference was stopped too quickly

**Solution:** Run inference for at least 30 seconds

### Issue: "High unknown predictions"

**Cause:** Poses don't match templates or threshold too strict

**Solutions:**
1. Adjust threshold in code:
   ```python
   self.pose_classifier = PoseClassifier(threshold=40)  # Increase from 30
   ```

2. Calibrate templates using diagnostic tool:
   ```bash
   python diagnostic_v2.py models/flex_to_rotation_model.pth
   ```

### Issue: "Kalman filtering too slow"

**Cause:** Process variance too low

**Solution:** Increase process variance:
```bash
--process-var 0.05
```

### Issue: "Evaluation script crashes"

**Cause:** Missing dependencies

**Solution:**
```bash
pip install matplotlib seaborn scikit-learn
```

---

## File Summary

### New Files

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `ble_flex_collector_single.py` | Collect sensor data (single pose) | Pose name, duration | JSON data file |
| `camera_collector_single.py` | Collect camera data (single pose) | Pose name, duration | JSON data file |
| `realtime_inference_kalman.py` | Real-time inference with Kalman | Model path, params | UDP stream + log |
| `evaluate_model.py` | Performance evaluation | Predictions log | Plots + metrics |

### Output Files

| File | Created By | Content |
|------|------------|---------|
| `predictions_log.json` | `realtime_inference_kalman.py` | All predictions with timestamps and confidence |
| `class_distribution.png` | `evaluate_model.py` | Bar chart of pose counts |
| `confidence_distribution.png` | `evaluate_model.py` | Confidence statistics per class |
| `confidence_timeline.png` | `evaluate_model.py` | Confidence over time |
| `confusion_matrix.png` | `evaluate_model.py` | Normalized confusion matrix |

---

## Best Practices

### Data Collection
1. Hold each pose consistently for the full duration
2. Collect at least 60 seconds per pose for robust training
3. Collect data from multiple sessions/days for generalization
4. Ensure good lighting for camera collection
5. Verify sensor connections before starting

### Kalman Filtering
1. Start with default parameters
2. Tune only if jitter or latency is noticeable
3. Test with and without filtering to compare
4. Lower process variance for smoother, slower response
5. Higher measurement variance for more smoothing

### Evaluation
1. Collect ground truth labels for at least 100 predictions
2. Perform poses deliberately during evaluation
3. Include transition periods in ground truth
4. Run evaluation multiple times for consistency
5. Compare results with and without Kalman filtering

---

## References

- **Kalman Filtering:** https://en.wikipedia.org/wiki/Kalman_filter
- **Scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Confusion Matrix Interpretation:** https://en.wikipedia.org/wiki/Confusion_matrix

---

## Quick Command Reference

```bash
# Single pose collection (60 seconds)
python ble_flex_collector_single.py session_001 fist 60
python camera_collector_single.py session_001 fist 60

# Align data
python data_aligner.py data/sensor_recordings/session_001 data/camera_recordings/session_001

# Train model
python train_model.py data/aligned/aligned_session_001_session_001.json

# Real-time with Kalman (default)
python realtime_inference_kalman.py models/flex_to_rotation_model.pth

# Real-time without Kalman
python realtime_inference_kalman.py models/flex_to_rotation_model.pth --no-kalman

# Evaluate performance
python evaluate_model.py predictions_log.json --output-dir results
```

---

**Last Updated:** November 2025  
**Version:** 2.0.0
