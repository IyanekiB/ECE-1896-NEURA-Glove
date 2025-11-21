# KPC Results Analysis

## KPC1: Frame-by-frame synchronization

```
PS D:\ECE 1896\ECE-1896-NEURA-Glove\src\week6-test> python test_kpc1_sync.py data/sensor_recordings/session_001 data/camera_recordings/session_001 --max-gap 0.05

============================================================
KPC1 – FRAME-BY-FRAME SYNCHRONIZATION TEST
============================================================
Sensor session : data\sensor_recordings\session_001
Camera session : data\camera_recordings\session_001
Max acceptable gap (KPC1 target) : 50.0 ms

Per-pose synchronization stats (overlapping region only):
  fist         | eval samples:  234 | mean:   8.64 ms | std:   5.55 ms | max:  25.20 ms | within target: 100.0%
  flat_hand    | eval samples:  235 | mean:   9.65 ms | std:   6.13 ms | max:  29.80 ms | within target: 100.0%
  grab         | eval samples:  244 | mean:   9.25 ms | std:   6.39 ms | max:  31.95 ms | within target: 100.0%
  peace_sign   | eval samples:  235 | mean:  10.50 ms | std:   6.31 ms | max:  30.80 ms | within target: 100.0%
  point        | eval samples:  245 | mean:  10.51 ms | std:   5.70 ms | max:  26.07 ms | within target: 100.0%

Overall synchronization stats (all poses, overlapping region):
  Total eval samples : 1193
  Mean gap           : 9.71 ms
  Std gap            : 6.07 ms
  Max gap (observed) : 31.95 ms
  Within target      : 100.0% of samples

KPC1 verdict:
  Mean gap <= target? PASS
  Std  gap <= target? PASS

Saved: kpc1_sync_histogram.png
```

## KPC2: Dataset quality and model convergence

```
PS D:\ECE 1896\ECE-1896-NEURA-Glove\src\week6-test> python test_kpc2_dataset_and_training.py data/aligned/aligned_session_001_session_001.json models/flex_to_rotation_model.pth

============================================================
KPC2 – DATASET QUALITY AND MODEL CONVERGENCE TEST
============================================================
Aligned dataset : data/aligned/aligned_session_001_session_001.json
Checkpoint      : models/flex_to_rotation_model.pth

Sample retention (per pose):
  fist       | sensor:  601 | pairs:  601 | retention: 100.0%
  flat_hand  | sensor:  600 | pairs:  600 | retention: 100.0%
  grab       | sensor:  601 | pairs:  601 | retention: 100.0%
  peace_sign | sensor:  599 | pairs:  599 | retention: 100.0%
  point      | sensor:  600 | pairs:  600 | retention: 100.0%

Overall retention: 100.00%
Retention ≥ 85% ? PASS
Saved: kpc2_retention_by_pose.png

Best validation loss: 0.0227
Validation loss < 0.1 ? PASS
Saved: kpc2_val_loss_from_checkpoint.png

Done.
```

## KPC3: Camera-free operation accuracy

```
PS D:\ECE 1896\ECE-1896-NEURA-Glove\src\week6-test> python test_kpc3_camera_free_accuracy.py predictions_log.json ground_truth_labels.json --output-dir kpc3_results
Loading predictions from: predictions_log.json
  Total predictions: 58
  Duration: 57.39s
  Kalman enabled: True

Loading ground truth from: ground_truth_labels.json
  Ground truth samples: 58

============================================================
KPC3 – CAMERA-FREE OPERATION ACCURACY
============================================================
Number of labeled frames: 58
Overall classification accuracy: 84.48%
KPC3 target (>= 80%) met? PASS

============================================================
RUNNING FULL EVALUATION
============================================================

============================================================
EVALUATION SUMMARY
============================================================

Class Distribution:
  flat_hand :  20 ( 34.5%)
  fist      :  18 ( 31.0%)
  grab      :   8 ( 13.8%)
  point     :   6 ( 10.3%)
  peace_sign:   6 ( 10.3%)

Confidence Statistics:

  flat_hand:
    Count:   20
    Mean:    0.959
    Std:     0.032
    Min:     0.873
    Max:     0.988
    Median:  0.968

  grab:
    Count:   8
    Mean:    0.942
    Std:     0.008
    Min:     0.935
    Max:     0.962
    Median:  0.939

  fist:
    Count:   18
    Mean:    0.947
    Std:     0.044
    Min:     0.880
    Max:     0.994
    Median:  0.950

  point:
    Count:   6
    Mean:    0.914
    Std:     0.042
    Min:     0.867
    Max:     0.972
    Median:  0.907

  peace_sign:
    Count:   6
    Mean:    0.970
    Std:     0.029
    Min:     0.907
    Max:     0.988
    Median:  0.983

------------------------------------------------------------
Overall Mean Confidence: 0.949
Overall Std Confidence:  0.038
Total Predictions:       58
Unknown Predictions:     0 (0.0%)

Generating plots.
Class distribution saved: kpc3_results\class_distribution.png
Confidence distribution saved: kpc3_results\confidence_distribution.png
Confidence timeline saved: kpc3_results\confidence_timeline.png
Confusion matrix saved: kpc3_results\confusion_matrix.png

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

        fist       0.56      1.00      0.71        10
   flat_hand       1.00      0.95      0.98        21
        grab       1.00      1.00      1.00         8
  peace_sign       0.83      1.00      0.91         5
       point       1.00      0.43      0.60        14

    accuracy                           0.84        58
   macro avg       0.88      0.88      0.84        58
weighted avg       0.91      0.84      0.84        58


Overall Accuracy: 84.48%

============================================================
EVALUATION COMPLETE
============================================================
Results saved to: kpc3_results

PS D:\ECE 1896\ECE-1896-NEURA-Glove\src\week6-test> python realtime_inference.py models/flex_to_rotation_model.pth
Loading model from: models/flex_to_rotation_model.pth
  Config: Input(5) -> [64, 128, 128, 64] -> Output(10)
Model loaded successfully
Kalman filtering enabled (Q=0.005, R=0.08)
UDP socket created for 127.0.0.1:5555

============================================================
REAL-TIME INFERENCE - FINAL FIX
============================================================
Streaming to Unity at 127.0.0.1:5555
Kalman filtering: ENABLED
Pose templates: ['flat_hand', 'fist', 'grab', 'peace_sign', 'point']

Pose-specific logic:
  Fist: Aggressive curl (1.8x scaling)
  Point: Index straight (5%), others curl (1.5x)
  Flat Hand: All joints capped to 5° max
  Peace Sign: Index+Middle straight (10%), Thumb+Ring+Pinky curl (1.3x)
  Smooth transitions: Pose blending (15 frames) + EMA temporal smoothing (α=0.3)

Fixes applied:
  Live IMU wrist orientation (not hardcoded)
  Proper Ctrl+C handling (saves log)
  Pinky curl direction
  Performance optimization

Scanning for ESP32...
Found: ESP32-BLE (88:57:21:86:3E:02)
Connecting...
Connected: True
Subscribed to notifications

STREAMING TO UNITY
Using LIVE IMU data for wrist orientation
Press Ctrl+C to stop and save log

Frame 10 | FPS: 10.9 | Kalman: ON | Pose: flat_hand (95.1%)
Frame 20 | FPS: 10.3 | Kalman: ON | Pose: flat_hand (97.1%)
Frame 30 | FPS: 10.3 | Kalman: ON | Pose: flat_hand (97.2%)
Frame 40 | FPS: 10.2 | Kalman: ON | Pose: flat_hand (97.0%)
Frame 50 | FPS: 10.1 | Kalman: ON | Pose: flat_hand (96.8%)
Frame 60 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (96.7%)
Frame 70 | FPS: 10.1 | Kalman: ON | Pose: flat_hand (96.5%)
Frame 80 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (96.4%)
Frame 90 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (96.3%)
Frame 100 | FPS: 10.1 | Kalman: ON | Pose: flat_hand (94.6%)
Frame 110 | FPS: 10.0 | Kalman: ON | Pose: grab (96.2%)
Frame 120 | FPS: 10.0 | Kalman: ON | Pose: grab (96.2%)
Frame 130 | FPS: 10.1 | Kalman: ON | Pose: grab (94.4%)
Frame 140 | FPS: 10.0 | Kalman: ON | Pose: grab (94.1%)
Frame 150 | FPS: 10.0 | Kalman: ON | Pose: grab (93.6%)
Frame 160 | FPS: 10.0 | Kalman: ON | Pose: grab (93.6%)
Frame 170 | FPS: 10.0 | Kalman: ON | Pose: grab (93.8%)
Frame 180 | FPS: 10.0 | Kalman: ON | Pose: grab (93.5%)
Frame 190 | FPS: 10.0 | Kalman: ON | Pose: grab (94.5%)
Frame 200 | FPS: 10.0 | Kalman: ON | Pose: fist (97.6%)
Frame 210 | FPS: 10.0 | Kalman: ON | Pose: fist (99.4%)
Frame 220 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 230 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 240 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 250 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 260 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 270 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 280 | FPS: 10.0 | Kalman: ON | Pose: fist (99.2%)
Frame 290 | FPS: 10.0 | Kalman: ON | Pose: point (92.5%)
Frame 300 | FPS: 10.0 | Kalman: ON | Pose: point (96.3%)
Frame 310 | FPS: 10.0 | Kalman: ON | Pose: point (97.2%)
Frame 320 | FPS: 10.0 | Kalman: ON | Pose: point (92.6%)
Frame 330 | FPS: 10.0 | Kalman: ON | Pose: point (88.8%)
Frame 340 | FPS: 10.0 | Kalman: ON | Pose: grab (86.7%)
Frame 350 | FPS: 10.0 | Kalman: ON | Pose: grab (87.0%)
Frame 360 | FPS: 10.0 | Kalman: ON | Pose: fist (88.0%)
Frame 370 | FPS: 10.0 | Kalman: ON | Pose: fist (89.3%)
Frame 380 | FPS: 10.0 | Kalman: ON | Pose: fist (89.7%)
Frame 390 | FPS: 10.0 | Kalman: ON | Pose: fist (91.9%)
Frame 400 | FPS: 10.0 | Kalman: ON | Pose: fist (92.2%)
Frame 410 | FPS: 10.0 | Kalman: ON | Pose: fist (89.1%)
Frame 420 | FPS: 10.0 | Kalman: ON | Pose: fist (90.4%)
Frame 430 | FPS: 10.0 | Kalman: ON | Pose: fist (90.6%)
Frame 440 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (87.3%)
Frame 450 | FPS: 10.0 | Kalman: ON | Pose: peace_sign (97.3%)
Frame 460 | FPS: 10.0 | Kalman: ON | Pose: peace_sign (98.1%)
Frame 470 | FPS: 10.0 | Kalman: ON | Pose: peace_sign (98.6%)
Frame 480 | FPS: 10.0 | Kalman: ON | Pose: peace_sign (98.8%)
Frame 490 | FPS: 10.0 | Kalman: ON | Pose: peace_sign (98.6%)
Frame 500 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (90.7%)
Frame 510 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (97.5%)
Frame 520 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (98.0%)
Frame 530 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (98.6%)
Frame 540 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (98.8%)
Frame 550 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (98.7%)
Frame 560 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (98.4%)
Frame 570 | FPS: 10.0 | Kalman: ON | Pose: flat_hand (88.1%)


Interrupt signal received, shutting down gracefully...

============================================================
SHUTDOWN
============================================================
Total frames sent: 571
Duration: 57.4s
Average FPS: 9.9

Saving predictions log...

Predictions log saved: predictions_log.json
  Total predictions: 58
```
