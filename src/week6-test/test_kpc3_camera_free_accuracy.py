# test_kpc3_camera_free_operation.py
"""
KPC3 – Camera-Free Operation Accuracy Test

Requires:
  - predictions_log.json  (from realtime_inference.py)
  - ground_truth.json     (filled in using the template from evaluate_model.py)

This script:
  - Aligns predictions and labels by frame
  - Computes accuracy
  - Prints whether KPC3 accuracy target (>= 80%) is met
"""

from pathlib import Path
import argparse

from sklearn.metrics import accuracy_score
from evaluate_model import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="KPC3 – Camera-free operation accuracy test")
    parser.add_argument("predictions", help="predictions_log.json produced by realtime_inference.py")
    parser.add_argument("ground_truth", help="Ground truth labels JSON")
    parser.add_argument("--output-dir", default="kpc3_results", help="Directory to save plots")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.predictions, args.ground_truth)

    # Use the same alignment logic as in evaluate_model
    y_true, y_pred = evaluator.align_with_ground_truth()
    if not y_true:
        print("ERROR: No aligned samples; check your ground_truth.json.")
        return

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print("KPC3 – CAMERA-FREE OPERATION ACCURACY")
    print("=" * 60)
    print(f"Number of labeled frames: {len(y_true)}")
    print(f"Overall classification accuracy: {acc:.2%}")
    print(f"KPC3 target (>= 80%) met? {'PASS' if acc >= 0.80 else 'FAIL'}")

    # Also generate full evaluation plots, including confusion matrix
    evaluator.run_full_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
