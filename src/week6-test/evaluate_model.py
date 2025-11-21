# evaluate_model.py
"""
Model Evaluation Script for KPC3 – Camera-Free Operation

Analyzes pose classification performance from predictions_log.json:
- Class distribution
- Confidence statistics
- Confidence over time
- (Optional) Confusion matrix & accuracy vs ground truth labels

Usage:
    # Basic evaluation (no ground truth, just distributions)
    python evaluate_model.py predictions_log.json

    # With ground truth labels
    python evaluate_model.py predictions_log.json --ground-truth ground_truth.json

    # Create a frame-based ground truth template from predictions_log.json
    python evaluate_model.py predictions_log.json --create-template
"""

import json
from pathlib import Path
from collections import Counter
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class ModelEvaluator:
    """Evaluate pose classification model performance."""

    def __init__(self, predictions_file, ground_truth_file=None):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file

        # ---- Load predictions ----
        print(f"Loading predictions from: {predictions_file}")
        with open(predictions_file, "r") as f:
            data = json.load(f)

        self.predictions = data["predictions"]
        self.metadata = data["metadata"]

        print(f"  Total predictions: {self.metadata.get('total_predictions', len(self.predictions))}")
        print(f"  Duration: {self.metadata.get('duration', 0):.2f}s")
        print(f"  Kalman enabled: {self.metadata.get('kalman_enabled', False)}")

        # ---- Load ground truth (optional) ----
        self.ground_truth_raw = None
        self.ground_truth_samples = None

        if ground_truth_file:
            print(f"\nLoading ground truth from: {ground_truth_file}")
            with open(ground_truth_file, "r") as f:
                self.ground_truth_raw = json.load(f)

            if isinstance(self.ground_truth_raw, dict) and "samples" in self.ground_truth_raw:
                self.ground_truth_samples = self.ground_truth_raw["samples"]
            elif isinstance(self.ground_truth_raw, list):
                self.ground_truth_samples = self.ground_truth_raw
            else:
                raise ValueError(
                    "Ground truth file must be either a list of samples or a dict with a 'samples' list."
                )

            print(f"  Ground truth samples: {len(self.ground_truth_samples)}")

    # ------------------------------------------------------------------ #
    # Basic stats
    # ------------------------------------------------------------------ #
    def get_class_distribution(self):
        poses = [p["pose"] for p in self.predictions]
        return Counter(poses)

    def get_confidence_stats(self):
        stats = {}
        pose_confidences = {}

        for pred in self.predictions:
            pose = pred["pose"]
            conf = pred["confidence"]
            pose_confidences.setdefault(pose, []).append(conf)

        for pose, confidences in pose_confidences.items():
            confidences = np.array(confidences)
            stats[pose] = {
                "count": len(confidences),
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "median": float(np.median(confidences)),
            }

        return stats

    # ------------------------------------------------------------------ #
    # Frame-aware ground truth alignment
    # ------------------------------------------------------------------ #
    def align_with_ground_truth(self):
        """
        Align predictions with ground truth.

        Supports two formats of ground truth samples:
        - Frame-based: { "frame": 120, "true_pose": "fist" }
        - Time-based:  { "timestamp": 1234567890.1, "true_pose": "fist" }

        If 'frame' is present, we use exact frame matching.
        Otherwise we fall back to nearest-timestamp (within 0.5s).
        """
        if not self.ground_truth_samples:
            return None, None

        # Build an index of predictions by frame for fast lookups
        frame_index = {}
        for pred in self.predictions:
            frame = pred.get("frame", None)
            if frame is not None:
                frame_index[frame] = pred

        y_true, y_pred = [], []

        for gt in self.ground_truth_samples:
            if "true_pose" not in gt:
                continue

            gt_label = gt["true_pose"]

            # Prefer frame-based alignment if 'frame' field is available
            if "frame" in gt:
                frame = gt["frame"]
                pred = frame_index.get(frame, None)
                if pred is None:
                    # No prediction for that frame
                    continue
                y_true.append(gt_label)
                y_pred.append(pred["pose"])
            else:
                # Fallback: timestamp-based alignment
                if "timestamp" not in gt:
                    continue
                gt_time = gt["timestamp"]

                closest_pred = min(
                    self.predictions,
                    key=lambda p: abs(p["timestamp"] - gt_time),
                )

                if abs(closest_pred["timestamp"] - gt_time) < 0.5:
                    y_true.append(gt_label)
                    y_pred.append(closest_pred["pose"])

        return y_true, y_pred

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def plot_class_distribution(self, save_path=None):
        dist = self.get_class_distribution()

        plt.figure(figsize=(10, 6))
        poses = list(dist.keys())
        counts = list(dist.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(poses)))

        bars = plt.bar(poses, counts, color=colors, alpha=0.8, edgecolor="black")

        for bar, cnt in zip(bars, counts):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{cnt}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.xlabel("Pose Class", fontsize=14)
        plt.ylabel("Number of Predictions", fontsize=14)
        plt.title("Class Distribution of Pose Predictions", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Class distribution saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confidence_distribution(self, save_path=None):
        stats = self.get_confidence_stats()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        poses = list(stats.keys())
        confidences_by_pose = {}
        for pred in self.predictions:
            pose = pred["pose"]
            confidences_by_pose.setdefault(pose, []).append(pred["confidence"])

        axes[0].boxplot([confidences_by_pose[p] for p in poses], labels=poses)
        axes[0].set_xlabel("Pose Class", fontsize=12)
        axes[0].set_ylabel("Confidence Score", fontsize=12)
        axes[0].set_title("Confidence Distribution by Class (Box Plot)", fontsize=14, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)

        # Mean ± std bar chart
        means = [stats[p]["mean"] for p in poses]
        stds = [stats[p]["std"] for p in poses]
        x = np.arange(len(poses))
        bars = axes[1].bar(x, means, yerr=stds, capsize=5, alpha=0.9, edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(poses, rotation=45, ha="right")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_xlabel("Pose Class", fontsize=12)
        axes[1].set_ylabel("Mean Confidence ± Std", fontsize=12)
        axes[1].set_title("Mean Confidence by Class", fontsize=14, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confidence distribution saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confidence_timeline(self, save_path=None):
        timestamps = [p["timestamp"] for p in self.predictions]
        confidences = [p["confidence"] for p in self.predictions]
        poses = [p["pose"] for p in self.predictions]

        timestamps = np.array(timestamps) - timestamps[0]

        plt.figure(figsize=(14, 6))
        unique_poses = sorted(list(set(poses)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_poses)))
        pose_colors = {pose: colors[i] for i, pose in enumerate(unique_poses)}

        for pose in unique_poses:
            mask = np.array(poses) == pose
            plt.scatter(
                timestamps[mask],
                np.array(confidences)[mask],
                label=pose,
                color=pose_colors[pose],
                alpha=0.6,
                s=20,
            )

        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Confidence Score", fontsize=12)
        plt.title("Confidence Scores Over Time", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right", framealpha=0.9)
        plt.grid(alpha=0.3)
        plt.ylim([0.0, 1.05])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confidence timeline saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, save_path=None):
        y_true, y_pred = self.align_with_ground_truth()
        if not y_true:
            print("No aligned predictions found; cannot plot confusion matrix.")
            return

        classes = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Proportion"},
            linewidths=0.5,
            linecolor="gray",
        )

        plt.xlabel("Predicted Pose", fontsize=14)
        plt.ylabel("True Pose", fontsize=14)
        plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved: {save_path}")
        else:
            plt.show()

        plt.close()

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, labels=classes))

        acc = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {acc:.2%}")

    # ------------------------------------------------------------------ #
    def print_summary(self):
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        dist = self.get_class_distribution()
        print("\nClass Distribution:")
        for pose, count in dist.most_common():
            pct = 100.0 * count / len(self.predictions)
            print(f"  {pose:<10}: {count:3d} ({pct:5.1f}%)")

        stats = self.get_confidence_stats()
        print("\nConfidence Statistics:")
        for pose, s in stats.items():
            print(f"\n  {pose}:")
            print(f"    Count:   {s['count']}")
            print(f"    Mean:    {s['mean']:.3f}")
            print(f"    Std:     {s['std']:.3f}")
            print(f"    Min:     {s['min']:.3f}")
            print(f"    Max:     {s['max']:.3f}")
            print(f"    Median:  {s['median']:.3f}")

        all_conf = [p["confidence"] for p in self.predictions]
        print("\n" + "-" * 60)
        print(f"Overall Mean Confidence: {np.mean(all_conf):.3f}")
        print(f"Overall Std Confidence:  {np.std(all_conf):.3f}")
        print(f"Total Predictions:       {len(self.predictions)}")
        print(f"Unknown Predictions:     {dist.get('unknown', 0)} "
              f"({dist.get('unknown', 0) / len(self.predictions) * 100:.1f}%)")

    def run_full_evaluation(self, output_dir="evaluation_results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60)

        self.print_summary()

        print("\nGenerating plots.")
        self.plot_class_distribution(output_dir / "class_distribution.png")
        self.plot_confidence_distribution(output_dir / "confidence_distribution.png")
        self.plot_confidence_timeline(output_dir / "confidence_timeline.png")

        if self.ground_truth_samples:
            self.plot_confusion_matrix(output_dir / "confusion_matrix.png")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")


# ---------------------------------------------------------------------- #
# Ground truth template helper
# ---------------------------------------------------------------------- #
def create_ground_truth_template(predictions_file=None, output_file="ground_truth_template.json"):
    """
    Create a frame-based ground truth template.

    If predictions_file is provided, we include each prediction's frame and
    timestamp so you can simply fill in 'true_pose' for each entry.
    """
    samples = []

    if predictions_file is not None:
        try:
            with open(predictions_file, "r") as f:
                data = json.load(f)
            for pred in data["predictions"]:
                samples.append(
                    {
                        "frame": pred.get("frame"),
                        "timestamp": pred.get("timestamp"),
                        "true_pose": "UNKNOWN",  # fill this in manually
                    }
                )
        except Exception as e:
            print(f"WARNING: Could not read {predictions_file} for template generation: {e}")

    if not samples:
        # Fallback example if predictions_file not available
        samples = [
            {"frame": 0, "timestamp": 1234567890.123, "true_pose": "fist"},
            {"frame": 10, "timestamp": 1234567890.456, "true_pose": "flat_hand"},
        ]

    template = {
        "metadata": {
            "description": "Ground truth labels for pose predictions",
            "instructions": (
                "For each sample, set true_pose to one of: "
                "'fist', 'flat_hand', 'grab', 'peace_sign', 'point'. "
                "Alignment is done by 'frame' when available."
            ),
        },
        "samples": samples,
    }

    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Ground truth template created: {output_file}")


# ---------------------------------------------------------------------- #
def main():
    # Template-only mode
    if "--create-template" in sys.argv:
        # If a predictions file is also provided, use it
        predictions_file = None
        # e.g. python evaluate_model.py predictions_log.json --create-template
        if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
            predictions_file = sys.argv[1]
        create_ground_truth_template(predictions_file)
        return

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python evaluate_model.py <predictions_log.json> [options]\n")
        print("Options:")
        print("  --ground-truth <file>  : Path to ground truth labels file")
        print("  --output-dir <dir>     : Directory for saving plots (default: evaluation_results)")
        print("  --create-template      : Create ground truth template file")
        sys.exit(1)

    predictions_file = sys.argv[1]
    ground_truth_file = None
    output_dir = "evaluation_results"

    if "--ground-truth" in sys.argv:
        idx = sys.argv.index("--ground-truth")
        ground_truth_file = sys.argv[idx + 1]

    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        output_dir = sys.argv[idx + 1]

    evaluator = ModelEvaluator(predictions_file, ground_truth_file)
    evaluator.run_full_evaluation(output_dir)


if __name__ == "__main__":
    main()
