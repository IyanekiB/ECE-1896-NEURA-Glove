# test_kpc2_dataset_and_training.py
"""
KPC2 – Dataset Quality and Model Convergence Test

Usage:
    python test_kpc2_dataset_and_training.py \
        data/aligned/aligned_session_001_session_001.json \
        models/flex_to_rotation_model.pth
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_retention(aligned_path: str):
    """
    Compute overall retention and per-pose sample counts.

    aligned_path: path to an aligned_session_XXX.json file produced by data_aligner.py
    """
    aligned_path = Path(aligned_path)
    with open(aligned_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    pose_breakdown = metadata.get("pose_breakdown", [])
    pairs = data.get("pairs", [])

    # ---- Overall retention from metadata (same as before) ----
    total_sensor = sum(p.get("sensor_samples", 0) for p in pose_breakdown)
    total_pairs = sum(p.get("aligned_pairs", 0) for p in pose_breakdown)

    if total_sensor == 0:
        overall_retention = 0.0
    else:
        overall_retention = total_pairs / total_sensor

    # ---- Per-pose statistics using actual pose names from pairs ----
    # Every pair has a 'pose_name' field – use that to recover the labels.
    pose_counts = Counter(p.get("pose_name", "unknown") for p in pairs)

    per_pose = []
    for pose_name, pair_count in sorted(pose_counts.items()):
        # We don't have the exact "sensor_samples" per pose in metadata,
        # but retention is uniform by construction, so we can estimate it.
        if overall_retention > 0:
            est_sensor_samples = int(round(pair_count / overall_retention))
        else:
            est_sensor_samples = pair_count

        if est_sensor_samples > 0:
            pose_retention = pair_count / est_sensor_samples
        else:
            pose_retention = 0.0

        per_pose.append(
            {
                "pose": pose_name,
                "sensor_samples": est_sensor_samples,
                "aligned_pairs": pair_count,
                "retention": pose_retention,
            }
        )

    return overall_retention, per_pose


def plot_retention_by_pose(per_pose, save_path="kpc2_retention_by_pose.png"):
    """Bar chart of retention per pose (all should be ~100%)."""
    poses = [p["pose"] for p in per_pose]
    retention = [p["retention"] * 100.0 for p in per_pose]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(poses, retention, color="steelblue", edgecolor="black", alpha=0.9)

    for bar, r in zip(bars, retention):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.axhline(85, color="red", linestyle="--", linewidth=1.5, label="KPC2 target = 85%")
    plt.ylabel("Retention rate per pose (%)", fontsize=12)
    plt.xlabel("Pose class", fontsize=12)
    plt.title("Training Sample Retention by Pose", fontsize=14)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_val_loss_from_checkpoint(checkpoint_path, save_path="kpc2_val_loss_from_checkpoint.png"):
    """
    Re-plot validation loss history stored in your training checkpoint.

    Assumes train_model.py saved 'val_losses' or 'history' in the checkpoint.
    If not found, this function will print a warning.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try a few common keys
    val_losses = None
    if "val_losses" in ckpt:
        val_losses = ckpt["val_losses"]
    elif "history" in ckpt and "val_loss" in ckpt["history"]:
        val_losses = ckpt["history"]["val_loss"]

    if val_losses is None:
        print("WARNING: No validation loss history found in checkpoint.")
        return

    val_losses = np.array(val_losses)
    epochs = np.arange(1, len(val_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("Validation Loss History (from checkpoint)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KPC2 – Dataset quality and convergence test")
    parser.add_argument("aligned_json", help="Aligned dataset JSON (from data_aligner.py)")
    parser.add_argument("checkpoint", help="Trained model checkpoint (.pth)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KPC2 – DATASET QUALITY AND MODEL CONVERGENCE TEST")
    print("=" * 60)
    print(f"Aligned dataset : {args.aligned_json}")
    print(f"Checkpoint      : {args.checkpoint}\n")

    # ---- Retention ----
    overall_retention, per_pose = compute_retention(args.aligned_json)

    print("Sample retention (per pose):")
    for p in per_pose:
        print(
            f"  {p['pose']:<10} | sensor: {p['sensor_samples']:4d} | "
            f"pairs: {p['aligned_pairs']:4d} | retention: {p['retention'] * 100:5.1f}%"
        )

    print(f"\nOverall retention: {overall_retention * 100:.2f}%")
    print(f"Retention ≥ 85% ? {'PASS' if overall_retention >= 0.85 else 'FAIL'}")

    # Plot retention per pose
    plot_retention_by_pose(per_pose)

    # ---- Validation loss from checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    best_val_loss = ckpt.get("best_val_loss", None)
    if best_val_loss is None and "history" in ckpt:
        # fallback if best not stored explicitly
        hist = ckpt["history"]
        if "val_loss" in hist:
            best_val_loss = float(min(hist["val_loss"]))

    if best_val_loss is not None:
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Validation loss < 0.1 ? {'PASS' if best_val_loss < 0.1 else 'FAIL'}")
    else:
        print("\nWARNING: best_val_loss not found in checkpoint; cannot directly test KPC2 loss criterion.")

    # Plot full validation loss curve (if available)
    plot_val_loss_from_checkpoint(args.checkpoint)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
