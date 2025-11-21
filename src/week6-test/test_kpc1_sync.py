# test_kpc1_sync.py
"""
KPC1 – Synchronization Quality (frame-by-frame version)

Measures how well sensor and camera recordings are time-aligned by comparing
each sensor sample timestamp to the nearest camera sample timestamp, *within
the time window where both streams overlap*.

Usage:
  python test_kpc1_sync.py <sensor_session_dir> <camera_session_dir> [--max-gap 0.1]

Example:
  python test_kpc1_sync.py data/sensor_recordings/session_001 \
                           data/camera_recordings/session_001 \
                           --max-gap 0.05   # 50 ms target
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_pose_json(session_dir: Path, pose_name: str, kind: str):
    """
    Load sensor or camera JSON for one pose in a session.

    session_dir / <pose_name> / ("sensor_data.json" | "camera_data.json")

    kind: 'sensor' or 'camera'
    """
    if kind == "sensor":
        fname = session_dir / pose_name / "sensor_data.json"
    else:
        fname = session_dir / pose_name / "camera_data.json"

    if not fname.exists():
        print(f"[WARN] Missing {kind} file for pose '{pose_name}': {fname}")
        return None

    with open(fname, "r") as f:
        data = json.load(f)

    return data["samples"]


def compute_time_diffs(sensor_samples, camera_samples):
    """
    Compute |Δt| between each sensor timestamp and the nearest camera timestamp.

    Important details:
      * We treat timestamps as *relative within the pose* (they already are,
        but we re-normalize to be safe).
      * We only evaluate over the overlapping time window where both sensor
        and camera have data. This mirrors how the aligner builds training
        pairs (it never uses parts where only one stream exists).
    """
    # Extract raw timestamps (seconds)
    sensor_ts = np.array([s["timestamp"] for s in sensor_samples], dtype=np.float64)
    cam_ts = np.array([c["timestamp"] for c in camera_samples], dtype=np.float64)

    if len(sensor_ts) == 0 or len(cam_ts) == 0:
        return np.array([])

    # Re-normalize both streams to start at 0 (just to be explicit)
    sensor_ts -= sensor_ts[0]
    cam_ts -= cam_ts[0]

    # Overlapping window where *both* streams have data
    overlap_start = max(sensor_ts[0], cam_ts[0])
    overlap_end = min(sensor_ts[-1], cam_ts[-1])

    if overlap_end <= overlap_start:
        # No meaningful overlap – return empty; caller will skip this pose
        return np.array([])

    # Keep only sensor samples inside the overlap window
    mask = (sensor_ts >= overlap_start) & (sensor_ts <= overlap_end)
    sensor_ts_overlap = sensor_ts[mask]

    if len(sensor_ts_overlap) == 0:
        return np.array([])

    # For each sensor timestamp, find nearest camera timestamp inside camera stream
    diffs = []
    for t in sensor_ts_overlap:
        nearest = np.min(np.abs(cam_ts - t))
        diffs.append(nearest)

    return np.array(diffs, dtype=np.float64)


def evaluate_sync(sensor_session_dir, camera_session_dir, target_gap_sec):
    sensor_session_dir = Path(sensor_session_dir)
    camera_session_dir = Path(camera_session_dir)

    # Poses = subdirectories under sensor session
    pose_dirs = [d for d in sensor_session_dir.iterdir() if d.is_dir()]
    pose_names = sorted(d.name for d in pose_dirs)

    all_diffs = []
    pose_stats = []

    print("\n" + "=" * 60)
    print("KPC1 – FRAME-BY-FRAME SYNCHRONIZATION TEST")
    print("=" * 60)
    print(f"Sensor session : {sensor_session_dir}")
    print(f"Camera session : {camera_session_dir}")
    print(f"Max acceptable gap (KPC1 target) : {target_gap_sec * 1000:.1f} ms\n")

    for pose in pose_names:
        sensor_samples = load_pose_json(sensor_session_dir, pose, "sensor")
        camera_samples = load_pose_json(camera_session_dir, pose, "camera")
        if sensor_samples is None or camera_samples is None:
            continue

        diffs = compute_time_diffs(sensor_samples, camera_samples)
        if diffs.size == 0:
            print(f"[WARN] Pose '{pose}': no overlapping samples, skipping.")
            continue

        all_diffs.append(diffs)

        pose_stats.append(
            {
                "pose": pose,
                "n_sensor": len(sensor_samples),
                "n_camera": len(camera_samples),
                "n_eval": len(diffs),
                "mean_gap": float(diffs.mean()),
                "std_gap": float(diffs.std()),
                "max_gap": float(diffs.max()),
                "within_target": float((diffs <= target_gap_sec).sum()) / len(diffs),
            }
        )

    if not all_diffs:
        print("No poses with overlapping data to evaluate.")
        return

    all_diffs = np.concatenate(all_diffs)

    # Print per-pose stats
    print("Per-pose synchronization stats (overlapping region only):")
    for s in pose_stats:
        print(
            f"  {s['pose']:12s} | "
            f"eval samples: {s['n_eval']:4d} | "
            f"mean: {s['mean_gap'] * 1000:6.2f} ms | "
            f"std: {s['std_gap'] * 1000:6.2f} ms | "
            f"max: {s['max_gap'] * 1000:6.2f} ms | "
            f"within target: {s['within_target'] * 100:5.1f}%"
        )

    # Overall stats
    mean_gap = float(all_diffs.mean())
    std_gap = float(all_diffs.std())
    max_gap_obs = float(all_diffs.max())
    within = float((all_diffs <= target_gap_sec).sum()) / len(all_diffs)

    print("\nOverall synchronization stats (all poses, overlapping region):")
    print(f"  Total eval samples : {len(all_diffs)}")
    print(f"  Mean gap           : {mean_gap * 1000:.2f} ms")
    print(f"  Std gap            : {std_gap * 1000:.2f} ms")
    print(f"  Max gap (observed) : {max_gap_obs * 1000:.2f} ms")
    print(f"  Within target      : {within * 100:.1f}% of samples")

    # PASS / FAIL vs KPC1 target
    pass_mean = mean_gap <= target_gap_sec
    pass_std = std_gap <= target_gap_sec

    print("\nKPC1 verdict:")
    print(f"  Mean gap <= target? {'PASS' if pass_mean else 'FAIL'}")
    print(f"  Std  gap <= target? {'PASS' if pass_std else 'FAIL'}")

    # Histogram figure
    plt.figure(figsize=(8, 5))
    plt.hist(all_diffs * 1000, bins=30, alpha=0.8, edgecolor="black")
    plt.axvline(
        target_gap_sec * 1000,
        color="red",
        linestyle="--",
        label=f"Target = {target_gap_sec * 1000:.1f} ms",
    )
    plt.xlabel("Absolute time difference |Δt| (ms)")
    plt.ylabel("Count")
    plt.title("Sensor–Camera Timestamp Gap Distribution (All Poses, Overlap Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kpc1_sync_histogram.png", dpi=300)
    print("\nSaved: kpc1_sync_histogram.png")


def main():
    parser = argparse.ArgumentParser(description="KPC1 frame-by-frame sync test")
    parser.add_argument("sensor_session_dir", help="Directory of sensor recordings for one session")
    parser.add_argument("camera_session_dir", help="Directory of camera recordings for one session")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.05,  # 50 ms target
        help="KPC1 target in seconds (default: 0.05 = 50 ms)",
    )
    args = parser.parse_args()

    evaluate_sync(args.sensor_session_dir, args.camera_session_dir, args.max_gap)


if __name__ == "__main__":
    main()
