"""
Compare MediaPipe Ground Truth vs Unity Hand Tracking Accuracy

PURPOSE:
Compare joint angles from MediaPipe (ground truth) with Unity's rendered hand poses
to measure accuracy of the glove → inference → Unity pipeline.

INPUTS:
- mediapipe_snapshots.json: Ground truth from MediaPipe webcam capture
- unity_test_snapshots/*.json: Unity hand angles from TestDataLogger.cs

OUTPUT:
- Console output with comparison tables and per-joint RMSE
- CSV files for each comparison:
  * {unity_file}_per_joint_summary.csv: Aggregated per-joint RMSE statistics
  * {unity_file}_detailed_comparison.csv: Full per-snapshot joint comparison
  * {unity_file}_overall_metrics.csv: Overall accuracy metrics

USAGE:
# Process all Unity snapshot files in directory:
python compare_mediapipe_unity.py

# Process a single Unity snapshot file:
python compare_mediapipe_unity.py --unity unity_test_snapshots/unity_test_snapshots_t6_p05_m02.json

# Specify custom paths:
python compare_mediapipe_unity.py --mediapipe FILE --unity-dir DIR --output-dir OUTPUT
"""

import json
import numpy as np
import argparse
import csv
from pathlib import Path
from datetime import datetime


class MediaPipeUnityComparison:
    """Compare MediaPipe ground truth with Unity hand tracking"""

    def __init__(self, mediapipe_file='mediapipe_snapshots.json',
                 unity_file='unity_test_snapshots.json',
                 output_dir='comparison_results'):
        self.mediapipe_file = mediapipe_file
        self.unity_file = unity_file
        self.output_dir = Path(output_dir)

        self.mediapipe_data = None
        self.unity_data = None

        # RMSE thresholds
        self.EXCELLENT_THRESHOLD = 8.0  # < 8° is excellent
        self.GOOD_THRESHOLD = 10.0      # < 10° is good
        self.ACCEPTABLE_THRESHOLD = 15.0  # < 15° is acceptable
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load both MediaPipe and Unity snapshot files"""
        print("Loading data files...")

        # Load MediaPipe ground truth
        try:
            with open(self.mediapipe_file, 'r') as f:
                self.mediapipe_data = json.load(f)
            print(f"✓ MediaPipe: {len(self.mediapipe_data['snapshots'])} snapshots")
        except FileNotFoundError:
            print(f"✗ ERROR: MediaPipe file not found: {self.mediapipe_file}")
            return False
        except Exception as e:
            print(f"✗ ERROR loading MediaPipe file: {e}")
            return False

        # Load Unity data
        try:
            with open(self.unity_file, 'r') as f:
                self.unity_data = json.load(f)
            print(f"✓ Unity: {len(self.unity_data['snapshots'])} snapshots")
        except FileNotFoundError:
            print(f"✗ ERROR: Unity file not found: {self.unity_file}")
            return False
        except Exception as e:
            print(f"✗ ERROR loading Unity file: {e}")
            return False

        return True

    def align_snapshots(self):
        """
        Align MediaPipe and Unity snapshots by order

        Assumes snapshots are captured in the same sequence:
        Pose 1 (MediaPipe snapshot 0, Unity snapshot 0)
        Pose 2 (MediaPipe snapshot 1, Unity snapshot 1)
        etc.

        Returns list of (mediapipe_snap, unity_snap) tuples
        """
        mp_snaps = self.mediapipe_data['snapshots']
        unity_snaps = self.unity_data['snapshots']

        # Align by snapshot index (assuming same order)
        aligned = []
        count = min(len(mp_snaps), len(unity_snaps))

        for i in range(count):
            aligned.append((mp_snaps[i], unity_snaps[i]))

        if len(mp_snaps) != len(unity_snaps):
            print(f"⚠ Warning: Snapshot count mismatch (MediaPipe: {len(mp_snaps)}, Unity: {len(unity_snaps)})")
            print(f"  Using first {count} snapshots from each")

        print(f"✓ Aligned {count} snapshot pairs")
        return aligned

    def calculate_joint_rmse(self, mp_angle, unity_angle):
        """Calculate RMSE between two angle values"""
        diff = mp_angle - unity_angle
        return np.sqrt(diff ** 2)

    def compare_finger(self, mp_finger, unity_finger, finger_name):
        """
        Compare angles for a single finger

        Returns dict with per-joint RMSE
        """
        results = {}

        # MediaPipe provides: mcp, pip, dip
        # Unity provides: metacarpal, proximal, intermediate, distal, tip

        # Map MediaPipe joints to Unity joints:
        # MediaPipe MCP → Unity Proximal (knuckle)
        # MediaPipe PIP → Unity Intermediate
        # MediaPipe DIP → Unity Distal

        joints_to_compare = [
            ('mcp', 'proximal'),
            ('pip', 'intermediate'),
            ('dip', 'distal')
        ]

        for mp_joint, unity_joint in joints_to_compare:
            mp_angle = mp_finger.get(mp_joint, 0)
            unity_angle = unity_finger.get(unity_joint, 0)

            rmse = self.calculate_joint_rmse(mp_angle, unity_angle)
            results[unity_joint] = {
                'mediapipe': mp_angle,
                'unity': unity_angle,
                'rmse': rmse
            }

        return results

    def compare_snapshot_pair(self, mp_snap, unity_snap):
        """Compare a single MediaPipe-Unity snapshot pair"""
        results = {}

        # Get joint angles from both
        mp_angles = mp_snap['joint_angles']
        unity_angles = unity_snap['joint_angles']

        # Compare each finger (excluding thumb for now - different joint structure)
        fingers = ['index', 'middle', 'ring', 'pinky']

        for finger in fingers:
            mp_finger = mp_angles.get(finger, {})
            unity_finger = unity_angles.get(finger, {})

            results[finger] = self.compare_finger(mp_finger, unity_finger, finger)

        return results

    def export_per_joint_summary_csv(self, all_results, filename):
        """Export per-joint RMSE summary to CSV"""
        fingers = ['index', 'middle', 'ring', 'pinky']
        joints = ['proximal', 'intermediate', 'distal']

        # Aggregate RMSE per joint
        joint_rmses = {finger: {joint: [] for joint in joints} for finger in fingers}

        for result in all_results:
            for finger in fingers:
                for joint in joints:
                    if joint in result[finger]:
                        joint_rmses[finger][joint].append(result[finger][joint]['rmse'])

        # Write to CSV
        csv_path = self.output_dir / filename
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Finger', 'Joint', 'Avg_RMSE_deg', 'Min_RMSE_deg', 'Max_RMSE_deg', 
                           'Std_RMSE_deg', 'Status'])

            for finger in fingers:
                for joint in joints:
                    rmses = joint_rmses[finger][joint]
                    if rmses:
                        avg = np.mean(rmses)
                        min_val = np.min(rmses)
                        max_val = np.max(rmses)
                        std_val = np.std(rmses)

                        # Determine status
                        if avg < self.EXCELLENT_THRESHOLD:
                            status = "Excellent"
                        elif avg < self.GOOD_THRESHOLD:
                            status = "Good"
                        elif avg < self.ACCEPTABLE_THRESHOLD:
                            status = "Acceptable"
                        else:
                            status = "Poor"

                        writer.writerow([finger, joint, f"{avg:.2f}", f"{min_val:.2f}", 
                                       f"{max_val:.2f}", f"{std_val:.2f}", status])

        print(f"✓ Saved per-joint summary: {csv_path}")

    def export_detailed_comparison_csv(self, all_results, aligned_pairs, filename):
        """Export detailed per-snapshot comparison to CSV"""
        fingers = ['index', 'middle', 'ring', 'pinky']
        joints = ['proximal', 'intermediate', 'distal']

        csv_path = self.output_dir / filename
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Snapshot_ID', 'Pose', 'Finger', 'Joint', 
                           'MediaPipe_deg', 'Unity_deg', 'RMSE_deg', 'Status'])

            for i, (result, (mp_snap, unity_snap)) in enumerate(zip(all_results, aligned_pairs)):
                pose_name = unity_snap.get('pose', 'unknown')

                for finger in fingers:
                    for joint in joints:
                        if joint in result[finger]:
                            data = result[finger][joint]
                            mp_angle = data['mediapipe']
                            unity_angle = data['unity']
                            rmse = data['rmse']

                            # Determine status
                            if rmse < self.EXCELLENT_THRESHOLD:
                                status = "Excellent"
                            elif rmse < self.GOOD_THRESHOLD:
                                status = "Good"
                            elif rmse < self.ACCEPTABLE_THRESHOLD:
                                status = "Acceptable"
                            else:
                                status = "Poor"

                            writer.writerow([i, pose_name, finger, joint, 
                                          f"{mp_angle:.2f}", f"{unity_angle:.2f}", 
                                          f"{rmse:.2f}", status])

        print(f"✓ Saved detailed comparison: {csv_path}")

    def export_overall_metrics_csv(self, all_results, filename):
        """Export overall accuracy metrics to CSV"""
        fingers = ['index', 'middle', 'ring', 'pinky']
        joints = ['proximal', 'intermediate', 'distal']

        # Calculate overall statistics
        overall_rmses = []
        for result in all_results:
            for finger in fingers:
                for joint in joints:
                    if joint in result[finger]:
                        overall_rmses.append(result[finger][joint]['rmse'])

        overall_avg = np.mean(overall_rmses)
        overall_max = np.max(overall_rmses)
        overall_min = np.min(overall_rmses)
        overall_std = np.std(overall_rmses)

        # Determine evaluation
        if overall_avg < self.EXCELLENT_THRESHOLD:
            evaluation = "Excellent"
        elif overall_avg < self.GOOD_THRESHOLD:
            evaluation = "Good"
        elif overall_avg < self.ACCEPTABLE_THRESHOLD:
            evaluation = "Acceptable"
        else:
            evaluation = "Poor"

        csv_path = self.output_dir / filename
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Average_RMSE_deg', f"{overall_avg:.2f}"])
            writer.writerow(['Max_RMSE_deg', f"{overall_max:.2f}"])
            writer.writerow(['Min_RMSE_deg', f"{overall_min:.2f}"])
            writer.writerow(['Std_RMSE_deg', f"{overall_std:.2f}"])
            writer.writerow(['Number_of_Snapshots', len(all_results)])
            writer.writerow(['Overall_Evaluation', evaluation])
            writer.writerow(['Excellent_Threshold_deg', self.EXCELLENT_THRESHOLD])
            writer.writerow(['Good_Threshold_deg', self.GOOD_THRESHOLD])
            writer.writerow(['Acceptable_Threshold_deg', self.ACCEPTABLE_THRESHOLD])

        print(f"✓ Saved overall metrics: {csv_path}")

    def run_comparison(self):
        """Main comparison workflow"""
        print("\n" + "="*60)
        print("  MEDIAPIPE vs UNITY ACCURACY COMPARISON")
        print("="*60 + "\n")

        # Load data
        if not self.load_data():
            return False

        # Align snapshots
        aligned_pairs = self.align_snapshots()

        if not aligned_pairs:
            print("✗ No snapshot pairs to compare")
            return False

        # Compare each pair
        print(f"\nComparing {len(aligned_pairs)} snapshot pairs...\n")

        all_results = []
        for i, (mp_snap, unity_snap) in enumerate(aligned_pairs):
            result = self.compare_snapshot_pair(mp_snap, unity_snap)
            all_results.append(result)

        # Calculate aggregate statistics and print summary
        self.print_summary(all_results, aligned_pairs)

        # Export to CSV
        print("\n" + "="*60)
        print("  EXPORTING TO CSV")
        print("="*60 + "\n")

        # Create filename prefix from unity file
        unity_basename = Path(self.unity_file).stem
        
        self.export_per_joint_summary_csv(all_results, f"{unity_basename}_per_joint_summary.csv")
        self.export_detailed_comparison_csv(all_results, aligned_pairs, f"{unity_basename}_detailed_comparison.csv")
        self.export_overall_metrics_csv(all_results, f"{unity_basename}_overall_metrics.csv")

        return True

    def print_summary(self, all_results, aligned_pairs):
        """Print summary statistics"""
        fingers = ['index', 'middle', 'ring', 'pinky']
        joints = ['proximal', 'intermediate', 'distal']

        print("="*60)
        print("  PER-JOINT RMSE SUMMARY")
        print("="*60)

        # Aggregate RMSE per joint
        joint_rmses = {finger: {joint: [] for joint in joints} for finger in fingers}

        for result in all_results:
            for finger in fingers:
                for joint in joints:
                    if joint in result[finger]:
                        joint_rmses[finger][joint].append(result[finger][joint]['rmse'])

        # Print table
        print(f"\n{'Finger':<10} {'Joint':<15} {'Avg RMSE':<12} {'Min':<8} {'Max':<8} {'Status'}")
        print("-" * 70)

        overall_rmses = []

        for finger in fingers:
            for joint in joints:
                rmses = joint_rmses[finger][joint]
                if rmses:
                    avg = np.mean(rmses)
                    min_val = np.min(rmses)
                    max_val = np.max(rmses)
                    overall_rmses.append(avg)

                    # Determine status
                    if avg < self.EXCELLENT_THRESHOLD:
                        status = "✓ Excellent"
                    elif avg < self.GOOD_THRESHOLD:
                        status = "✓ Good"
                    elif avg < self.ACCEPTABLE_THRESHOLD:
                        status = "⚠ Acceptable"
                    else:
                        status = "✗ Poor"

                    print(f"{finger:<10} {joint:<15} {avg:>6.2f}°      {min_val:>5.2f}°  {max_val:>5.2f}°  {status}")

        # Overall summary
        overall_avg = np.mean(overall_rmses)
        overall_max = np.max(overall_rmses)

        print("\n" + "="*60)
        print("  OVERALL ACCURACY")
        print("="*60)
        print(f"Average RMSE (all joints): {overall_avg:.2f}°")
        print(f"Maximum RMSE (worst joint): {overall_max:.2f}°")
        print(f"Number of snapshots compared: {len(all_results)}")

        # Pass/Fail evaluation
        print("\n" + "="*60)
        print("  EVALUATION")
        print("="*60)

        passed_excellent = overall_avg < self.EXCELLENT_THRESHOLD
        passed_good = overall_avg < self.GOOD_THRESHOLD
        passed_acceptable = overall_avg < self.ACCEPTABLE_THRESHOLD

        if passed_excellent:
            print(f"✓ EXCELLENT: Average RMSE {overall_avg:.2f}° < {self.EXCELLENT_THRESHOLD}°")
        elif passed_good:
            print(f"✓ GOOD: Average RMSE {overall_avg:.2f}° < {self.GOOD_THRESHOLD}°")
        elif passed_acceptable:
            print(f"⚠ ACCEPTABLE: Average RMSE {overall_avg:.2f}° < {self.ACCEPTABLE_THRESHOLD}°")
        else:
            print(f"✗ POOR: Average RMSE {overall_avg:.2f}° >= {self.ACCEPTABLE_THRESHOLD}°")

        print("="*60 + "\n")

        # Per-snapshot detailed breakdown
        print("="*60)
        print("  PER-SNAPSHOT DETAILED BREAKDOWN")
        print("="*60)

        for i, (result, (mp_snap, unity_snap)) in enumerate(zip(all_results, aligned_pairs)):
            print(f"\n--- Snapshot {i} ---")

            # Get pose name if available
            pose_name = unity_snap.get('pose', 'unknown')
            if pose_name and pose_name != 'unknown':
                print(f"Pose: {pose_name}")

            # Print per-finger, per-joint breakdown
            print(f"\n{'Finger':<10} {'Joint':<15} {'MediaPipe':<12} {'Unity':<12} {'RMSE':<10} {'Status'}")
            print("-" * 70)

            snapshot_rmses = []
            for finger in fingers:
                for joint in joints:
                    if joint in result[finger]:
                        data = result[finger][joint]
                        mp_angle = data['mediapipe']
                        unity_angle = data['unity']
                        rmse = data['rmse']
                        snapshot_rmses.append(rmse)

                        # Determine status
                        if rmse < self.EXCELLENT_THRESHOLD:
                            status = "✓ Excellent"
                        elif rmse < self.GOOD_THRESHOLD:
                            status = "✓ Good"
                        elif rmse < self.ACCEPTABLE_THRESHOLD:
                            status = "⚠ Acceptable"
                        else:
                            status = "✗ Poor"

                        print(f"{finger:<10} {joint:<15} {mp_angle:>6.2f}°      {unity_angle:>6.2f}°      {rmse:>5.2f}°    {status}")

            # Snapshot summary
            avg_rmse = np.mean(snapshot_rmses)
            max_rmse = np.max(snapshot_rmses)
            min_rmse = np.min(snapshot_rmses)

            print(f"\nSnapshot {i} Summary:")
            print(f"  Average RMSE: {avg_rmse:.2f}°")
            print(f"  Min RMSE: {min_rmse:.2f}°")
            print(f"  Max RMSE: {max_rmse:.2f}°")

        print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare MediaPipe ground truth with Unity hand tracking')
    parser.add_argument('--mediapipe', default='mediapipe_snapshots.json',
                       help='MediaPipe snapshots JSON file')
    parser.add_argument('--unity', default=None,
                       help='Unity snapshots JSON file (if not specified, processes all files in unity_test_snapshots/)')
    parser.add_argument('--unity-dir', default='unity_test_snapshots',
                       help='Directory containing Unity snapshot files')
    parser.add_argument('--output-dir', default='comparison_results',
                       help='Output directory for CSV files')

    args = parser.parse_args()

    # Determine which unity files to process
    unity_files = []
    
    if args.unity:
        # Single file mode
        unity_files = [args.unity]
    else:
        # Process all JSON files in unity_test_snapshots directory
        unity_dir = Path(args.unity_dir)
        if not unity_dir.exists():
            print(f"✗ ERROR: Unity snapshots directory not found: {unity_dir}")
            return 1
        
        unity_files = sorted(unity_dir.glob('*.json'))
        
        if not unity_files:
            print(f"✗ ERROR: No JSON files found in {unity_dir}")
            return 1
        
        print(f"\nFound {len(unity_files)} Unity snapshot files to process:\n")
        for f in unity_files:
            print(f"  - {f.name}")
        print()

    # Process each Unity file
    all_success = True
    
    for i, unity_file in enumerate(unity_files, 1):
        print("\n" + "█"*70)
        print(f"  PROCESSING FILE {i}/{len(unity_files)}: {Path(unity_file).name}")
        print("█"*70)
        
        comparator = MediaPipeUnityComparison(
            mediapipe_file=args.mediapipe,
            unity_file=str(unity_file),
            output_dir=args.output_dir
        )
        
        success = comparator.run_comparison()
        
        if not success:
            print(f"\n✗ Failed to process: {unity_file}")
            all_success = False
        else:
            print(f"\n✓ Successfully processed: {Path(unity_file).name}")

    # Final summary
    print("\n" + "="*70)
    print("  BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(unity_files)} file(s)")
    print(f"Output directory: {Path(args.output_dir).absolute()}")
    print()

    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
