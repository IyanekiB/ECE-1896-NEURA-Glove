"""
TRAINING DATA DIAGNOSTIC TOOL
Checks for common issues that cause model collapse

Run this FIRST to diagnose your data issues!
"""

import json
import numpy as np
from collections import Counter

def diagnose_training_data(dataset_path='data/training_dataset.json'):
    """Comprehensive diagnostics"""
    
    print("\n" + "="*70)
    print("TRAINING DATA DIAGNOSTIC REPORT")
    print("="*70)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"\n📊 Total Samples: {len(samples)}")
    
    # Check 1: Pose distribution
    print("\n" + "-"*70)
    print("CHECK 1: Pose Distribution")
    print("-"*70)
    pose_counts = Counter(s['pose_name'] for s in samples)
    for pose, count in sorted(pose_counts.items()):
        print(f"  {pose:15s}: {count:5d} samples")
    
    # Check if severely imbalanced
    max_count = max(pose_counts.values())
    min_count = min(pose_counts.values())
    if max_count / min_count > 3:
        print(f"\n  ⚠️  WARNING: Severe class imbalance! (ratio: {max_count/min_count:.1f}:1)")
        print(f"     This will cause pose classification to fail!")
    
    # Check 2: Sensor data variation
    print("\n" + "-"*70)
    print("CHECK 2: Sensor Data Variation")
    print("-"*70)
    
    all_flex = []
    all_imu_ori = []
    all_imu_accel = []
    all_imu_gyro = []
    
    for sample in samples[:1000]:  # Check first 1000
        all_flex.append(sample['flex_sensors'])
        all_imu_ori.append(sample['imu_orientation'])
        all_imu_accel.append(sample['imu_accel'])
        all_imu_gyro.append(sample['imu_gyro'])
    
    flex_std = np.std(all_flex, axis=0)
    imu_ori_std = np.std(all_imu_ori, axis=0)
    imu_accel_std = np.std(all_imu_accel, axis=0)
    imu_gyro_std = np.std(all_imu_gyro, axis=0)
    
    print(f"  Flex sensors STD:    {flex_std}")
    print(f"  IMU orientation STD: {imu_ori_std}")
    print(f"  IMU accel STD:       {imu_accel_std}")
    print(f"  IMU gyro STD:        {imu_gyro_std}")
    
    if np.mean(flex_std) < 0.01:
        print(f"\n  🚨 CRITICAL: Flex sensor variation TOO LOW!")
        print(f"     STD < 0.01 means synthetic data has NO variation!")
        print(f"     Model will collapse to mean prediction.")
    
    # Check 3: Joint data format and variation
    print("\n" + "-"*70)
    print("CHECK 3: Joint Data Format & Variation")
    print("-"*70)
    
    first_sample = samples[0]
    joints_data = first_sample['joints']
    
    print(f"  Joints data type: {type(joints_data)}")
    
    if isinstance(joints_data, list):
        print(f"  Number of joints: {len(joints_data)}")
        print(f"  First joint structure: {joints_data[0].keys() if joints_data else 'EMPTY'}")
        
        # Check position variation
        all_positions = []
        all_rotations = []
        for sample in samples[:1000]:
            for joint in sample['joints']:
                all_positions.append(joint['position'])
                all_rotations.append(joint['rotation'])
        
        pos_std = np.std(all_positions, axis=0)
        rot_std = np.std(all_rotations, axis=0)
        
        print(f"\n  Position STD: {pos_std}")
        print(f"  Rotation STD: {rot_std}")
        
        if np.all(pos_std == 0):
            print(f"\n  ⚠️  All positions are IDENTICAL (likely all [0,0,0])")
            print(f"     This is CORRECT for Unity format but explains pos_loss=0.0000")
        
        if np.mean(rot_std) < 0.01:
            print(f"\n  🚨 CRITICAL: Rotation variation TOO LOW!")
            print(f"     STD < 0.01 means all rotations are nearly identical!")
            print(f"     Model has nothing to learn!")
    
    elif isinstance(joints_data, dict):
        print(f"  Dict keys: {list(joints_data.keys())}")
        
        # Try to extract values
        all_values = []
        for sample in samples[:1000]:
            jdata = sample['joints']
            # Flatten all values
            def flatten_dict(d):
                vals = []
                for v in d.values():
                    if isinstance(v, dict):
                        vals.extend(flatten_dict(v))
                    elif isinstance(v, list):
                        vals.extend(v)
                    else:
                        vals.append(v)
                return vals
            all_values.append(flatten_dict(jdata))
        
        if all_values:
            values_std = np.std(all_values, axis=0)
            print(f"\n  Output values STD: {values_std[:10]}... (showing first 10)")
            
            if np.mean(values_std) < 0.001:
                print(f"\n  🚨 CRITICAL: Output variation EXTREMELY LOW!")
                print(f"     All outputs are nearly identical!")
    
    # Check 4: Dataset numbers (synthetic data check)
    print("\n" + "-"*70)
    print("CHECK 4: Synthetic Data Quality")
    print("-"*70)
    
    dataset_nums = Counter()
    for sample in samples:
        key = (sample['pose_name'], sample.get('dataset_number', 0))
        dataset_nums[key] += 1
    
    print(f"\n  Total unique (pose, dataset_num) pairs: {len(dataset_nums)}")
    print(f"  Top 10 dataset sizes:")
    for (pose, dnum), count in dataset_nums.most_common(10):
        is_synthetic = dnum > 10  # Assuming original datasets are 1-10
        marker = " [SYNTHETIC]" if is_synthetic else ""
        print(f"    ({pose}, dataset_{dnum}): {count} samples{marker}")
    
    # Check if synthetic datasets are identical to originals
    if len(dataset_nums) > 10:
        print(f"\n  ✓ Multiple datasets found (original + synthetic)")
        print(f"    Checking if synthetic data has variation...")
        
        # Compare a few samples from different datasets of same pose
        pose_samples = {}
        for sample in samples:
            pose = sample['pose_name']
            dnum = sample.get('dataset_number', 0)
            if pose not in pose_samples:
                pose_samples[pose] = {}
            if dnum not in pose_samples[pose]:
                pose_samples[pose][dnum] = []
            if len(pose_samples[pose][dnum]) < 5:
                pose_samples[pose][dnum].append(sample)
        
        # Check variation between datasets
        for pose, datasets in list(pose_samples.items())[:1]:  # Check first pose
            if len(datasets) >= 2:
                dnums = list(datasets.keys())[:2]
                samples1 = datasets[dnums[0]]
                samples2 = datasets[dnums[1]]
                
                flex1 = [s['flex_sensors'] for s in samples1]
                flex2 = [s['flex_sensors'] for s in samples2]
                
                diff = np.mean(np.abs(np.array(flex1) - np.array(flex2)))
                print(f"\n    Comparing dataset {dnums[0]} vs {dnums[1]} for '{pose}':")
                print(f"      Mean absolute difference in flex sensors: {diff:.6f}")
                
                if diff < 0.001:
                    print(f"      🚨 CRITICAL: Datasets are IDENTICAL!")
                    print(f"         Synthetic generation is NOT working!")
                elif diff < 0.01:
                    print(f"      ⚠️  WARNING: Datasets are very similar")
                    print(f"         Synthetic variation might be too small")
                else:
                    print(f"      ✓ Good variation between datasets")
    
    # Check 5: Frame numbers (temporal sequences)
    print("\n" + "-"*70)
    print("CHECK 5: Temporal Sequence Integrity")
    print("-"*70)
    
    frame_gaps = []
    for pose, dnum in list(dataset_nums.keys())[:5]:  # Check first 5
        pose_dataset_samples = [
            s for s in samples 
            if s['pose_name'] == pose and s.get('dataset_number', 0) == dnum
        ]
        pose_dataset_samples.sort(key=lambda x: x['frame_number'])
        
        if len(pose_dataset_samples) > 1:
            gaps = []
            for i in range(len(pose_dataset_samples) - 1):
                gap = pose_dataset_samples[i+1]['frame_number'] - pose_dataset_samples[i]['frame_number']
                gaps.append(gap)
            
            if gaps:
                frame_gaps.extend(gaps)
    
    if frame_gaps:
        print(f"  Frame number gaps: min={min(frame_gaps)}, max={max(frame_gaps)}, mean={np.mean(frame_gaps):.1f}")
        if max(frame_gaps) > 2:
            print(f"  ⚠️  WARNING: Large frame gaps detected!")
            print(f"     Sequences might not be temporally coherent")
    
    # FINAL DIAGNOSIS
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues_found = []
    
    # Check class imbalance
    if max_count / min_count > 3:
        issues_found.append("⚠️  SEVERE CLASS IMBALANCE → Will cause pose classification to fail")
    
    # Check sensor variation
    if np.mean(flex_std) < 0.01:
        issues_found.append("🚨 CRITICAL: NO SENSOR VARIATION → Model will collapse")
    
    # Check synthetic data
    if len(dataset_nums) > 10:
        # Already checked above
        pass
    else:
        issues_found.append("⚠️  Only a few datasets → Need more synthetic data")
    
    if not issues_found:
        print("\n✅ No critical issues found!")
        print("   Your training data looks good.")
    else:
        print("\n🚨 ISSUES FOUND:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
        
        print("\n" + "="*70)
        print("RECOMMENDED ACTIONS")
        print("="*70)
        print("\n1. If sensor variation is too low:")
        print("   → Increase noise_std in synthetic_data_generator.py")
        print("   → Regenerate synthetic data")
        print("\n2. If datasets are identical:")
        print("   → Check synthetic_data_generator.py is adding noise correctly")
        print("   → Verify random seed is NOT fixed")
        print("\n3. If class imbalance:")
        print("   → Collect more data for underrepresented poses")
        print("   → Or remove overrepresented pose samples")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training_dataset.json'
    
    try:
        diagnose_training_data(dataset_path)
    except FileNotFoundError:
        print(f"\n❌ File not found: {dataset_path}")
        print("Usage: python diagnose_data.py [path/to/training_dataset.json]")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()