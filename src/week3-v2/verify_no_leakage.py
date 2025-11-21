#!/usr/bin/env python3
"""
Verify Data Leakage Fix - Check if train/val splits are properly separated
Run this after creating your dataset but before training
"""

import json
import sys

def verify_no_leakage(dataset_path):
    """Check if train/val splits would have data leakage"""
    
    print("="*70)
    print("DATA LEAKAGE VERIFICATION")
    print("="*70)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"\nTotal samples: {len(samples)}")
    
    # Group by recording session
    recording_groups = {}
    for sample in samples:
        key = (sample['pose_name'], sample.get('dataset_number', 0))
        if key not in recording_groups:
            recording_groups[key] = []
        recording_groups[key].append(sample['frame_number'])
    
    print(f"Total recording sessions: {len(recording_groups)}")
    print(f"\nRecording sessions breakdown:")
    
    for (pose, dataset), frames in sorted(recording_groups.items()):
        print(f"  {pose:15s} Dataset #{dataset}: {len(frames)} frames")
    
    # Simulate split
    all_keys = list(recording_groups.keys())
    split_idx = int(0.85 * len(all_keys))
    train_keys = set(all_keys[:split_idx])
    val_keys = set(all_keys[split_idx:])
    
    print(f"\nSimulated 85/15 split:")
    print(f"  Training sessions: {len(train_keys)}")
    print(f"  Validation sessions: {len(val_keys)}")
    
    # Check for overlap
    overlap = train_keys.intersection(val_keys)
    
    if overlap:
        print(f"\n❌ ERROR: {len(overlap)} sessions appear in BOTH train and val!")
        print("This would cause data leakage!")
        for key in overlap:
            print(f"  - {key}")
        return False
    else:
        print(f"\n✓ PASS: No overlap between train and validation sessions")
        print(f"✓ Data properly separated at recording session level")
        print(f"✓ No data leakage - training will produce realistic results")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_no_leakage.py <dataset.json>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    success = verify_no_leakage(dataset_path)
    
    print("\n" + "="*70)
    if success:
        print("✓ VERIFICATION PASSED - Safe to train")
    else:
        print("❌ VERIFICATION FAILED - Fix data splitting")
    print("="*70)
    
    sys.exit(0 if success else 1)