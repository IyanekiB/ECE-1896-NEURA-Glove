"""
Merge Multiple Aligned Sessions for Combined Training
Combines aligned datasets from multiple sessions into one training file
"""

import json
import sys
from pathlib import Path


def merge_aligned_sessions(aligned_files, output_file):
    """Merge multiple aligned session files
    
    Args:
        aligned_files: List of paths to aligned_*.json files
        output_file: Output path for merged dataset
    """
    print(f"\n{'='*60}")
    print("MERGING ALIGNED SESSIONS")
    print(f"{'='*60}")
    
    all_pairs = []
    session_info = []
    
    for file_path in aligned_files:
        print(f"\nLoading: {file_path}")
        
        with open(file_path) as f:
            data = json.load(f)
        
        pairs = data['pairs']
        metadata = data['metadata']
        
        print(f"  Pairs: {len(pairs)}")
        print(f"  Sensor session: {metadata.get('sensor_session', 'unknown')}")
        print(f"  Camera session: {metadata.get('camera_session', 'unknown')}")
        
        all_pairs.extend(pairs)
        session_info.append({
            'file': str(file_path),
            'sensor_session': metadata.get('sensor_session', 'unknown'),
            'camera_session': metadata.get('camera_session', 'unknown'),
            'pairs_count': len(pairs)
        })
    
    # Create merged dataset
    merged_data = {
        'metadata': {
            'total_sessions': len(aligned_files),
            'total_pairs': len(all_pairs),
            'input_dimensions': 5,
            'output_dimensions': 10,
            'input_description': ['thumb_angle', 'index_angle', 'middle_angle', 'ring_angle', 'pinky_angle'],
            'output_description': [
                'thumb_ip_x', 'thumb_ip_y',
                'index_pip_x', 'index_pip_y',
                'middle_pip_x', 'middle_pip_y',
                'ring_pip_x', 'ring_pip_y',
                'pinky_pip_x', 'pinky_pip_y'
            ],
            'sessions': session_info
        },
        'pairs': all_pairs
    }
    
    # Save merged dataset
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Total sessions: {len(aligned_files)}")
    print(f"Total training pairs: {len(all_pairs)}")
    print(f"\nPer-session breakdown:")
    for info in session_info:
        print(f"  {info['sensor_session']}: {info['pairs_count']} pairs")
    
    return output_file


def main():
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python merge_sessions.py <output_file> <aligned_file1> <aligned_file2> ...")
        print("\nExample:")
        print("  python merge_sessions.py data/aligned/merged_all_sessions.json \\")
        print("         data/aligned/aligned_session_001_session_001.json \\")
        print("         data/aligned/aligned_session_002_session_002.json \\")
        print("         data/aligned/aligned_session_003_session_003.json")
        print("\nThen train with:")
        print("  python train_model.py data/aligned/merged_all_sessions.json")
        sys.exit(1)
    
    output_file = sys.argv[1]
    aligned_files = [Path(f) for f in sys.argv[2:]]
    
    # Verify all files exist
    for f in aligned_files:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
    
    merge_aligned_sessions(aligned_files, output_file)


if __name__ == "__main__":
    main()