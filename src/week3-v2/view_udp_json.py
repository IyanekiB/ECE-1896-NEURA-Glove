"""
Simple UDP Listener - View JSON packets being sent to Unity

This script listens on the same port as Unity (5555) and displays
the JSON data being sent by the inference engine.

Usage:
    # Make sure Unity is NOT running (so port 5555 is free)
    python view_udp_json.py
    
    # In another terminal, run inference engine
    python enhanced_inference_engine.py --model models/best_model.pth
"""

import socket
import json
import sys
from datetime import datetime


def view_udp_packets(port=5555, max_packets=10, pretty_print=True):
    """
    Listen for UDP packets and display JSON content
    
    Args:
        port: UDP port to listen on (default: 5555)
        max_packets: Number of packets to display (0 = unlimited)
        pretty_print: If True, format JSON nicely
    """
    
    print("="*80)
    print(f"UDP JSON VIEWER - Listening on port {port}")
    print("="*80)
    print(f"Max packets: {'Unlimited' if max_packets == 0 else max_packets}")
    print(f"Pretty print: {pretty_print}")
    print()
    print("⚠️  IMPORTANT: Make sure Unity is NOT running!")
    print("   (Only one program can listen on a port at a time)")
    print()
    print("Press Ctrl+C to stop")
    print("-"*80)
    print()
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind(('0.0.0.0', port))
        print(f"✓ Listening on 0.0.0.0:{port}")
        print("  Waiting for packets...\n")
        
        packet_count = 0
        
        while max_packets == 0 or packet_count < max_packets:
            # Receive packet (up to 64KB)
            data, addr = sock.recvfrom(65535)
            packet_count += 1
            
            # Decode and parse JSON
            try:
                json_str = data.decode('utf-8')
                json_data = json.loads(json_str)
                
                # Display packet info
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n{'='*80}")
                print(f"PACKET #{packet_count} - {timestamp}")
                print(f"From: {addr[0]}:{addr[1]}")
                print(f"Size: {len(data)} bytes")
                print(f"{'='*80}")
                
                if pretty_print:
                    # Pretty print JSON
                    print(json.dumps(json_data, indent=2))
                else:
                    # Compact JSON
                    print(json_str)
                
                # Display summary info
                print(f"\n{'─'*80}")
                print(f"Summary:")
                print(f"  Timestamp: {json_data.get('timestamp', 'N/A')}")
                print(f"  Hand: {json_data.get('hand', 'N/A')}")
                print(f"  Wrist position: {json_data.get('wrist', {}).get('position', 'N/A')}")
                print(f"  Wrist rotation: {json_data.get('wrist', {}).get('rotation', 'N/A')}")
                print(f"  Fingers: {[k for k in json_data.keys() if k not in ['timestamp', 'hand', 'wrist']]}")
                print(f"{'─'*80}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Error decoding JSON: {e}")
                print(f"   Raw data: {data[:100]}...")
            except Exception as e:
                print(f"❌ Error processing packet: {e}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping...")
    
    except OSError as e:
        print(f"\n❌ Error binding to port {port}: {e}")
        print(f"   Is another program (Unity?) already using this port?")
        sys.exit(1)
    
    finally:
        sock.close()
        print(f"\n{'='*80}")
        print(f"STOPPED - Total packets received: {packet_count}")
        print(f"{'='*80}")


def view_single_packet(port=5555):
    """Capture and display just one packet"""
    
    print("Waiting for a single packet...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    
    try:
        data, addr = sock.recvfrom(65535)
        json_data = json.loads(data.decode('utf-8'))
        
        print("\n✓ Packet received!\n")
        print(json.dumps(json_data, indent=2))
        
        return json_data
        
    finally:
        sock.close()


def validate_json_structure(port=5555):
    """
    Capture a packet and validate it has the correct structure
    for Unity FrameConstructor format
    """
    
    print("="*80)
    print("JSON STRUCTURE VALIDATOR")
    print("="*80)
    print("Waiting for a packet to validate...\n")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    
    try:
        data, addr = sock.recvfrom(65535)
        json_data = json.loads(data.decode('utf-8'))
        
        print("✓ Packet received, validating structure...\n")
        
        errors = []
        warnings = []
        
        # Check top-level fields
        required_fields = ['timestamp', 'hand', 'wrist']
        for field in required_fields:
            if field not in json_data:
                errors.append(f"Missing required field: '{field}'")
        
        # Check wrist structure
        if 'wrist' in json_data:
            if 'position' not in json_data['wrist']:
                errors.append("Wrist missing 'position'")
            elif len(json_data['wrist']['position']) != 3:
                errors.append(f"Wrist position should have 3 values, has {len(json_data['wrist']['position'])}")
            
            if 'rotation' not in json_data['wrist']:
                errors.append("Wrist missing 'rotation'")
            elif len(json_data['wrist']['rotation']) != 4:
                errors.append(f"Wrist rotation should have 4 values (quaternion), has {len(json_data['wrist']['rotation'])}")
        
        # Check fingers
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger in fingers:
            if finger not in json_data:
                errors.append(f"Missing finger: '{finger}'")
                continue
            
            # Check finger joints
            joints = ['metacarpal', 'proximal', 'intermediate', 'distal']
            for joint in joints:
                if joint not in json_data[finger]:
                    errors.append(f"{finger} missing joint: '{joint}'")
                    continue
                
                # Check joint structure
                if 'position' not in json_data[finger][joint]:
                    errors.append(f"{finger}.{joint} missing 'position'")
                elif len(json_data[finger][joint]['position']) != 3:
                    errors.append(f"{finger}.{joint} position should have 3 values")
                
                if 'rotation' not in json_data[finger][joint]:
                    errors.append(f"{finger}.{joint} missing 'rotation'")
                elif len(json_data[finger][joint]['rotation']) != 4:
                    errors.append(f"{finger}.{joint} rotation should have 4 values (quaternion)")
        
        # Print results
        print("="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        
        if len(errors) == 0:
            print("✅ PASSED - JSON structure is correct!")
            print("\nStructure Summary:")
            print(f"  ✓ Timestamp: {json_data['timestamp']}")
            print(f"  ✓ Hand: {json_data['hand']}")
            print(f"  ✓ Wrist: position(3) + rotation(4)")
            print(f"  ✓ Thumb: 4 joints × [position(3) + rotation(4)]")
            print(f"  ✓ Index: 4 joints × [position(3) + rotation(4)]")
            print(f"  ✓ Middle: 4 joints × [position(3) + rotation(4)]")
            print(f"  ✓ Ring: 4 joints × [position(3) + rotation(4)]")
            print(f"  ✓ Pinky: 4 joints × [position(3) + rotation(4)]")
            print(f"\n  Total joints: 21")
            print(f"  Total values: 21 × 7 = 147 ✓")
        else:
            print("❌ FAILED - Found errors:\n")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
        
        print("="*80)
        
        return len(errors) == 0
        
    finally:
        sock.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='View JSON packets sent over UDP to Unity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View 10 packets with pretty printing
  python view_udp_json.py
  
  # View unlimited packets
  python view_udp_json.py --max-packets 0
  
  # View compact JSON (no formatting)
  python view_udp_json.py --compact
  
  # View on different port
  python view_udp_json.py --port 5556
  
  # Validate structure
  python view_udp_json.py --validate
  
  # View just one packet
  python view_udp_json.py --single
"""
    )
    
    parser.add_argument('--port', type=int, default=5555,
                       help='UDP port to listen on (default: 5555)')
    parser.add_argument('--max-packets', type=int, default=10,
                       help='Max packets to display (0 = unlimited, default: 10)')
    parser.add_argument('--compact', action='store_true',
                       help='Display compact JSON (no pretty printing)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate JSON structure and exit')
    parser.add_argument('--single', action='store_true',
                       help='Capture and display just one packet')
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            validate_json_structure(args.port)
        elif args.single:
            view_single_packet(args.port)
        else:
            view_udp_packets(
                port=args.port,
                max_packets=args.max_packets,
                pretty_print=not args.compact
            )
    except KeyboardInterrupt:
        print("\n\nStopped by user")