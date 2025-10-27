"""
Unity UDP Bridge for NEURA GLOVE
Converts LSTM inference output (147 floats) to Unity-compatible JSON format

This bridge enables the inference engine to send hand pose data to Unity
using the same UDP/JSON protocol that works with mp_udp_streamer.py

IMPORTANT: Unity only needs rotation data (quaternions), NOT positions.
All positions are set to [0, 0, 0] to match FrameConstructor format.

The inference engine outputs 147 floats (21 joints Ã— 7 values):
- Each joint: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
- This bridge IGNORES the position values (pos_x, pos_y, pos_z)
- Only rotation quaternions (rot_x, rot_y, rot_z, rot_w) are used
- All positions sent to Unity are [0, 0, 0]

This matches mp_udp_streamer.py behavior using FrameConstructor.

Author: NeuraGlove Senior Design Team
Date: October 2025
"""

import socket
import json
import time
import numpy as np
from typing import Optional


class UnityUDPBridge:
    """
    Bridge between inference engine and Unity UDP receiver
    
    Converts LSTM output format (147 floats) to Unity JSON format
    that matches FrameConstructor structure used by mp_udp_streamer.py
    """
    
    def __init__(self, unity_ip: str = '127.0.0.1', unity_port: int = 5555):
        """
        Initialize UDP bridge
        
        Args:
            unity_ip: IP address where Unity is running
            unity_port: UDP port Unity is listening on (default: 5555)
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = (unity_ip, unity_port)
        self.packet_count = 0
        self.error_count = 0
        
        print(f"âœ“ UDP Bridge initialized: {unity_ip}:{unity_port}")
    
    def normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit length
        
        Args:
            quat: [x, y, z, w] quaternion
            
        Returns:
            Normalized quaternion
        """
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            # Degenerate case: return identity quaternion
            return np.array([0.0, 0.0, 0.0, 1.0])
        return quat / norm
    
    def angle_to_quaternion(self, angle_radians: float, axis: str = 'x') -> np.ndarray:
        """
        Convert bend angle to quaternion rotation around specified axis
        
        Args:
            angle_radians: Rotation angle in radians  
            axis: Rotation axis ('x' for finger bending)
            
        Returns:
            [x, y, z, w] quaternion
        """
        half_angle = angle_radians / 2.0
        c = np.cos(half_angle)
        s = np.sin(half_angle)
        
        if axis == 'x':
            return np.array([s, 0.0, 0.0, c])
        elif axis == 'y':
            return np.array([0.0, s, 0.0, c])
        elif axis == 'z':
            return np.array([0.0, 0.0, s, c])
        else:
            return np.array([0.0, 0.0, 0.0, 1.0])
    
    def convert_to_unity_format(self, joint_data: np.ndarray, hand: str = "left") -> str:
        """
        Convert inference output to Unity JSON format
        Auto-detects format: 147 (old) or 16-21 (new bend angles)
        """
        data_size = joint_data.shape[0]
        
        if data_size == 147:
            return self.convert_147_to_json(joint_data, hand)
        elif 16 <= data_size <= 21:
            return self.convert_angles_to_json(joint_data, hand)
        else:
            raise ValueError(f"Unexpected data size: {data_size}. Expected 147 or 16-21")
    
    def convert_147_to_json(self, joint_data: np.ndarray, hand: str = "left") -> str:
        """
        Convert 147 float array to Unity JSON format (FrameConstructor format)
        
        Args:
            joint_data: (147,) array from inference engine
                       21 joints Ã— 7 values per joint
                       Each joint: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
                       NOTE: Positions are IGNORED, Unity only uses rotations
            hand: "left" or "right"
        
        Returns:
            JSON string matching FrameConstructor format
            
        Format matches mp_udp_streamer.py output (FrameConstructor):
        - ALL positions are [0, 0, 0] (Unity doesn't use position data)
        - Only rotations (quaternions) are used
        {
            "timestamp": float,
            "hand": "left" or "right",
            "wrist": {"position": [0,0,0], "rotation": [x,y,z,w]},
            "thumb": {
                "metacarpal": {"position": [0,0,0], "rotation": [x,y,z,w]},
                ...
            },
            ...
        }
        """
        # Ensure correct shape
        if joint_data.shape != (147,):
            raise ValueError(f"Expected (147,) array, got {joint_data.shape}")
        
        # Reshape: (147,) â†’ (21, 7)
        # Each row: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
        # We IGNORE positions (0:3) and only use rotations (3:7)
        joints = joint_data.reshape(21, 7)
        
        def extract_joint(idx: int) -> dict:
            """
            Extract rotation for a joint (matching FrameConstructor._zero_pos_rot)
            Position is ALWAYS [0, 0, 0] - Unity doesn't need position data
            """
            # IGNORE position data (joints[idx, 0:3])
            rot = joints[idx, 3:7]  # Extract ONLY rotation quaternion
            
            # Normalize quaternion to prevent Unity errors
            rot = self.normalize_quaternion(rot)
            
            # ALL positions are [0, 0, 0] - matches FrameConstructor format
            return {
                "position": [0, 0, 0],
                "rotation": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
            }
        
        # Build frame structure matching FrameConstructor
        # Joint indices follow MediaPipe hand landmark standard:
        # 0: Wrist
        # 1-4: Thumb (CMC, MCP, IP, Tip)
        # 5-8: Index (MCP, PIP, DIP, Tip)
        # 9-12: Middle (MCP, PIP, DIP, Tip)
        # 13-16: Ring (MCP, PIP, DIP, Tip)
        # 17-20: Pinky (MCP, PIP, DIP, Tip)
        
        frame = {
            "timestamp": time.time(),
            "hand": hand,
            "wrist": extract_joint(0),
            "thumb": {
                "metacarpal": extract_joint(1),
                "proximal": extract_joint(2),
                "intermediate": extract_joint(3),
                "distal": extract_joint(4)
            },
            "index": {
                "metacarpal": extract_joint(5),
                "proximal": extract_joint(6),
                "intermediate": extract_joint(7),
                "distal": extract_joint(8)
            },
            "middle": {
                "metacarpal": extract_joint(9),
                "proximal": extract_joint(10),
                "intermediate": extract_joint(11),
                "distal": extract_joint(12)
            },
            "ring": {
                "metacarpal": extract_joint(13),
                "proximal": extract_joint(14),
                "intermediate": extract_joint(15),
                "distal": extract_joint(16)
            },
            "pinky": {
                "metacarpal": extract_joint(17),
                "proximal": extract_joint(18),
                "intermediate": extract_joint(19),
                "distal": extract_joint(20)
            }
        }
        
        return json.dumps(frame)
    
    def convert_angles_to_json(self, joint_data: np.ndarray, hand: str = "left") -> str:
        """
        Convert bend angles (16-21 values) to Unity JSON format
        
        Args:
            joint_data: Array of 16-21 bend angles (radians)
                       Order: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)
            hand: "left" or "right"
            
        Returns:
            JSON string matching FrameConstructor format
        """
        def angle_to_joint(angle: float) -> dict:
            """Convert bend angle to joint dict"""
            quat = self.angle_to_quaternion(angle, axis='x')
            return {
                "position": [0, 0, 0],
                "rotation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            }
        
        # Parse angles - handle variable length
        idx = 0
        
        # Wrist
        wrist_angle = joint_data[idx] if idx < len(joint_data) else 0.0
        idx += 1
        
        # Get angles for each finger (4 per finger)
        def get_angles(n: int) -> list:
            nonlocal idx
            angles = []
            for _ in range(n):
                angles.append(joint_data[idx] if idx < len(joint_data) else 0.0)
                idx += 1
            return angles
        
        # Angles are in alphabetically sorted order within each finger
        thumb_angles = get_angles(4)  # distal, metacarpal, proximal, tip
        index_angles = get_angles(4)  # distal, intermediate, metacarpal, proximal
        middle_angles = get_angles(4)
        ring_angles = get_angles(4)
        pinky_angles = get_angles(4)
        
        frame = {
            "timestamp": time.time(),
            "hand": hand,
            "wrist": angle_to_joint(wrist_angle),
            "thumb": {
                "metacarpal": angle_to_joint(thumb_angles[1]),
                "proximal": angle_to_joint(thumb_angles[2]),
                "intermediate": angle_to_joint(thumb_angles[0]),
                "distal": angle_to_joint(thumb_angles[3])
            },
            "index": {
                "metacarpal": angle_to_joint(index_angles[2]),
                "proximal": angle_to_joint(index_angles[3]),
                "intermediate": angle_to_joint(index_angles[1]),
                "distal": angle_to_joint(index_angles[0])
            },
            "middle": {
                "metacarpal": angle_to_joint(middle_angles[2]),
                "proximal": angle_to_joint(middle_angles[3]),
                "intermediate": angle_to_joint(middle_angles[1]),
                "distal": angle_to_joint(middle_angles[0])
            },
            "ring": {
                "metacarpal": angle_to_joint(ring_angles[2]),
                "proximal": angle_to_joint(ring_angles[3]),
                "intermediate": angle_to_joint(ring_angles[1]),
                "distal": angle_to_joint(ring_angles[0])
            },
            "pinky": {
                "metacarpal": angle_to_joint(pinky_angles[2]),
                "proximal": angle_to_joint(pinky_angles[3]),
                "intermediate": angle_to_joint(pinky_angles[1]),
                "distal": angle_to_joint(pinky_angles[0])
            }
        }
        
        return json.dumps(frame)
    
    def send_to_unity(self, joint_data: np.ndarray, hand: str = "left") -> bool:
        """
        Send joint data to Unity as JSON over UDP
        
        Args:
            joint_data: (147,) array from inference engine
            hand: "left" or "right"
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Convert to JSON (auto-detects format)
            json_packet = self.convert_to_unity_format(joint_data, hand)
            
            # Send via UDP
            self.sock.sendto(json_packet.encode('utf-8'), self.unity_address)
            
            self.packet_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            if self.error_count % 100 == 1:  # Print every 100th error
                print(f"âš ï¸  UDP send error ({self.error_count} total): {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get bridge statistics"""
        return {
            "packets_sent": self.packet_count,
            "errors": self.error_count,
            "success_rate": (self.packet_count / (self.packet_count + self.error_count)) 
                           if (self.packet_count + self.error_count) > 0 else 0.0
        }
    
    def close(self):
        """Close UDP socket"""
        self.sock.close()
        print(f"\nâœ“ UDP Bridge closed")
        print(f"  Total packets sent: {self.packet_count}")
        print(f"  Total errors: {self.error_count}")


def test_bridge():
    """Test the UDP bridge with random data"""
    print("\n" + "="*70)
    print("UDP BRIDGE TEST")
    print("="*70)
    
    # Create bridge
    bridge = UnityUDPBridge('127.0.0.1', 5555)
    
    # Generate test data (147 random floats)
    print("\nGenerating test data...")
    joint_data = np.random.randn(147).astype(np.float32)
    
    # Normalize quaternions in test data
    joints = joint_data.reshape(21, 7)
    for i in range(21):
        quat = joints[i, 3:7]
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        joints[i, 3:7] = quat
    joint_data = joints.reshape(147)
    
    # Convert to JSON
    print("Converting to JSON...")
    json_str = bridge.convert_to_unity_format(joint_data)
    data = json.loads(json_str)
    
    # Validate structure
    print("\nValidating structure...")
    assert "timestamp" in data, "Missing timestamp"
    assert "hand" in data, "Missing hand"
    assert "wrist" in data, "Missing wrist"
    assert "thumb" in data, "Missing thumb"
    assert "index" in data, "Missing index"
    assert "middle" in data, "Missing middle"
    assert "ring" in data, "Missing ring"
    assert "pinky" in data, "Missing pinky"
    
    # Check wrist structure
    assert "position" in data["wrist"], "Wrist missing position"
    assert "rotation" in data["wrist"], "Wrist missing rotation"
    assert len(data["wrist"]["position"]) == 3, "Position should have 3 values"
    assert len(data["wrist"]["rotation"]) == 4, "Rotation should have 4 values (quaternion)"
    assert data["wrist"]["position"] == [0, 0, 0], "Wrist position should be [0, 0, 0]"
    
    # Check finger structure
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        assert "metacarpal" in data[finger], f"{finger} missing metacarpal"
        assert "proximal" in data[finger], f"{finger} missing proximal"
        assert "intermediate" in data[finger], f"{finger} missing intermediate"
        assert "distal" in data[finger], f"{finger} missing distal"
        
        # Verify ALL positions are [0, 0, 0] (FrameConstructor format)
        for joint in ["metacarpal", "proximal", "intermediate", "distal"]:
            assert data[finger][joint]["position"] == [0, 0, 0], \
                f"{finger}.{joint} position should be [0, 0, 0], got {data[finger][joint]['position']}"
    
    print("âœ“ Structure validation passed")
    print("âœ“ ALL positions correctly set to [0, 0, 0] (FrameConstructor format)")
    
    # Send test packet
    print("\nSending test packet to Unity...")
    success = bridge.send_to_unity(joint_data)
    if success:
        print("âœ“ Packet sent successfully")
    else:
        print("âœ— Failed to send packet")
    
    # Print stats
    print("\nBridge Statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Close bridge
    bridge.close()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_bridge()