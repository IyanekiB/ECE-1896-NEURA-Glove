"""
Mock Sender - Replay predictions from predictions_log.json
Sends packets to Unity using logged frame data instead of live inference
Also receives UDP notifications at port 5556 and logs them
Forwards notifications to BLE device (ESP32) when connected
"""

import json
import socket
import time
import numpy as np
import sys
import threading
import asyncio
from bleak import BleakClient, BleakScanner


# Unity UDP Configuration (same as realtime_inference.py)
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5555

# UDP Listener Configuration (for receiving notifications)
UDP_LISTEN_IP = "127.0.0.1"
UDP_LISTEN_PORT = 5556

# Finger bend ratios (same as realtime_inference.py)
FINGER_BEND_RATIOS = {
    'thumb': {
        'metacarpal': 0.4,
        'proximal': 1.2,
        'intermediate': 1.5,
        'distal': 0.8
    },
    'index': {
        'metacarpal': 0.7,
        'proximal': 1.3,
        'intermediate': 2.2,
        'distal': 1.2
    },
    'middle': {
        'metacarpal': 0.7,
        'proximal': 1.3,
        'intermediate': 2.2,
        'distal': 1.2
    },
    'ring': {
        'metacarpal': 0.7,
        'proximal': 1.3,
        'intermediate': 1.8,
        'distal': 1.0
    },
    'pinky': {
        'metacarpal': 0.7,
        'proximal': 1.3,
        'intermediate': 1.8,
        'distal': 1.0
    }
}


class MockSender:
    """Replays logged predictions and sends to Unity"""
    
    def __init__(self, log_file, playback_speed=1.0, log_notifications=True, use_week4_correction=False):
        self.log_file = log_file
        self.playback_speed = playback_speed
        self.log_notifications = log_notifications
        self.use_week4_correction = use_week4_correction
        
        # Load predictions log
        print(f"Loading predictions from: {log_file}")
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.predictions = data['predictions']
        
        print(f"✓ Loaded {self.metadata['total_predictions']} predictions")
        print(f"  Duration: {self.metadata['duration']:.1f}s")
        print(f"  Kalman enabled: {self.metadata['kalman_enabled']}")
        if self.use_week4_correction:
            print(f"  ⚠ Week4 IMU correction: ENABLED (for testing)")
        
        # UDP socket for sending to Unity
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = (UNITY_IP, UNITY_PORT)
        print(f"✓ UDP socket created for {UNITY_IP}:{UNITY_PORT}")
        
        # UDP listener for receiving notifications
        self.listener_sock = None
        self.listener_thread = None
        self.listening = False
        self.notifications = []
        self.notification_log_file = "grab_notifications.json"
        
        if self.log_notifications:
            self._setup_listener()
    
    def _setup_listener(self):
        """Setup UDP listener for incoming notifications"""
        try:
            self.listener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.listener_sock.settimeout(0.1)  # Non-blocking with timeout
            self.listener_sock.bind((UDP_LISTEN_IP, UDP_LISTEN_PORT))
            print(f"✓ UDP listener created on {UDP_LISTEN_IP}:{UDP_LISTEN_PORT}")
        except Exception as e:
            print(f"⚠ Warning: Could not create UDP listener: {e}")
            self.listener_sock = None
            self.log_notifications = False
    
    def _listen_for_notifications(self):
        """Background thread to listen for UDP notifications"""
        print("🎧 Notification listener started")
        
        while self.listening and self.listener_sock:
            try:
                data, addr = self.listener_sock.recvfrom(4096)
                message = data.decode('utf-8')
                
                notification = {
                    'timestamp': time.time(),
                    'from': f"{addr[0]}:{addr[1]}",
                    'message': message
                }
                
                self.notifications.append(notification)
                print(f"📨 Received notification: {message} from {addr}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.listening:
                    print(f"⚠ Error receiving notification: {e}")
        
        print("🎧 Notification listener stopped")
    
    def _start_listener(self):
        """Start the notification listener thread"""
        if not self.log_notifications or not self.listener_sock:
            return
        
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_for_notifications, daemon=True)
        self.listener_thread.start()
    
    def _stop_listener(self):
        """Stop the notification listener thread"""
        if not self.listener_thread:
            return
        
        self.listening = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
    
    def _save_notifications(self):
        """Save received notifications to file"""
        if not self.notifications:
            print("⚠ No notifications received to save")
            return
        
        output = {
            'metadata': {
                'total_notifications': len(self.notifications),
                'log_file': self.log_file,
                'playback_speed': self.playback_speed,
                'listen_port': UDP_LISTEN_PORT
            },
            'notifications': self.notifications
        }
        
        try:
            with open(self.notification_log_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n✓ Notifications saved to: {self.notification_log_file}")
            print(f"  Total notifications: {len(self.notifications)}")
        except Exception as e:
            print(f"\n✗ Error saving notifications: {e}")
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion [x, y, z, w]"""
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return [qx, qy, qz, qw]
    
    def correct_imu_quaternion(self, imu_quat):
        """Correct IMU quaternion for mounting orientation (WEEK 4 MAPPING - TEST ONLY)
        
        This applies the correction from week4/realtime_inference.py
        Week5 removed this because it was causing incorrect orientation
        """
        qx, qy, qz, qw = imu_quat
        
        # 180° rotation around X-axis (from week4)
        corrected_qx = qw
        corrected_qy = -qz
        corrected_qz = qy
        corrected_qw = qx
        
        return [corrected_qx, corrected_qy, corrected_qz, corrected_qw]
    
    def distribute_rotations(self, proximal_angle, ratios):
        """Distribute proximal angle across finger joints"""
        proximal_angle = max(0, proximal_angle)
        
        return {
            'metacarpal': proximal_angle * ratios['metacarpal'],
            'proximal': proximal_angle * ratios['proximal'],
            'intermediate': proximal_angle * ratios['intermediate'],
            'distal': proximal_angle * ratios['distal']
        }
    
    def build_unity_packet(self, prediction):
        """Build Unity UDP packet from logged prediction
        
        The prediction contains:
        - angles: [thumb_y, index_y, middle_y, ring_y, pinky_y]
        - imu_quat: [qx, qy, qz, qw]
        """
        # Extract proximal Y-axis rotations from logged angles
        thumb_y = prediction['angles'][0]
        index_y = prediction['angles'][1]
        middle_y = prediction['angles'][2]
        ring_y = prediction['angles'][3]
        pinky_y = prediction['angles'][4]
        
        # Extract IMU quaternion
        imu_quat = prediction['imu_quat']  # [qx, qy, qz, qw]
        
        # Apply week4 correction if enabled (for testing)
        if self.use_week4_correction:
            imu_quat = self.correct_imu_quaternion(imu_quat)
        
        # Distribute rotations across joints
        thumb_joints = self.distribute_rotations(thumb_y, FINGER_BEND_RATIOS['thumb'])
        index_joints = self.distribute_rotations(index_y, FINGER_BEND_RATIOS['index'])
        middle_joints = self.distribute_rotations(middle_y, FINGER_BEND_RATIOS['middle'])
        ring_joints = self.distribute_rotations(ring_y, FINGER_BEND_RATIOS['ring'])
        pinky_joints = self.distribute_rotations(pinky_y, FINGER_BEND_RATIOS['pinky'])
        
        # Convert to quaternions (X-axis rotation for finger curl)
        def angle_to_quat_x(angle):
            theta = np.radians(max(0, angle))
            return [np.sin(theta/2), 0, 0, np.cos(theta/2)]
        
        # Fixed thumb metacarpal
        thumb_metacarpal_rot = self.euler_to_quaternion(21.194, 43.526, -69.284)
        
        # Build packet (same structure as realtime_inference.py)
        packet = {
            "timestamp": prediction['timestamp'],
            "isGrabbing": prediction['pose'] == 'grab' or prediction['pose'] == 'fist',
            "hand": "left",
            "wrist": {
                "position": [0, 0, 0],
                "rotation": imu_quat  # From logged data
            },
            "thumb": {
                "metacarpal": {"position": [0, 0, 0], "rotation": thumb_metacarpal_rot},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(thumb_joints['distal'])}
            },
            "index": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(index_joints['distal'])}
            },
            "middle": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(middle_joints['distal'])}
            },
            "ring": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(ring_joints['distal'])}
            },
            "pinky": {
                "metacarpal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['metacarpal'])},
                "proximal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['proximal'])},
                "intermediate": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['intermediate'])},
                "distal": {"position": [0, 0, 0], "rotation": angle_to_quat_x(pinky_joints['distal'])}
            }
        }
        
        return packet
    
    def send_to_unity(self, packet):
        """Send packet to Unity via UDP"""
        try:
            packet_json = json.dumps(packet, separators=(',', ':'))
            self.sock.sendto(packet_json.encode('utf-8'), self.unity_address)
            return True
        except Exception as e:
            print(f"Error sending to Unity: {e}")
            return False
    
    def replay(self, loop=False):
        """Replay logged predictions to Unity
        
        Args:
            loop: If True, continuously loop the playback
        """
        print(f"\n{'='*60}")
        print("MOCK SENDER - REPLAYING LOGGED PREDICTIONS")
        print(f"{'='*60}")
        print(f"Playback speed: {self.playback_speed}x")
        print(f"Loop mode: {'ENABLED' if loop else 'DISABLED'}")
        print(f"Streaming to Unity at {UNITY_IP}:{UNITY_PORT}")
        if self.log_notifications:
            print(f"Listening for notifications on {UDP_LISTEN_IP}:{UDP_LISTEN_PORT}")
        if self.use_week4_correction:
            print(f"⚠ Week4 IMU correction: ENABLED (testing buggy mapping)")
        print("\nPress Ctrl+C to stop\n")
        
        # Start notification listener
        self._start_listener()
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                if iteration > 1:
                    print(f"\n🔄 Starting loop iteration {iteration}")
                
                start_time = time.time()
                frames_sent = 0
                
                # Calculate time deltas between predictions
                for i, prediction in enumerate(self.predictions):
                    # Build and send packet
                    packet = self.build_unity_packet(prediction)
                    self.send_to_unity(packet)
                    frames_sent += 1
                    
                    # Calculate delay to next frame
                    if i < len(self.predictions) - 1:
                        next_prediction = self.predictions[i + 1]
                        time_delta = next_prediction['timestamp'] - prediction['timestamp']
                        # Adjust for playback speed
                        sleep_time = time_delta / self.playback_speed
                        
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                    # Print status every 50 frames
                    if frames_sent % 50 == 0:
                        elapsed = time.time() - start_time
                        fps = frames_sent / elapsed if elapsed > 0 else 0
                        notif_count = f" | Notifications: {len(self.notifications)}" if self.log_notifications else ""
                        print(f"Frame {frames_sent}/{len(self.predictions)} | "
                              f"FPS: {fps:.1f} | "
                              f"Pose: {prediction['pose']} ({prediction['confidence']:.1%}){notif_count}")
                
                elapsed = time.time() - start_time
                print(f"\n✓ Completed playback iteration {iteration}")
                print(f"  Frames sent: {frames_sent}")
                print(f"  Duration: {elapsed:.1f}s")
                print(f"  Average FPS: {frames_sent / elapsed:.1f}")
                if self.log_notifications:
                    print(f"  Notifications received: {len(self.notifications)}")
                
                if not loop:
                    break
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Playback stopped by user")
        finally:
            # Stop listener and save notifications
            self._stop_listener()
            
            if self.listener_sock:
                self.listener_sock.close()
            
            self.sock.close()
            
            # Save notifications if any were received
            if self.log_notifications and self.notifications:
                print("\n💾 Saving notifications...")
                self._save_notifications()
            
            print(f"\n{'='*60}")
            print("SHUTDOWN COMPLETE")
            print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python mock_sender.py <predictions_log.json> [--speed SPEED] [--loop] [--no-log] [--week4-correction]")
        print("\nExamples:")
        print("  python mock_sender.py predictions_log.json")
        print("  python mock_sender.py predictions_log.json --speed 2.0")
        print("  python mock_sender.py predictions_log.json --loop")
        print("  python mock_sender.py predictions_log.json --speed 0.5 --loop")
        print("  python mock_sender.py predictions_log.json --no-log")
        print("  python mock_sender.py predictions_log.json --week4-correction")
        print("\nOptions:")
        print("  --speed SPEED        Playback speed multiplier (default: 2.0)")
        print("  --loop               Continuously loop the playback")
        print("  --no-log             Disable UDP notification logging (default: enabled)")
        print("  --week4-correction   Apply week4 IMU correction mapping (for testing)")
        print("\nNotifications:")
        print("  Listens on UDP port 5556 for incoming notifications")
        print("  Saves received notifications to 'grab_notifications.json'")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Parse optional arguments
    playback_speed = 2.0
    loop = False
    log_notifications = '--no-log' not in sys.argv
    use_week4_correction = '--week4-correction' in sys.argv
    
    if '--speed' in sys.argv:
        idx = sys.argv.index('--speed')
        playback_speed = float(sys.argv[idx + 1])
    
    if '--loop' in sys.argv:
        loop = True
    
    # Create sender and replay
    sender = MockSender(
        log_file, 
        playback_speed=playback_speed, 
        log_notifications=log_notifications,
        use_week4_correction=use_week4_correction
    )
    sender.replay(loop=loop)


if __name__ == "__main__":
    main()

