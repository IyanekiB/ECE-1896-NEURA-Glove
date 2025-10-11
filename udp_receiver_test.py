import socket
import json
import time
from datetime import datetime


class UDPReceiver:
    """Test receiver for Unity VR hand tracking data packets"""
    
    def __init__(self, listen_ip='127.0.0.1', listen_port=5555):
        self.listen_address = (listen_ip, listen_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.listen_address)
        self.sock.settimeout(5.0)  # 5 second timeout
        
        # Metrics
        self.packets_received = 0
        self.packets_failed = 0
        self.start_time = time.time()
        self.last_packet_time = None
        self.latencies = []
        
        print(f"UDP Receiver listening on {listen_ip}:{listen_port}")
        print("Waiting for packets: \n")
    
    def calculate_stats(self):
        """Calculate reception statistics"""
        elapsed = time.time() - self.start_time
        packet_rate = self.packets_received / elapsed if elapsed > 0 else 0
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            'total_packets': self.packets_received,
            'failed_packets': self.packets_failed,
            'packet_rate': packet_rate,
            'avg_latency_ms': avg_latency,
            'elapsed_time': elapsed
        }
    
    def validate_packet(self, packet_data):
        """Validate packet structure"""
        try:
            # Check required fields
            if 'Timestamp' not in packet_data:
                return False, "Missing Timestamp"
            
            if 'joints' not in packet_data:
                return False, "Missing joints array"
            
            joints = packet_data['joints']
            if not isinstance(joints, list):
                return False, "joints is not a list"
            
            if len(joints) != 21:
                return False, f"Expected 21 joints, got {len(joints)}"
            
            # Validate first joint structure
            joint = joints[0]
            if 'joint_id' not in joint:
                return False, "Joint missing joint_id"
            if 'position' not in joint:
                return False, "Joint missing position"
            if 'rotation' not in joint:
                return False, "Joint missing rotation"
            
            # Check position format [x, y, z]
            if len(joint['position']) != 3:
                return False, "Position should have 3 coordinates"
            
            # Check rotation format [x, y, z, w]
            if len(joint['rotation']) != 4:
                return False, "Rotation should have 4 quaternion components"
            
            return True, "Valid packet"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def print_packet_summary(self, packet_data):
        """Print summary of received packet"""
        timestamp = packet_data.get('Timestamp', 0)
        
        # Calculate latency (current time - packet timestamp)
        current_time_ms = int(time.time() * 1000)
        latency = current_time_ms - timestamp
        self.latencies.append(latency)
        
        # Keep only last 100 latencies for averaging
        if len(self.latencies) > 100:
            self.latencies.pop(0)
        
        print(f"\n{'='*60}")
        print(f"Packet #{self.packets_received}")
        print(f"Timestamp: {timestamp}")
        print(f"Latency: {latency}ms")
        print(f"Joints: {len(packet_data.get('joints', []))}")
        
        # Print first joint as example
        if packet_data.get('joints'):
            joint = packet_data['joints'][0]
            print(f"\nWrist (Joint 0):")
            print(f"  Position: [{joint['position'][0]:.3f}, {joint['position'][1]:.3f}, {joint['position'][2]:.3f}]")
            print(f"  Rotation: [{joint['rotation'][0]:.3f}, {joint['rotation'][1]:.3f}, {joint['rotation'][2]:.3f}, {joint['rotation'][3]:.3f}]")
    
    def listen(self, max_packets=None, verbose=True):
        """
        Listen for incoming UDP packets
        
        Args:
            max_packets: Maximum number of packets to receive (None = unlimited)
            verbose: Print detailed packet information
        """
        try:
            while max_packets is None or self.packets_received < max_packets:
                try:
                    # Receive data
                    data, addr = self.sock.recvfrom(65536)  # Max UDP packet size
                    
                    # Parse JSON
                    packet_data = json.loads(data.decode('utf-8'))
                    
                    # Validate packet
                    is_valid, message = self.validate_packet(packet_data)
                    
                    if is_valid:
                        self.packets_received += 1
                        self.last_packet_time = time.time()
                        
                        if verbose:
                            self.print_packet_summary(packet_data)
                        else:
                            # Just print a dot for each packet
                            print('.', end='', flush=True)
                            if self.packets_received % 50 == 0:
                                print(f" [{self.packets_received}]")
                    else:
                        self.packets_failed += 1
                        print(f"\n✗ Invalid packet: {message}")
                
                except socket.timeout:
                    if self.packets_received == 0:
                        print("No packets received yet. Make sure streamer is running...")
                    else:
                        print(f"\nTimeout - no packet received for 5 seconds")
                        print(f"Last packet received {time.time() - self.last_packet_time:.1f}s ago")
                        break
                
                except json.JSONDecodeError as e:
                    self.packets_failed += 1
                    print(f"\n✗ Failed to parse JSON: {e}")
                
                except Exception as e:
                    self.packets_failed += 1
                    print(f"\n✗ Error receiving packet: {e}")
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        finally:
            self.print_final_stats()
            self.sock.close()
    
    def print_final_stats(self):
        """Print final reception statistics"""
        stats = self.calculate_stats()
        
        print(f"\n\n{'='*60}")
        print("RECEPTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total Packets Received: {stats['total_packets']}")
        print(f"Failed Packets: {stats['failed_packets']}")
        print(f"Packet Rate: {stats['packet_rate']:.2f} packets/sec")
        print(f"Average Latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"Total Time: {stats['elapsed_time']:.2f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Get configuration
    ip = input("\nListen IP (default 127.0.0.1): ").strip()
    ip = ip if ip else '127.0.0.1'
    
    port = input("Listen Port (default 5555): ").strip()
    port = int(port) if port else 5555
    
    verbose = input("Verbose output? (y/n, default y): ").strip().lower()
    verbose = verbose != 'n'
    
    max_packets = input("Max packets to receive (Enter for unlimited): ").strip()
    max_packets = int(max_packets) if max_packets else None
    
    print(f"\nStarting receiver...")
    print("Press Ctrl+C to stop\n")
    
    # Create and run receiver
    receiver = UDPReceiver(listen_ip=ip, listen_port=port)
    receiver.listen(max_packets=max_packets, verbose=verbose)