import sys
from week1.mediapipe_collector import MediaPipeCollector
from week1.udp_streamer import HandDataStreamer


def print_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print("NEURA GLOVE - MediaPipe Hand Tracking Demo")
    print("Week 1: Ground Truth Data Collection")
    print("="*50)
    print("\nOptions:")
    print("1. Collect training dataset (save to file)")
    print("2. Stream to Unity VR (real-time UDP)")
    print("3. Exit")
    print("\n" + "="*50)


def collect_dataset():
    """Run data collection mode"""
    print("\n--- DATA COLLECTION MODE ---")
    
    duration = input("Collection duration in seconds (default 60): ").strip()
    duration = int(duration) if duration else 60
    
    collector = MediaPipeCollector(save_dir='datasets')
    collector.collect_session(duration_seconds=duration, display=True)
    collector.save_dataset()
    
    print("\n✓ Dataset saved to 'datasets/' folder")


def stream_to_unity():
    """Run streaming mode"""
    print("\n--- STREAMING MODE ---")
    
    ip = input("Unity IP address (default 127.0.0.1): ").strip()
    ip = ip if ip else '127.0.0.1'
    
    port = input("Unity port (default 5555): ").strip()
    port = int(port) if port else 5555
    
    streamer = HandDataStreamer(unity_ip=ip, unity_port=port)
    streamer.stream(show_video=True)

def main():
    """Main program loop"""
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                collect_dataset()
            elif choice == '2':
                stream_to_unity()
            elif choice == '3':
                print("\nExiting...")
                sys.exit(0)
            else:
                print("Invalid option. Please select 1-3.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()