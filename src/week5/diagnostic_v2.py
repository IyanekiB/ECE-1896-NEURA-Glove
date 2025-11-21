"""
Diagnostic Tool - FIXED VERSION
Use this to:
1. See actual flex angles and ML predictions
2. Calibrate pose templates (FIXED: now actually saves templates)
3. Fix IMU orientation
4. Diagnose performance issues
"""

import asyncio
import json
import numpy as np
import torch
import time
from bleak import BleakClient, BleakScanner
from collections import deque
import threading
import sys


# BLE Configuration
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DEVICE_NAME = "ESP32-BLE"

# Flex sensor calibration
FLEX_MIN_VOLTAGE = 0.55
FLEX_MAX_VOLTAGE = 1.65


class DiagnosticTool:
    """Diagnostic tool for debugging the system - FIXED VERSION"""
    
    def __init__(self, model_path):
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.input_scaler = checkpoint['input_scaler']
        self.output_scaler = checkpoint['output_scaler']
        
        from train_model import FlexToRotationModel
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_dim = config['input_dim']
            output_dim = config['output_dim']
            hidden_dims = config['hidden_dims']
        else:
            state_dict = checkpoint['model_state_dict']
            linear_keys = sorted(
                [k for k in state_dict.keys() if k.endswith(".weight") and k.startswith("network.")],
                key=lambda k: int(k.split('.')[1])
            )
            input_dim = state_dict[linear_keys[0]].shape[1]
            layer_dims = [state_dict[k].shape[0] for k in linear_keys]
            hidden_dims = layer_dims[:-1]
            output_dim = layer_dims[-1]
        
        self.model = FlexToRotationModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ Model loaded\n")
        
        # Statistics
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None
        
        # Angle history for template calibration
        self.angle_history = []
        
        # FIXED: Template generation
        self.templates_generated = {}
        self.command_queue = []
        self.running = True
        
    def voltage_to_angle(self, voltage):
        """Convert flex voltage to bend angle"""
        voltage = np.clip(voltage, FLEX_MIN_VOLTAGE, FLEX_MAX_VOLTAGE)
        normalized = (voltage - FLEX_MIN_VOLTAGE) / (FLEX_MAX_VOLTAGE - FLEX_MIN_VOLTAGE)
        angle = 90.0 * (1.0 - normalized)
        return float(angle)
    
    def parse_ble_data(self, data_string):
        """Parse BLE data from ESP32"""
        try:
            values = [float(x) for x in data_string.split(',')]
            if len(values) != 15:
                return None, None
            
            # Flex angles
            flex_angles = [
                self.voltage_to_angle(values[0]),
                self.voltage_to_angle(values[1]),
                self.voltage_to_angle(values[2]),
                self.voltage_to_angle(values[3]),
                self.voltage_to_angle(values[4])
            ]
            
            # IMU quaternion [qx, qy, qz, qw]
            imu_quat = [values[6], values[7], values[8], values[5]]
            
            return np.array(flex_angles), imu_quat
        
        except Exception as e:
            return None, None
    
    def predict_rotations(self, flex_angles):
        """Predict joint rotations"""
        flex_scaled = self.input_scaler.transform([flex_angles])
        flex_tensor = torch.FloatTensor(flex_scaled)
        
        with torch.no_grad():
            output_scaled = self.model(flex_tensor).numpy()
        
        rotations = self.output_scaler.inverse_transform(output_scaled)[0]
        return rotations
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications"""
        # Measure frame time
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        data_string = data.decode('utf-8')
        flex_angles, imu_quat = self.parse_ble_data(data_string)
        
        if flex_angles is None:
            return
        
        # Predict rotations
        rotations = self.predict_rotations(flex_angles)
        
        # Extract proximal Y-axis rotations (the bend angles)
        proximal_angles = [
            rotations[1],  # Thumb
            rotations[3],  # Index
            rotations[5],  # Middle
            rotations[7],  # Ring
            rotations[9]   # Pinky
        ]
        
        # Store for template generation
        self.angle_history.append({
            'flex': flex_angles.tolist(),
            'proximal': proximal_angles,
            'timestamp': current_time
        })
        
        # Keep only last 1000 samples
        if len(self.angle_history) > 1000:
            self.angle_history.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0
        
        # Print diagnostic info every 30 frames
        if len(self.angle_history) % 30 == 0:
            print(f"\n{'='*80}")
            print(f"FRAME {len(self.angle_history)} | FPS: {fps:.1f}")
            print(f"{'='*80}")
            
            print("\n📊 FLEX SENSOR ANGLES (Input):")
            print(f"  Thumb:  {flex_angles[0]:6.1f}°")
            print(f"  Index:  {flex_angles[1]:6.1f}°")
            print(f"  Middle: {flex_angles[2]:6.1f}°")
            print(f"  Ring:   {flex_angles[3]:6.1f}°")
            print(f"  Pinky:  {flex_angles[4]:6.1f}°")
            
            print("\n🤖 ML PREDICTED PROXIMAL ANGLES:")
            print(f"  Thumb:  {proximal_angles[0]:6.1f}°")
            print(f"  Index:  {proximal_angles[1]:6.1f}°")
            print(f"  Middle: {proximal_angles[2]:6.1f}°")
            print(f"  Ring:   {proximal_angles[3]:6.1f}°")
            print(f"  Pinky:  {proximal_angles[4]:6.1f}°")
            
            print("\n🧭 IMU QUATERNION:")
            print(f"  [x={imu_quat[0]:7.4f}, y={imu_quat[1]:7.4f}, z={imu_quat[2]:7.4f}, w={imu_quat[3]:7.4f}]")
            
            # Convert to euler for readability
            w, x, y, z = imu_quat[3], imu_quat[0], imu_quat[1], imu_quat[2]
            roll = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
            pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
            yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
            
            print(f"  Euler: Roll={roll:6.1f}°, Pitch={pitch:6.1f}°, Yaw={yaw:6.1f}°")
            
            if len(self.frame_times) > 10:
                print(f"\n⏱️  PERFORMANCE:")
                print(f"  Avg frame time: {np.mean(self.frame_times)*1000:.1f}ms")
                print(f"  Min frame time: {np.min(self.frame_times)*1000:.1f}ms")
                print(f"  Max frame time: {np.max(self.frame_times)*1000:.1f}ms")
            
            print(f"\n💡 Commands: [f]lat_hand  [F]ist  [g]rab  [p]eace  [o]k_sign  [s]ave_all  [q]uit")
    
    def generate_template(self, pose_name):
        """FIXED: Generate pose template from recent samples"""
        if len(self.angle_history) < 30:
            print(f"\n⚠ Not enough samples ({len(self.angle_history)}). Hold the pose longer.")
            return None
        
        # Use last 60 samples (last 6 seconds at 10Hz)
        recent_samples = self.angle_history[-60:]
        
        # Average the proximal angles
        avg_angles = np.mean([s['proximal'] for s in recent_samples], axis=0)
        
        print(f"\n{'='*80}")
        print(f"✓ TEMPLATE GENERATED: {pose_name}")
        print(f"{'='*80}")
        print(f"'{pose_name}': [{avg_angles[0]:.1f}, {avg_angles[1]:.1f}, {avg_angles[2]:.1f}, {avg_angles[3]:.1f}, {avg_angles[4]:.1f}],")
        print(f"\nCopy this line into POSE_TEMPLATES in realtime_inference scripts")
        
        # Store template
        self.templates_generated[pose_name] = {
            'angles': avg_angles.tolist(),
            'samples_used': len(recent_samples),
            'timestamp': time.time()
        }
        
        return avg_angles
    
    def save_all_templates(self):
        """FIXED: Save all generated templates to file"""
        if not self.templates_generated:
            print("\n⚠ No templates generated yet!")
            return
        
        output = {
            'metadata': {
                'generated_at': time.time(),
                'total_templates': len(self.templates_generated)
            },
            'templates': self.templates_generated
        }
        
        filename = "pose_templates.json"
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n{'='*80}")
            print(f"✓ TEMPLATES SAVED: {filename}")
            print(f"{'='*80}")
            print(f"Total templates: {len(self.templates_generated)}")
            
            # Print Python code format
            print(f"\nPython code format for POSE_TEMPLATES:")
            print("POSE_TEMPLATES = {")
            for pose_name, data in self.templates_generated.items():
                angles = data['angles']
                print(f"    '{pose_name}': [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}, {angles[4]:.1f}],")
            print("}")
            
        except Exception as e:
            print(f"\n✗ Error saving templates: {e}")
    
    def input_thread(self):
        """FIXED: Thread for handling keyboard input"""
        print("\n🎮 Input thread started. Type commands and press Enter:")
        
        while self.running:
            try:
                command = input().strip().lower()
                
                if command == 'f':
                    self.generate_template('flat_hand')
                elif command == 'F':
                    self.generate_template('fist')
                elif command == 'g':
                    self.generate_template('grab')
                elif command == 's':
                    self.save_all_templates()
                elif command == 'q':
                    print("\n⏹️  Quit requested...")
                    self.running = False
                    break
                else:
                    print(f"\n❓ Unknown command: {command}")
                    print("Commands: [f]lat_hand  [F]ist  [g]rab  [p]eace  [o]k_sign  [s]ave_all  [q]uit")
                    
            except EOFError:
                break
            except Exception as e:
                print(f"\n✗ Input error: {e}")
    
    async def run(self):
        """Main diagnostic loop"""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC MODE - FIXED VERSION")
        print(f"{'='*80}")
        print("\nThis tool will show you:")
        print("  1. Actual flex sensor angles")
        print("  2. ML model predictions")
        print("  3. IMU quaternion values")
        print("  4. Performance metrics (FPS, frame times)")
        print("\n✨ NEW: Template generation commands:")
        print("  [f] = Generate template for 'flat_hand'")
        print("  [F] = Generate template for 'fist'")
        print("  [g] = Generate template for 'grab'")
        print("  [s] = Save all templates to file")
        print("  [q] = Quit")
        print("\nHold a pose for 6+ seconds, then type the command and press Enter")
        
        print("\nScanning for ESP32...")
        devices = await BleakScanner.discover(timeout=5.0)
        target_device = None
        
        for device in devices:
            if device.name == DEVICE_NAME:
                target_device = device
                break
        
        if not target_device:
            print(f"ERROR: Device '{DEVICE_NAME}' not found!")
            return
        
        print(f"Found: {target_device.name} ({target_device.address})")
        
        print("Connecting...")
        async with BleakClient(target_device.address) as client:
            print(f"Connected: {client.is_connected}\n")
            
            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            print("✓ Subscribed to notifications\n")
            
            # FIXED: Start input thread
            input_thread = threading.Thread(target=self.input_thread, daemon=True)
            input_thread.start()
            
            print("🚀 DIAGNOSTIC RUNNING - Make poses and type commands\n")
            
            try:
                while self.running:
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping...")
                self.running = False
            
            await client.stop_notify(CHARACTERISTIC_UUID)
        
        print("\n✓ Diagnostic ended")
        
        # Print summary
        if len(self.angle_history) > 0:
            print(f"\nCollected {len(self.angle_history)} samples")
            
            # Calculate average angles across all samples
            all_proximal = [s['proximal'] for s in self.angle_history]
            avg_all = np.mean(all_proximal, axis=0)
            std_all = np.std(all_proximal, axis=0)
            
            print("\n📊 OVERALL STATISTICS:")
            print(f"  Thumb:  {avg_all[0]:6.1f}° ± {std_all[0]:5.1f}°")
            print(f"  Index:  {avg_all[1]:6.1f}° ± {std_all[1]:5.1f}°")
            print(f"  Middle: {avg_all[2]:6.1f}° ± {std_all[2]:5.1f}°")
            print(f"  Ring:   {avg_all[3]:6.1f}° ± {std_all[3]:5.1f}°")
            print(f"  Pinky:  {avg_all[4]:6.1f}° ± {std_all[4]:5.1f}°")
        
        # Auto-save templates if any were generated
        if self.templates_generated:
            print("\n📝 Auto-saving templates...")
            self.save_all_templates()


async def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python diagnostic_v2_fixed.py <model_path>")
        print("\nExample:")
        print("  python diagnostic_v2_fixed.py models/flex_to_rotation_model.pth")
        print("\nThis tool helps you:")
        print("  - See actual angles from sensors and ML model")
        print("  - Generate and save pose templates (FIXED!)")
        print("  - Debug IMU orientation")
        print("  - Identify performance bottlenecks")
        print("\nWorkflow:")
        print("  1. Hold a pose for 6+ seconds")
        print("  2. Type command letter (f/F/g) + Enter")
        print("  3. Repeat for all poses")
        print("  4. Type 's' + Enter to save all templates")
        sys.exit(1)
    
    model_path = sys.argv[1]
    tool = DiagnosticTool(model_path)
    await tool.run()


if __name__ == "__main__":
    asyncio.run(main())