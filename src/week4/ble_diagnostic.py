"""
BLE Diagnostic Tool
Helps diagnose connection issues with ESP32-BLE device
"""

import asyncio
import sys
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "ESP32-BLE"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

async def scan_devices():
    """Scan for all BLE devices"""
    print("=" * 60)
    print("BLE DIAGNOSTIC TOOL")
    print("=" * 60)
    print("\n[1/4] Scanning for BLE devices (10 seconds)...")
    print("      Please wait...\n")
    
    devices = await BleakScanner.discover(timeout=10.0)
    
    print(f"Found {len(devices)} BLE devices:\n")
    
    target_found = False
    for i, device in enumerate(devices, 1):
        is_target = device.name == DEVICE_NAME
        marker = " <-- TARGET DEVICE" if is_target else ""
        rssi = getattr(device, 'rssi', 'N/A')
        print(f"  {i}. Name: {device.name or '(Unknown)'}")
        print(f"     Address: {device.address}")
        print(f"     RSSI: {rssi} dBm{marker}" if rssi != 'N/A' else f"     RSSI: N/A{marker}")
        print()
        
        if is_target:
            target_found = True
    
    return devices, target_found

async def test_connection(device):
    """Test connection to specific device"""
    print(f"\n[2/4] Testing connection to {device.name} ({device.address})...")
    
    try:
        async with BleakClient(device.address, timeout=20.0) as client:
            print(f"      [OK] Connected successfully!")
            print(f"      Connection state: {client.is_connected}")
            
            print(f"\n[3/4] Discovering services...")
            services = await client.get_services()
            
            print(f"      Found {len(services)} services:\n")
            
            target_service_found = False
            target_char_found = False
            
            for service in services:
                is_target = service.uuid.lower() == SERVICE_UUID.lower()
                marker = " <-- TARGET SERVICE" if is_target else ""
                print(f"      Service: {service.uuid}{marker}")
                
                if is_target:
                    target_service_found = True
                
                for char in service.characteristics:
                    is_target_char = char.uuid.lower() == CHARACTERISTIC_UUID.lower()
                    char_marker = " <-- TARGET CHARACTERISTIC" if is_target_char else ""
                    props = ", ".join(char.properties)
                    print(f"        - Characteristic: {char.uuid}")
                    print(f"          Properties: {props}{char_marker}")
                    
                    if is_target_char:
                        target_char_found = True
                print()
            
            print(f"\n[4/4] Verification:")
            print(f"      Target service found: {'YES' if target_service_found else 'NO'}")
            print(f"      Target characteristic found: {'YES' if target_char_found else 'NO'}")
            
            if target_char_found:
                print(f"\n      [4/4] Testing notifications...")
                
                def notification_handler(sender, data):
                    print(f"      [OK] Received data: {data[:50]}...")
                
                try:
                    await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                    print(f"      [OK] Subscribed to notifications")
                    print(f"      Waiting 3 seconds for data...")
                    await asyncio.sleep(3)
                    await client.stop_notify(CHARACTERISTIC_UUID)
                    print(f"      [OK] Notifications working!")
                except Exception as e:
                    print(f"      [ERROR] Notification test failed: {e}")
            
            return True
            
    except asyncio.TimeoutError:
        print(f"      [ERROR] Connection timeout (>20 seconds)")
        print(f"      This usually means:")
        print(f"        - Device is out of range")
        print(f"        - Device is already connected to another client")
        print(f"        - Windows Bluetooth driver issue")
        return False
        
    except Exception as e:
        print(f"      [ERROR] Connection failed: {e}")
        print(f"      Error type: {type(e).__name__}")
        return False

async def main():
    print("\nThis tool will help diagnose BLE connection issues\n")
    
    # Scan for devices
    devices, target_found = await scan_devices()
    
    if not devices:
        print("\n" + "=" * 60)
        print("DIAGNOSIS: No BLE devices found")
        print("=" * 60)
        print("\nPossible causes:")
        print("  1. Bluetooth is disabled on your PC")
        print("  2. No BLE devices are nearby or powered on")
        print("  3. Bluetooth driver issues")
        print("\nSolutions:")
        print("  1. Check: Settings > Bluetooth & devices")
        print("  2. Make sure ESP32 is powered on")
        print("  3. Try: Device Manager > Bluetooth > Update driver")
        return
    
    if not target_found:
        print("\n" + "=" * 60)
        print(f"DIAGNOSIS: '{DEVICE_NAME}' not found")
        print("=" * 60)
        print("\nPossible causes:")
        print("  1. ESP32 is not powered on")
        print("  2. ESP32 firmware is not running (check serial monitor)")
        print("  3. ESP32 is already connected to another device")
        print("  4. ESP32 is out of Bluetooth range")
        print("\nSolutions:")
        print("  1. Power cycle the ESP32")
        print("  2. Upload the firmware again")
        print("  3. Check ESP32 serial output for errors")
        print("  4. Move ESP32 closer to your PC")
        return
    
    # Find target device
    target_device = None
    for device in devices:
        if device.name == DEVICE_NAME:
            target_device = device
            break
    
    # Test connection
    success = await test_connection(target_device)
    
    if success:
        print("\n" + "=" * 60)
        print("DIAGNOSIS: Everything looks good!")
        print("=" * 60)
        print("\nYour BLE connection should work fine.")
        print("If realtime_inference.py still fails, try:")
        print("  1. Close any other apps using Bluetooth")
        print("  2. Restart the ESP32")
        print("  3. Run the script as Administrator")
    else:
        print("\n" + "=" * 60)
        print("DIAGNOSIS: Connection issues detected")
        print("=" * 60)
        print("\nCommon Windows BLE issues:")
        print("  1. Python needs to run as Administrator")
        print("  2. Windows Privacy Settings blocking Bluetooth")
        print("  3. Antivirus blocking Bluetooth access")
        print("  4. ESP32 already paired in Windows (unpair it!)")
        print("\nSolutions:")
        print("  1. Right-click PowerShell > Run as Administrator")
        print("  2. Settings > Privacy > Bluetooth > Allow apps")
        print("  3. Temporarily disable antivirus")
        print("  4. Settings > Bluetooth > Remove ESP32-BLE if paired")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nScan cancelled by user")

