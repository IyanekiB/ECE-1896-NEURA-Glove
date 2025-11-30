import asyncio                 #Allows tasks run asynchronously
from bleak import BleakScanner, BleakClient  #Python Bluetooth Low Energy Library
import matplotlib.pyplot as plt       #Plotting sensor data
import time                           #Keep track of time for plots 
import numpy as np
import socket                         #UDP communication 

#UUID of ESP32 - found by doing a scan 
ESP32_ADDRESS = "BF755A30-4365-A657-B4FC-3325F7231B2A"

DEVICE_NAME = "ESP32-BLE"

#Service and Characteristic UUIDs (must match ESp32 sketch)
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # ESP32 -> PC
CHARACTERISTIC_UUID_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # PC -> ESP32

# UDP Configuration
UDP_IP = "127.0.0.1"  # Listen on localhost
UDP_PORT = 5555       # Port to listen on

NUM_SENSORS = 5

#Data storage 
times = []
flex_data = []
window_size = 100

#Plot set-up 
#plt.close('all')
#plt.figure("Flex Sensor Voltage vs Time")
#plt.ion()
#plt.title("Flex Sensor Voltage vs Time")
#plt.xlabel("Time (s)")
#plt.ylabel("Voltage (V)")
#plt.ylim(0,3.5)

colors = ['m', 'c', 'r', 'g', 'b']
labels = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']

#Creates five line objects for plotting, plus an empty list of lists for flex data
#lines = []
#for i in range(NUM_SENSORS):
#    line, = plt.plot([], [], colors[i] +'-', label=labels[i])
#    lines.append(line)
#    flex_data.append([])

#plt.legend()
#plt.grid(True)

#Keep record of start time
start_time = time.time()
send_time = 0  # Global variable to track when commands are sent


#Prints out data that is received from ESP32 
async def notification_handler(sender, data):
    global send_time
    
    #Gets string of data transmitted from ESP32 
    decoded = data.decode('utf-8').strip()
    
    if decoded.startswith("MOTOR:"):
        # Motor trigger notification 
        esp_time = int(decoded.split(":")[1])
        receive_time = time.time() * 1000
        motor_latency = receive_time - send_time 
        # Standout print
        print("\n" + "="*60)
        print(f"MOTOR TRIGGERED")
        print(f"ESP32 timestamp: {esp_time} ms")
        print(f"Latency: {motor_latency:.2f} ms")
        print("="*60 + "\n")
        return 
    
    elif decoded.startswith("SENSOR:"):
        # Sensor data notification
        sensor_data_str = decoded[len("SENSOR:"):]  # Remove prefix
        
    else:
        print("Unknown notification type:", decoded)
        return
    
    #Decodes the values stored in string, values are separated by commas
    values = sensor_data_str.split(',')
    
    # print("Received")

    
    try:
        # Parse float values for flex sensors and IMU
        flex_values = [float(v) for v in values[:5]]
        flex2, flex5, flex4, flex3, flex1 = flex_values

        qw, qx, qy, qz = [float(v) for v in values[5:9]]
        ax, ay, az = [float(v) for v in values[9:12]]
        gx, gy, gz = [float(v) for v in values[12:15]]

        # Parse joystick values
        joyX, joyY, cPressed, zPressed = [int(v) for v in values[15:19]]

        # Parse touch sensor values (last 5)
        ct_th, ct_in, ct_mi, ct_r, ct_p = [int(v) for v in values[-5:]]
       
        #Get the current time 
        t = time.time() - start_time
        
        times.append(t)
        
    
        #for i in range(NUM_SENSORS):
        #    flex_data[i].append(flex_values[i])
        
        # print("=== Sensor Readings ===")
        # print(f"Flex Sensors:  Flex1={flex1:.3f}  Flex2={flex2:.3f}  Flex3={flex3:.3f}  Flex4={flex4:.3f}  Flex5={flex5:.3f}\n")

        # print(f"Quaternions:  qw={qw:.3f}  qx={qx:.3f}  qy={qy:.3f}  qz={qz:.3f}")
        # print(f"Accelerometer:  ax={ax:.3f}  ay={ay:.3f}  az={az:.3f}")
        # print(f"Gyroscope:      gx={gx:.3f}  gy={gy:.3f}  gz={gz:.3f}\n")

        # print(f"Joystick: X={joyX}  Y={joyY}  |  Buttons: C={int(cPressed)}  Z={int(zPressed)}\n")

        # print(f"Contact Sensors → Thumb: {ct_th}, Index: {ct_in}, Middle: {ct_mi}, Ring: {ct_r}, Pinky: {ct_p}")
        # print("="*50)

        
        #If the length of the list is greater than the window size -> pop firts item out 
        #if len(flex_data[0]) > window_size:
        #   times.pop(0)
        #    for i in range(NUM_SENSORS):   
        #        flex_data[i].pop(0)
                
        #Update each line on the same plot
        #for i in range(NUM_SENSORS):
        #   lines[i].set_xdata(times)
        #    lines[i].set_ydata(flex_data[i])

        #Updates the x axis to scroll with time 
        #plt.xlim(times[-1] - 10, times[-1] + 0.5) #show last 10 seconds 
        #plt.pause(0.001)
        
    except ValueError:
        print("Bad data format:", decoded)

# async def send_periodic(client: BleakClient, interval: float = 5.0):
#     """Send '1' every `interval` seconds."""
#     global send_time
#     while True:
#         try:
#             send_time = time.time() * 1000  # seconds since epoch
#             await client.write_gatt_char(CHARACTERISTIC_UUID_RX, b"1", response=False)
#             print(f"Sent: 1 at {int(send_time)} ms")
#         except Exception as e:
#             print(f"Failed to send command: {e}")
#         await asyncio.sleep(interval)

async def udp_server(client: BleakClient):
    """Listen for UDP messages and send them to ESP32."""
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    
    print(f"UDP Server listening on {UDP_IP}:{UDP_PORT}")
    print("Send messages to trigger ESP32 commands")
    
    while True:
        try:
            # Non-blocking receive with asyncio
            data, addr = await loop.sock_recvfrom(sock, 1024)
            message = data.decode('utf-8').strip()
            
            print(f"\n[UDP] Received '{message}' from {addr}")
            
            # Send the received message to ESP32
            global send_time
            send_time = time.time() * 1000
            await client.write_gatt_char(CHARACTERISTIC_UUID_RX,  b"1", response=False)
            print(f"[UDP] Forwarded to ESP32: {message}")
            
        except Exception as e:
            print(f"UDP error: {e}")
            await asyncio.sleep(0.1)

async def main():
    print("Scanning for ESP32...")
    print("  Looking for device with UART service or matching name...")
    
    # Use new scanning style with advertisement data
    devices = await BleakScanner.discover(timeout=10.0, return_adv=True)
    esp32 = None
    
    # Try to find ESP32 by name OR by UART service UUID OR by known address
    for address, (device, adv_data) in devices.items():
        # Check by name first
        if device.name == DEVICE_NAME:
            esp32 = device
            print(f"  Found by name: {device.name} ({device.address})")
            break
        
        # Check by UART service UUID (for devices that don't advertise name properly)
        if SERVICE_UUID.lower() in [uuid.lower() for uuid in adv_data.service_uuids]:
            esp32 = device
            print(f"  Found by UART service: {device.address}")
            if device.name:
                print(f"  Device name: {device.name}")
            else:
                print(f"  Device name: Unknown (Windows compatibility issue)")
            break
        
        # Check by known MAC address
        if device.address == "04:83:08:0E:7C:7E":
            esp32 = device
            print(f"  Found by MAC address: {device.address}")
            if device.name:
                print(f"  Device name: {device.name}")
            else:
                print(f"  Device name: Unknown")
            break

    if not esp32:
        print(f"ERROR: No device with name '{DEVICE_NAME}', UART service, or known address found!")
        print(f"  Expected service UUID: {SERVICE_UUID}")
        print(f"  Expected MAC address: 04:83:08:0E:7C:7E")
        return

    print(f"Connecting to: {esp32.address}")

    async with BleakClient(esp32.address) as client:
        if not client.is_connected:
            print("Failed to connect to ESP32")
            return
        print("Connected to ESP32")

        # Start listening for notifications
        await client.start_notify(CHARACTERISTIC_UUID_TX, notification_handler)
        print("Listening for notifications...")

        # Start the UDP server task
        udp_task = asyncio.create_task(udp_server(client))
        
        # Start the periodic sending task (optional - comment out if you only want UDP triggers)
        # periodic_task = asyncio.create_task(send_periodic(client, interval=5.0))

        try:
            # Keep running while still able to handle notifications
            await asyncio.Event().wait()  # Wait forever until program is stopped

        except asyncio.CancelledError:
            print("Stopping BLE communication...")

        finally:
            udp_task.cancel()
            # periodic_task.cancel()  # Uncomment if using periodic task
            await client.stop_notify(CHARACTERISTIC_UUID_TX)
            print("Notifications stopped")

if __name__ == "__main__":
    asyncio.run(main())

