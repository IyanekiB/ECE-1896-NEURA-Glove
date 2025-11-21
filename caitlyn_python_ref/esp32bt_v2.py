import sys
import asyncio                 #Allows tasks run asynchronously

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        from bleak.backends.winrt.util import allow_sta, uninitialize_sta
    except ImportError:
        allow_sta = None  # type: ignore
        uninitialize_sta = None  # type: ignore
    else:
        uninitialize_sta()
        allow_sta()
else:
    uninitialize_sta = None  # type: ignore
    allow_sta = None  # type: ignore

from bleak import BleakClient, BleakScanner  #Python Bluetooth Low Energy Library
from bleak.exc import BleakDeviceNotFoundError
import matplotlib.pyplot as plt       #Plotting sensor data
import time                           #Keep track of time for plots 
import numpy as np 

#UUID of ESP32 - found by doing a scan 
ESP32_ADDRESS = "BF755A30-4365-A657-B4FC-3325F7231B2A"
ESP32_NAME = "ESP32-BLE"

#Service and Characteristic UUIDs (must match ESp32 sketch)
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

NUM_SENSORS = 5

#Data storage 
times = []
flex_data = []
window_size = 100

#Plot set-up 
plt.close('all')
plt.figure("Flex Sensor Voltage vs Time")
plt.ion()
plt.title("Flex Sensor Voltage vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.ylim(0,3.5)

colors = ['m', 'c', 'r', 'g', 'b']
labels = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']

#Creates five line objects for plotting, plus an empty list of lists for flex data
lines = []
for i in range(NUM_SENSORS):
    line, = plt.plot([], [], colors[i] +'-', label=labels[i])
    lines.append(line)
    flex_data.append([])

plt.legend()
plt.grid(True)

#Keep record of start time
start_time = time.time()


#Prints out data that is received from ESP32 
def notification_handler(sender, data):
    
    #Gets string of data transmitted from ESP32 
    decoded = data.decode('utf-8').strip()
    #Decodes the values stored in string, values are separated by commas
    values = decoded.split(',')
    
    try:
        #Convert values to float values for plotting 
        flex_values = [float(v) for v in values]  
        flex1, flex2, flex3, flex4, flex5 = flex_values
        
        #Get the current time 
        t = time.time() - start_time
        
        times.append(t)
        
        print(f"Flex1={flex1}, Flex2={flex2}, Flex3={flex3}, Flex4={flex4}, Flex5={flex5}")
        for i in range(NUM_SENSORS):
            flex_data[i].append(flex_values[i])
            
        
        #Calculate SNR (Only does Flex1 Sensor for now)
        if len(flex_data[0]) >= window_size:
            signal = np.array(flex_data[0])
            mean = np.mean(signal)
            variance = np.var(signal)
            if variance > 0:
                snr = (mean**2) / variance 
            else: 
                snr = float('inf')
            
            print()
            print(f"(Flex1) SNR = {snr:.2f}, Mean: {mean:.3f}, Variance: {variance: .6f}")
            
        #If the length of the list is greater than the window size -> pop firts item out 
        if len(flex_data[0]) > window_size:
            times.pop(0)
            for i in range(NUM_SENSORS):   
                flex_data[i].pop(0)
                
        #Update each line on the same plot
        for i in range(NUM_SENSORS):
            lines[i].set_xdata(times)
            lines[i].set_ydata(flex_data[i])

        #Updates the x axis to scroll with time 
        plt.xlim(times[-1] - 10, times[-1] + 0.5) #show last 10 seconds 
        plt.pause(0.001)
        
    except ValueError:
        print("Bad data format:", decoded)


#Async allows tasks to run concurrently - listen for Bluetooth & keep program running
async def main():
    target = ESP32_ADDRESS
    if ESP32_NAME and ESP32_ADDRESS.count(":") != 5:
        device = await BleakScanner.find_device_by_address(ESP32_ADDRESS, timeout=5.0)
        if device is None:
            print(f"Address '{ESP32_ADDRESS}' not found, scanning by name '{ESP32_NAME}'...")
            device = await BleakScanner.find_device_by_filter(
                lambda d, _: d and d.name == ESP32_NAME,
                timeout=10.0,
            )
        if device is None:
            raise BleakDeviceNotFoundError(ESP32_ADDRESS, f"Device '{ESP32_NAME}' not found during scan.")
        target = device.address

    #Connect to ESP32 using its address
    async with BleakClient(target) as client:
        print("Connected to ESP32")
        
        #Subscribe to notifications with the characteristic UUID
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Listening... press Ctrl+C to stop.")
        
        #Keep script running and listening
        while True:
            await asyncio.sleep(1)

def _run():
    if sys.platform.startswith("win"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    else:
        asyncio.run(main())


if __name__ == "__main__":
    _run()
