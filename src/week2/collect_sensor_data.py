import asyncio
from bleak import BleakClient, BleakScanner
import struct
import time
import json
import os

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
SAVE_DIR     = "sensor_data"
BATCH_SIZE   = 50          # save in batches
COLLECTION_DURATION = 60  # seconds

os.makedirs(SAVE_DIR, exist_ok=True)

sensor_buffer = []

def handle_sensor_data(sender, data: bytearray):
    try:
        timestamp_us = time.time()  # host timestamp in seconds
        # parse 50 bytes (5×uint16 + 3×float + 3×float + 4×float)
        flex = struct.unpack('<5H', data[0:10])
        accel = struct.unpack('<3f', data[10:22])
        gyro  = struct.unpack('<3f', data[22:34])
        quat  = struct.unpack('<4f', data[34:50])

        sample = {
            'timestamp': timestamp_us,
            'flex': list(flex),
            'accel': list(accel),
            'gyro': list(gyro),
            'quat': list(quat)
        }
        sensor_buffer.append(sample)

        # save batch
        if len(sensor_buffer) >= BATCH_SIZE:
            for s in sensor_buffer:
                filename = f"{SAVE_DIR}/sensor_{int(s['timestamp']*1e6)}.json"
                with open(filename, 'w') as f:
                    json.dump(s, f)
            sensor_buffer.clear()

    except Exception as e:
        print(f"Error parsing sensor packet: {e}")

async def main():
    print("Scanning for NEURA_GLOVE...")
    devices = await BleakScanner.discover()
    target = None
    for d in devices:
        if d.name == "NEURA_GLOVE":
            target = d
            break
    if not target:
        print("NEURA_GLOVE not found!")
        return

    async with BleakClient(target.address) as client:
        print("Connected to NEURA_GLOVE")
        await client.start_notify(CHAR_UUID, handle_sensor_data)
        print(f"Streaming sensor data for {COLLECTION_DURATION}s...")

        start_time = time.time()
        try:
            while (time.time() - start_time) < COLLECTION_DURATION:
                await asyncio.sleep(0.01)  # sleep briefly to allow async notifications

        except KeyboardInterrupt:
            print("Stopping early due to KeyboardInterrupt...")

        finally:
            await client.stop_notify(CHAR_UUID)
            # save any remaining samples
            for s in sensor_buffer:
                filename = f"{SAVE_DIR}/sensor_{int(s['timestamp']*1e6)}.json"
                with open(filename, 'w') as f:
                    json.dump(s, f)
            sensor_buffer.clear()

            print(f"Collection complete. Data saved to {SAVE_DIR}.")

if __name__ == "__main__":
    asyncio.run(main())
