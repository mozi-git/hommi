#!/usr/bin/env python3
from gi.repository import Aravis

Aravis.update_device_list()
n_devices = Aravis.get_n_devices()

print(f"Found {n_devices} camera(s):")
for i in range(n_devices):
    device_id = Aravis.get_device_id(i)
    print(f"\nCamera {i}:")
    print(f"  Device ID: {device_id}")
    
    camera = Aravis.Camera.new(device_id)
    if camera:
        print(f"  Model: {camera.get_model_name()}")
        print(f"  Vendor: {camera.get_vendor_name()}")
        print(f"  Serial: {camera.get_device_serial_number()}")
        
        # Try to get MAC address
        try:
            mac = camera.get_string("GevMACAddress")
            print(f"  MAC Address: {mac}")
        except:
            pass