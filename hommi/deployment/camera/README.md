# Camera Setup

HoMMI deployment uses GigE Vision cameras through Aravis. Camera streaming code
lives in `hommi/deployment/camera`, and camera names/serials are configured in
`hommi/deployment/config/camera.yaml`.

## Network

Put the camera network interface on the same subnet as the cameras. For the
current RBY1 setup this is typically a static address such as `192.168.88.2/24`.
Use the interface name reported by `ip link`:

```bash
sudo ip addr flush dev <interface>
sudo ip addr add 192.168.88.2/24 dev <interface>
sudo ip link set <interface> up
sudo ip link set <interface> mtu 9000
```

If the machine uses an Intel X710/fiber card, make sure the `i40e` driver is
installed and loaded. Driver installation is machine-specific; follow the Intel
or distribution instructions for your kernel and Secure Boot setup.

## Aravis

Install Aravis, Aravis tools, and Python GI bindings. One working setup is:

```bash
sudo apt update
sudo apt install aravis-tools gstreamer1.0-plugins-bad \
  libxml2-dev gobject-introspection libgirepository1.0-dev
conda install -c conda-forge pygobject
```

If Aravis was built from source under `/usr/local`, expose its typelib path:

```bash
export GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0:$GI_TYPELIB_PATH
```

Check that cameras are discoverable:

```bash
arv-tool-0.8 list
arv-viewer-0.8
```

## HoMMI Config

Edit `hommi/deployment/config/camera.yaml` and set `camera_map` entries to the
camera identifiers returned by Aravis. The keys should match the observation
names expected by the policy, for example:

```yaml
camera_map:
  camera_head_main_rgb: "FLIR-Blackfly S BFS-PGE-50S5C-25260985"
  camera_left_main_rgb: "FLIR-Blackfly S BFS-PGE-23S3C-24260091"
```

Use `mock_mode: true` in the config when hardware is unavailable and you only
need to validate the software pipeline.

## Validate Streaming

From the HoMMI repo root:

```bash
python -m hommi.deployment.camera_stream_viewer --list
python -m hommi.deployment.camera_stream_viewer
```

## Troubleshooting

- If cameras appear on the network but Aravis cannot discover or stream from
them, check firewall rules. Temporarily disabling `ufw` can isolate the issue.
- If streaming is unstable, verify MTU/jumbo-frame settings on the camera NIC
and switch.
- If `gi.repository.Aravis` cannot be imported, check that PyGObject is installed
in the active environment and that `GI_TYPELIB_PATH` includes the Aravis typelib
location.
