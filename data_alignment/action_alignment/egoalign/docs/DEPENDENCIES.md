# Qualified environment, not a clean-install lock

This is the environment observed for the 2026-10-08 local fresh-source build and
new-launcher qualification. Existing environments were reused; clean-machine
installation remains unverified. This list is not a portable dependency lock.

| Component | Qualified version |
|---|---|
| Python | 3.10.20 |
| GNU C++ / CMake build | GCC 11.4; Release configuration |
| CUDA compiler / TensorRT | 12.4 / 10.13.3.9 |
| ROS | ROS 2 Humble, source its setup for shared libraries |
| Data Torch / torchvision | 2.12.0 (CUDA 13.0 build) / 0.27.0 |
| Data NumPy / SciPy | 1.26.4 / 1.15.3 |
| Data Pinocchio distribution `pin` | 3.9.0 |
| Data pandas / PyArrow | 2.3.3 / 24.0.0 |
| Data datasets / LeRobot | 4.8.5 / 0.1.0 (legacy `lerobot.common.datasets` API) |
| Data PyAV / OpenCV | 17.1.0 / 4.11.0.86 |
| Data ONNX Runtime / pyzmq | 1.23.2 / 27.1.0 |
| Data MuJoCo / simulation MuJoCo | 3.8.1 / 3.10.0 |
| tyro / h5py / huggingface-hub | 1.0.15 / 3.16.0 / 0.35.3 |
| Pillow / pytest | 12.3.0 / 9.1.1 |

The controller uses its C++ ONNX Runtime/TensorRT libraries, not necessarily the
data interpreter's Python ONNX Runtime. Compiler CUDA and PyTorch CUDA builds
are also distinct. Follow upstream SDK compatibility requirements; do not install
these versions blindly into a working training environment.

LeRobot version strings alone cannot identify local modifications; use
[`configs/qualified_dependencies.json`](../configs/qualified_dependencies.json)
alongside this table.
The installed package records public LeRobot Git revision
`b536f47e3ff8c3b340fc5efa52f0ece0a7212a57`. The inventory hashes 100 installed
Python files, identifying tested content without claiming unmodified upstream.
Regenerate with `python tools/dependency_inventory.py` in the data environment.
The complete external dependency source distribution is not vendored here.

Build commands, pinned public source and dependency discovery are documented in
[public_runtime.md](public_runtime.md). The successful test used NVIDIA GPU EGL,
not Mesa llvmpipe: set `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` and
verify the EGL vendor as described there. Ports are explicitly configurable;
never reuse an active deployment's ports. Run one replay lane for the qualified
sample recipe. Metadata/video packaging additionally uses `ffmpeg`/`ffprobe`.
