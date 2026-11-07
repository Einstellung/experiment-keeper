# Experiment Keeper

A small Linux user service that monitors NVIDIA GPUs and starts a continuous CUDA workload on devices it has claimed. It records device state, restores claims after a restart, and grows each worker's own GPU-memory allocation toward 80% of the device's capacity.

## Behavior

- Every 60 seconds, inspect each GPU. A GPU is eligible when it has no compute processes, its utilization is 0%, and its reported memory use is at most 256 MiB. The monitor checks again before claiming it.
- Save claimed device UUIDs to disk and start one worker per claim. Workers continuously run cuBLAS SGEMM and retry memory allocation every 60 seconds in chunks of at most 256 MiB. The target is 80% of total GPU memory per worker, subject to at least 1,024 MiB reported free at allocation time.
- Log status, claims, allocation attempts, external compute processes, and worker exits. Failed workers are restarted. Existing claims are restored when the monitor restarts.
- External compute processes do **not** cause a claimed GPU to be released. Concurrent workloads share compute and memory, and may run slower or fail to allocate memory. Sampling and rechecking do not provide an exclusive reservation against other schedulers.

## Requirements

- Linux, Python 3, an NVIDIA driver and `nvidia-smi`.
- CUDA 11 runtime (`libcudart.so.11.0`) and cuBLAS 11 (`libcublas.so.11`) available through the dynamic linker. The source uses Python's standard library and `ctypes`.
- A user-level systemd manager for the included service.

## Installation

Copy the program to your home directory and install the user unit:

```sh
install -m 755 gpu-idle-monitor.py "$HOME/gpu-idle-monitor.py"
install -Dm 644 gpu-idle-monitor.service "$HOME/.config/systemd/user/gpu-idle-monitor.service"
systemctl --user daemon-reload
systemctl --user enable --now gpu-idle-monitor.service
```

For the user service to start when the machine boots without an interactive login, enable lingering for that user if it is not already enabled: `loginctl enable-linger "$USER"` (administrator authorization may be required).

## Inspect and stop

```sh
systemctl --user status gpu-idle-monitor.service
tail -F "$HOME/.local/share/gpu-idle-monitor/monitor.log"
systemctl --user stop gpu-idle-monitor.service
systemctl --user disable gpu-idle-monitor.service
```

`stop` ends the monitor and its workers but preserves their claims. A later start restores them. `disable` prevents automatic starts but does not stop an already running service. To discard all claims, first stop the service, then remove `~/.local/share/gpu-idle-monitor/claims.json`.

To exclude devices, stop the service and write `~/.local/share/gpu-idle-monitor/excluded.json` as a JSON array of GPU UUIDs, such as `["GPU-..."]`, then start the service. Exclusions are loaded at startup. Use `nvidia-smi --query-gpu=index,uuid --format=csv` to identify devices.

Logs are rotated at 2 MiB with three backups. The memory reservation is a target, not a guarantee: available memory can change between sampling and allocation, and CUDA allocation can fail.

## Testing

Run the offline tests with `python3 -m unittest discover -s tests -v`. They do not use a GPU. Test real CUDA operation separately on an appropriate idle device.
