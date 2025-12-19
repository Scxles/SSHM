# Hardware Monitor (Revamp)

This is a revamp of the original Tkinter hardware monitor.

## What was fixed/improved

- **CPU name**: shows the real CPU model string (Windows registry / macOS sysctl / Linux /proc/cpuinfo) instead of `Intel64 Family ...`.
- **CPU usage accuracy**: avoids back-to-back `psutil.cpu_percent()` calls that break sampling; CPU sampling is primed once, then updated on a stable interval.
- **UI performance**: disk + network rows are **updated in place** (no destroy/recreate every refresh), which removes the stutter.
- **No UI blocking**: hardware polling runs in a **background thread**; Tk mainloop only renders.

## Run (Windows)

1. Install Python 3.10+
2. Double-click `start_windows.bat`

## Notes

- NVIDIA GPU stats require NVIDIA drivers (NVML comes with the driver). If NVML isn't available, the GPU card will show a clear message.
