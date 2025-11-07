
# Simple Hardware Monitor (GUI)

A lightweight, cross‑platform hardware monitor written in Python with Tkinter. It shows:
- CPU overall and per‑core usage (and frequency)
- RAM and swap usage
- NVIDIA GPU stats (utilization, memory, temperature) if drivers + NVML are present
- Disk usage per mount and per‑second read/write
- Network throughput per interface

## Quick start (Windows — easiest)
1. **Install Python 3.10+** from python.org (check "Add Python to PATH").
2. Double‑click `start_windows.bat`. It will install requirements (first run only) and start the app.
   - If Windows warns about running a script, click **More info → Run anyway** (you can open it in Notepad to inspect).

## macOS
1. Install Python 3.10+ (e.g., via python.org or Homebrew).
2. Open Terminal and run: `chmod +x start_mac.command`, then **double‑click** it in Finder.
   - The first run installs requirements, later runs skip that.

## Linux
1. Make sure Python 3.10+ and pip are installed.
2. Run `chmod +x start_linux.sh` once, then double‑click it (depending on your file manager) or run `./start_linux.sh`.

## Notes
- **NVIDIA GPU metrics** require NVIDIA drivers and NVML (provided with drivers). If not available, the GPU section will say so gracefully.
- This app uses only Tkinter (bundled with Python), `psutil`, and `pynvml`. No CLI knowledge needed—use the start scripts.
- You can also open the folder in VS Code and press the **Run** button on `app.py` if you prefer.

## Uninstall
Delete the folder. To remove the packages from your system Python, you can uninstall with:
```
pip uninstall psutil pynvml
```
