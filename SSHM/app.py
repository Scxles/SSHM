# SSHM - Hardware Monitor (Revamp)
# Cross-platform GUI hardware monitor using Tkinter + psutil (+ optional NVIDIA NVML via pynvml)
#
# Goals of this revamp:
# - Accurate CPU usage (avoid back-to-back cpu_percent() calls that break sampling)
# - Better CPU name detection (real model string vs platform.processor() garbage on Windows)
# - Faster UI (no destroying/recreating disk/network widgets every refresh)
# - Separate sampling from UI (background thread collects metrics; UI only renders)

from __future__ import annotations

import platform
import threading
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil
import tkinter as tk

# Optional NVIDIA NVML
GPU_SUPPORTED = False
GPU_ERROR: Optional[str] = None
try:
    import pynvml as N  # type: ignore
    try:
        N.nvmlInit()
        GPU_SUPPORTED = True
    except Exception as e:  # pragma: no cover
        GPU_ERROR = str(e)
except Exception as e:  # pragma: no cover
    GPU_ERROR = str(e)

APP_TITLE = "Hardware Monitor"
SAMPLE_INTERVAL_S = 1.0     # background sampler interval
UI_INTERVAL_MS = 250        # UI repaint interval (fast but light)

COLORS = {
    "bg": "#0a0e27",
    "surface": "#151b3d",
    "surface_light": "#1e2749",
    "chart_bg": "#0f1629",
    "text": "#e2e8f0",
    "text_dim": "#94a3b8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "accent": "#667eea",
    "accent2": "#22c55e",
}

FONTS = {
    "title": ("Segoe UI", 24, "bold"),
    "card_title": ("Segoe UI", 12, "bold"),
    "big": ("Segoe UI", 20, "bold"),
    "med": ("Segoe UI", 12, "bold"),
    "body": ("Segoe UI", 10),
    "mono": ("Consolas", 10),
    "small": ("Segoe UI", 9),
}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def bytes2human(n: float) -> str:
    symbols = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    n = float(n)
    while n >= 1024 and i < len(symbols) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {symbols[i]}"


def bps2human(n: float) -> str:
    # bytes per second
    return f"{bytes2human(n)}/s"


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def get_cpu_brand() -> str:
    """
    Returns a human CPU model string.
    - Windows: registry ProcessorNameString
    - macOS: sysctl machdep.cpu.brand_string
    - Linux: /proc/cpuinfo model name
    Fallback: platform.processor()/machine()
    """
    sysname = platform.system()

    if sysname == "Windows":
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass

        # Fallback (wmic deprecated, but still present on some boxes)
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"], stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            lines = [x.strip() for x in out.splitlines() if x.strip() and "Name" not in x]
            if lines:
                return lines[0]
        except Exception:
            pass

    elif sysname == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            if out.strip():
                return out.strip()
        except Exception:
            pass

    elif sysname == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    cpu_name = platform.processor() or platform.machine() or "CPU"
    return cpu_name.strip() or "CPU"


def pick_cpu_temp_c() -> Optional[float]:
    """
    Best-effort CPU temperature read.
    psutil exposes this only on some platforms/sensors.
    Returns the highest temperature among likely CPU sensors.
    """
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)  # type: ignore[attr-defined]
    except Exception:
        return None

    if not temps:
        return None

    candidates: List[float] = []
    for _, entries in temps.items():
        for e in entries:
            name = (getattr(e, "label", "") or "").lower()
            cur = getattr(e, "current", None)
            if cur is None:
                continue
            # Heuristic: prioritize CPU-ish labels, but accept anything
            if any(k in name for k in ("package", "cpu", "tctl", "core", "ccd", "soc")):
                candidates.append(float(cur))
            else:
                # Some platforms use empty labels; keep as fallback
                candidates.append(float(cur))

    if not candidates:
        return None
    return max(candidates)


@dataclass
class DiskItem:
    mountpoint: str
    fstype: str


@dataclass
class GpuSample:
    name: str
    util: Optional[int] = None
    mem_used: Optional[int] = None  # bytes
    mem_total: Optional[int] = None  # bytes
    temp_c: Optional[int] = None
    power_w: Optional[float] = None


class MetricsSampler(threading.Thread):
    """
    Background sampler to avoid blocking Tk mainloop.
    Stores latest snapshot + short histories for sparklines.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        super().__init__(daemon=True)
        self.interval_s = max(0.25, float(interval_s))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.cpu_brand = get_cpu_brand()
        self.logical_cores = psutil.cpu_count(logical=True) or 0
        self.physical_cores = psutil.cpu_count(logical=False) or 0

        # Prime psutil's cpu_percent sampling so first UI update isn't bogus.
        try:
            psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            pass

        self.prev_disk = psutil.disk_io_counters()
        self.prev_net = psutil.net_io_counters(pernic=True)
        self.prev_t = time.monotonic()

        self.disks: List[DiskItem] = self._load_disks()
        self._last_disk_refresh = time.monotonic()

        # Latest snapshot
        self.snapshot: Dict[str, object] = {}

        # Small histories (last ~60 samples)
        self.hist_len = 60
        self.cpu_hist: List[float] = []
        self.mem_hist: List[float] = []
        self.disk_hist: List[float] = []  # total disk B/s (read+write)
        self.net_hist: List[float] = []   # total net B/s (up+down)

    def _load_disks(self) -> List[DiskItem]:
        items: List[DiskItem] = []
        try:
            parts = psutil.disk_partitions(all=False)
        except Exception:
            return items
        for p in parts:
            if not getattr(p, "fstype", ""):
                continue
            # Skip obvious pseudo mounts
            if platform.system() == "Windows":
                # Skip CD-ROM if no media
                if "cdrom" in (p.opts or "").lower():
                    continue
            items.append(DiskItem(mountpoint=p.mountpoint, fstype=p.fstype))
        # Deduplicate by mountpoint
        seen = set()
        out: List[DiskItem] = []
        for d in items:
            if d.mountpoint in seen:
                continue
            seen.add(d.mountpoint)
            out.append(d)
        return out

    def stop(self) -> None:
        self.stop_event.set()

    def _push_hist(self, arr: List[float], v: float) -> None:
        arr.append(float(v))
        if len(arr) > self.hist_len:
            del arr[0 : len(arr) - self.hist_len]

    def run(self) -> None:
        while not self.stop_event.is_set():
            t0 = time.monotonic()

            # Refresh disk list occasionally (hotplug/USB)
            if t0 - self._last_disk_refresh > 15:
                self.disks = self._load_disks()
                self._last_disk_refresh = t0

            # CPU
            try:
                per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                cpu_total = sum(per_cpu) / len(per_cpu) if per_cpu else 0.0
            except Exception:
                per_cpu = []
                cpu_total = 0.0

            freq_cur = None
            freq_max = None
            try:
                freqs = psutil.cpu_freq(percpu=True)  # may be None
                if freqs:
                    cur_list = [f.current for f in freqs if f and f.current]
                    max_list = [f.max for f in freqs if f and f.max]
                    if cur_list:
                        freq_cur = sum(cur_list) / len(cur_list)
                    if max_list:
                        freq_max = max(max_list)
            except Exception:
                pass

            temp_c = pick_cpu_temp_c()

            # Memory
            try:
                vm = psutil.virtual_memory()
                mem_used = float(vm.used)
                mem_total = float(vm.total)
                mem_pct = float(vm.percent)
            except Exception:
                mem_used = mem_total = mem_pct = 0.0

            try:
                sw = psutil.swap_memory()
                swap_used = float(sw.used)
                swap_total = float(sw.total)
                swap_pct = float(sw.percent)
            except Exception:
                swap_used = swap_total = swap_pct = 0.0

            # Disk IO (global)
            disk_read_bps = disk_write_bps = 0.0
            disk_total_bps = 0.0
            try:
                cur_disk = psutil.disk_io_counters()
                t1 = time.monotonic()
                dt = max(0.001, t1 - self.prev_t)
                disk_read_bps = max(0, cur_disk.read_bytes - self.prev_disk.read_bytes) / dt
                disk_write_bps = max(0, cur_disk.write_bytes - self.prev_disk.write_bytes) / dt
                disk_total_bps = disk_read_bps + disk_write_bps
                self.prev_disk = cur_disk
                self.prev_t = t1
            except Exception:
                pass

            # Disk usage (per mount)
            disk_usage: List[Tuple[str, float, float, float]] = []
            for d in self.disks:
                try:
                    u = psutil.disk_usage(d.mountpoint)
                    disk_usage.append((d.mountpoint, float(u.used), float(u.total), float(u.percent)))
                except Exception:
                    continue

            # Network
            nic_rates: Dict[str, Tuple[float, float]] = {}  # nic -> (up_bps, down_bps)
            total_up = total_down = 0.0
            try:
                cur_net = psutil.net_io_counters(pernic=True)
                t2 = time.monotonic()
                dt = max(0.001, t2 - t0)  # for this tick, good enough

                for nic, ctr in cur_net.items():
                    prev = self.prev_net.get(nic)
                    if prev is None:
                        continue
                    up = max(0, ctr.bytes_sent - prev.bytes_sent) / dt
                    down = max(0, ctr.bytes_recv - prev.bytes_recv) / dt
                    # Keep even if small; UI can decide to hide.
                    nic_rates[nic] = (up, down)
                    total_up += up
                    total_down += down

                self.prev_net = cur_net
            except Exception:
                pass

            # GPU
            gpus: List[GpuSample] = []
            if GPU_SUPPORTED:
                try:
                    count = N.nvmlDeviceGetCount()
                    for i in range(count):
                        h = N.nvmlDeviceGetHandleByIndex(i)
                        name = N.nvmlDeviceGetName(h)
                        if isinstance(name, bytes):
                            name = name.decode(errors="ignore")
                        util = N.nvmlDeviceGetUtilizationRates(h).gpu
                        mem = N.nvmlDeviceGetMemoryInfo(h)
                        temp = None
                        try:
                            temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                        except Exception:
                            temp = None
                        power_w = None
                        try:
                            # milliwatts
                            power_w = float(N.nvmlDeviceGetPowerUsage(h)) / 1000.0
                        except Exception:
                            power_w = None

                        gpus.append(
                            GpuSample(
                                name=str(name),
                                util=int(util),
                                mem_used=int(mem.used),
                                mem_total=int(mem.total),
                                temp_c=int(temp) if temp is not None else None,
                                power_w=power_w,
                            )
                        )
                except Exception:
                    # If NVML flakes out at runtime, treat as unsupported
                    gpus = []

            snap: Dict[str, object] = {
                "time": datetime.now(),
                "cpu_total": float(cpu_total),
                "cpu_per": [float(x) for x in per_cpu],
                "cpu_freq_cur": freq_cur,
                "cpu_freq_max": freq_max,
                "cpu_temp_c": temp_c,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "mem_pct": mem_pct,
                "swap_used": swap_used,
                "swap_total": swap_total,
                "swap_pct": swap_pct,
                "disk_read_bps": disk_read_bps,
                "disk_write_bps": disk_write_bps,
                "disk_usage": disk_usage,
                "net_total_up": total_up,
                "net_total_down": total_down,
                "net_nics": nic_rates,
                "gpus": gpus,
            }

            with self.lock:
                self.snapshot = snap
                self._push_hist(self.cpu_hist, cpu_total)
                self._push_hist(self.mem_hist, mem_pct)
                self._push_hist(self.disk_hist, disk_total_bps)
                self._push_hist(self.net_hist, total_up + total_down)

            # Sleep remainder
            elapsed = time.monotonic() - t0
            time.sleep(max(0.05, self.interval_s - elapsed))


class ModernCard(tk.Frame):
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, bg=COLORS["surface"], highlightthickness=0, **kwargs)

        title_frame = tk.Frame(self, bg=COLORS["surface"])
        title_frame.pack(fill="x", padx=18, pady=(16, 10))

        tk.Label(
            title_frame,
            text=title,
            font=FONTS["card_title"],
            fg=COLORS["text"],
            bg=COLORS["surface"],
        ).pack(side="left")

        self.header_right = tk.Label(
            title_frame,
            text="",
            font=FONTS["small"],
            fg=COLORS["text_dim"],
            bg=COLORS["surface"],
        )
        self.header_right.pack(side="right")

        self.content = tk.Frame(self, bg=COLORS["surface"])
        self.content.pack(fill="both", expand=True, padx=18, pady=(0, 16))


class CircularProgress(tk.Canvas):
    def __init__(self, parent, size=118, width=12):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=COLORS["surface"],
            highlightthickness=0,
            bd=0,
        )
        self.size = size
        self.width = width

        self.create_oval(
            width, width, size - width, size - width,
            outline=COLORS["surface_light"], width=width
        )
        self.arc = self.create_arc(
            width, width, size - width, size - width,
            start=90, extent=0,
            outline=COLORS["success"], width=width, style="arc"
        )
        self.text = self.create_text(
            size // 2, size // 2,
            text="0%", font=("Segoe UI", 18, "bold"), fill=COLORS["text"]
        )

    def set_value(self, value: float) -> None:
        v = clamp(float(value), 0, 100)
        extent = -int(v * 3.6)
        self.itemconfig(self.arc, extent=extent)

        if v < 50:
            col = COLORS["success"]
        elif v < 80:
            col = COLORS["warning"]
        else:
            col = COLORS["danger"]

        self.itemconfig(self.arc, outline=col)
        self.itemconfig(self.text, text=f"{int(round(v))}%")


class MiniBar(tk.Canvas):
    def __init__(self, parent, width=240, height=8):
        super().__init__(
            parent, width=width, height=height,
            bg=COLORS["surface"], highlightthickness=0, bd=0
        )
        self.w = width
        self.h = height
        self.create_rectangle(0, 0, width, height, fill=COLORS["surface_light"], outline="")
        self.bar = self.create_rectangle(0, 0, 0, height, fill=COLORS["accent"], outline="")

    def set_value(self, value: float, color: Optional[str] = None) -> None:
        v = clamp(float(value), 0, 100)
        w = int(self.w * v / 100.0)

        if color:
            col = color
        else:
            if v < 50:
                col = COLORS["success"]
            elif v < 80:
                col = COLORS["warning"]
            else:
                col = COLORS["danger"]

        self.itemconfig(self.bar, fill=col)
        self.coords(self.bar, 0, 0, w, self.h)


class Sparkline(tk.Canvas):
    def __init__(self, parent, width=260, height=46, max_points=60):
        super().__init__(
            parent, width=width, height=height,
            bg=COLORS["chart_bg"], highlightthickness=0, bd=0
        )
        self.w = width
        self.h = height
        self.max_points = max_points
        self.line = None
        self.create_rectangle(0, 0, width, height, outline="", fill=COLORS["chart_bg"])

    def set_series(self, values: List[float], vmin: float, vmax: float, color: str) -> None:
        if not values:
            self.delete("line")
            return

        vals = values[-self.max_points :]
        n = len(vals)
        if n < 2:
            self.delete("line")
            return

        # Normalize
        lo, hi = float(vmin), float(vmax)
        if hi <= lo:
            hi = lo + 1.0

        pts = []
        for i, v in enumerate(vals):
            x = int(i * (self.w - 4) / max(1, n - 1)) + 2
            y_norm = clamp((float(v) - lo) / (hi - lo), 0.0, 1.0)
            y = int((1.0 - y_norm) * (self.h - 4)) + 2
            pts.extend([x, y])

        self.delete("line")
        self.create_line(*pts, fill=color, width=2, smooth=True, tags="line")


class App(tk.Tk):
    def __init__(self, sampler: MetricsSampler):
        super().__init__()
        self.sampler = sampler

        self.title(APP_TITLE)
        self.geometry("1240x820")
        self.minsize(1080, 720)
        self.configure(bg=COLORS["bg"])

        # Main container
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header = tk.Frame(main, bg=COLORS["bg"])
        header.pack(fill="x", pady=(0, 18))

        tk.Label(
            header,
            text="⚡ Hardware Monitor",
            font=FONTS["title"],
            fg=COLORS["text"],
            bg=COLORS["bg"],
        ).pack(side="left")

        self.time_label = tk.Label(
            header, text="", font=FONTS["body"], fg=COLORS["text_dim"], bg=COLORS["bg"]
        )
        self.time_label.pack(side="right")

        # Grid layout: 2 columns, 3 rows
        grid = tk.Frame(main, bg=COLORS["bg"])
        grid.pack(fill="both", expand=True)

        for c in range(2):
            grid.grid_columnconfigure(c, weight=1, uniform="col")
        for r in range(3):
            grid.grid_rowconfigure(r, weight=1)

        # Cards
        self.cpu_card = self._build_cpu_card(grid)
        self.mem_card = self._build_memory_card(grid)
        self.gpu_card = self._build_gpu_card(grid)
        self.disk_card = self._build_disk_card(grid)
        self.net_card = self._build_network_card(grid)

        self.cpu_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.mem_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        self.gpu_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.disk_card.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        self.net_card.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Footer
        footer = tk.Frame(main, bg=COLORS["bg"])
        footer.pack(fill="x", pady=(12, 0))

        self.status_label = tk.Label(
            footer,
            text=f"{platform.system()} {platform.release()} • Python {platform.python_version()} • psutil {psutil.__version__}",
            font=FONTS["small"],
            fg=COLORS["text_dim"],
            bg=COLORS["bg"],
        )
        self.status_label.pack(side="left")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI tick
        self.after(100, self.ui_tick)

    # ---------- CPU ----------
    def _build_cpu_card(self, parent) -> ModernCard:
        card = ModernCard(parent, "CPU")

        top = tk.Frame(card.content, bg=COLORS["surface"])
        top.pack(fill="x", pady=(0, 10))

        self.cpu_circle = CircularProgress(top)
        self.cpu_circle.pack(side="left", padx=(0, 16))

        info = tk.Frame(top, bg=COLORS["surface"])
        info.pack(side="left", fill="both", expand=True)

        self.cpu_name = tk.Label(
            info,
            text="Detecting CPU…",
            font=FONTS["body"],
            fg=COLORS["text_dim"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.cpu_name.pack(fill="x", pady=(6, 2))

        self.cpu_freq = tk.Label(
            info,
            text="-- MHz",
            font=FONTS["big"],
            fg=COLORS["text"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.cpu_freq.pack(fill="x")

        self.cpu_meta = tk.Label(
            info,
            text="",
            font=FONTS["body"],
            fg=COLORS["text_dim"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.cpu_meta.pack(fill="x", pady=(2, 0))

        # Sparkline + temp
        mid = tk.Frame(card.content, bg=COLORS["surface"])
        mid.pack(fill="x", pady=(6, 10))

        left = tk.Frame(mid, bg=COLORS["surface"])
        left.pack(side="left", fill="x", expand=True)

        tk.Label(
            left, text="Usage (last ~60s)", font=FONTS["small"],
            fg=COLORS["text_dim"], bg=COLORS["surface"]
        ).pack(anchor="w")
        self.cpu_spark = Sparkline(left, width=320, height=46)
        self.cpu_spark.pack(anchor="w", pady=(6, 0))

        right = tk.Frame(mid, bg=COLORS["surface"])
        right.pack(side="right")

        self.cpu_temp = tk.Label(
            right, text="Temp: --", font=FONTS["med"], fg=COLORS["text"], bg=COLORS["surface"]
        )
        self.cpu_temp.pack(anchor="e", pady=(18, 0))

        # Divider
        tk.Frame(card.content, height=1, bg=COLORS["surface_light"]).pack(fill="x", pady=(6, 10))

        tk.Label(
            card.content,
            text="Per-core",
            font=FONTS["small"],
            fg=COLORS["text_dim"],
            bg=COLORS["surface"],
            anchor="w",
        ).pack(fill="x")

        self.core_frame = tk.Frame(card.content, bg=COLORS["surface"])
        self.core_frame.pack(fill="both", expand=True, pady=(8, 0))

        self.core_bars: List[MiniBar] = []
        self._build_core_bars(self.sampler.logical_cores)

        return card

    def _build_core_bars(self, n: int) -> None:
        # Grid 4 columns
        for w in self.core_frame.winfo_children():
            w.destroy()
        self.core_bars = []

        cols = 4
        for i in range(n):
            r = i // cols
            c = i % cols
            cell = tk.Frame(self.core_frame, bg=COLORS["surface"])
            cell.grid(row=r, column=c, sticky="ew", padx=6, pady=4)
            tk.Label(
                cell,
                text=f"C{i}",
                font=("Segoe UI", 8),
                fg=COLORS["text_dim"],
                bg=COLORS["surface"],
                width=3,
            ).pack(side="left")
            bar = MiniBar(cell, width=120, height=5)
            bar.pack(side="left", padx=(6, 0))
            self.core_bars.append(bar)

        # Make columns expand evenly
        for c in range(cols):
            self.core_frame.grid_columnconfigure(c, weight=1)

    # ---------- Memory ----------
    def _build_memory_card(self, parent) -> ModernCard:
        card = ModernCard(parent, "Memory")

        # RAM
        ram = tk.Frame(card.content, bg=COLORS["surface"])
        ram.pack(fill="x", pady=(0, 12))

        tk.Label(ram, text="RAM", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"]).pack(anchor="w")
        self.mem_bar = MiniBar(ram, width=330, height=8)
        self.mem_bar.pack(anchor="w", pady=(6, 8))
        self.mem_label = tk.Label(
            ram, text="-- / --", font=FONTS["big"], fg=COLORS["text"], bg=COLORS["surface"], anchor="w"
        )
        self.mem_label.pack(fill="x")

        # Sparkline
        tk.Label(
            ram, text="Usage (last ~60s)", font=FONTS["small"],
            fg=COLORS["text_dim"], bg=COLORS["surface"]
        ).pack(anchor="w", pady=(10, 0))
        self.mem_spark = Sparkline(ram, width=330, height=46)
        self.mem_spark.pack(anchor="w", pady=(6, 0))

        tk.Frame(card.content, height=1, bg=COLORS["surface_light"]).pack(fill="x", pady=(14, 12))

        # Swap
        sw = tk.Frame(card.content, bg=COLORS["surface"])
        sw.pack(fill="x")

        tk.Label(sw, text="Swap", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"]).pack(anchor="w")
        self.swap_bar = MiniBar(sw, width=330, height=8)
        self.swap_bar.pack(anchor="w", pady=(6, 8))
        self.swap_label = tk.Label(
            sw, text="-- / --", font=FONTS["med"], fg=COLORS["text"], bg=COLORS["surface"], anchor="w"
        )
        self.swap_label.pack(fill="x")

        return card

    # ---------- GPU ----------
    def _build_gpu_card(self, parent) -> ModernCard:
        card = ModernCard(parent, "GPU")

        self.gpu_container = tk.Frame(card.content, bg=COLORS["surface"])
        self.gpu_container.pack(fill="both", expand=True)

        self.gpu_empty = tk.Label(
            self.gpu_container,
            text="Detecting GPU…",
            font=FONTS["body"],
            fg=COLORS["text_dim"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
        )
        self.gpu_empty.pack(fill="both", expand=True)

        # Rows will be built dynamically and updated
        self.gpu_rows: List[Dict[str, object]] = []
        return card

    def _ensure_gpu_rows(self, gpus: List[GpuSample]) -> None:
        # If count changed, rebuild rows
        if len(self.gpu_rows) == len(gpus):
            return

        for w in self.gpu_container.winfo_children():
            w.destroy()
        self.gpu_rows = []

        if not gpus:
            msg = "🔌 No NVIDIA GPU stats available.\n\n"
            if GPU_SUPPORTED:
                msg += "NVML is available, but no GPUs were returned."
            else:
                msg += "Install NVIDIA drivers (NVML ships with the driver)."
                if GPU_ERROR:
                    msg += f"\n\nDetails: {GPU_ERROR}"
            lbl = tk.Label(
                self.gpu_container,
                text=msg,
                font=FONTS["body"],
                fg=COLORS["text_dim"],
                bg=COLORS["surface"],
                justify="left",
                anchor="nw",
            )
            lbl.pack(fill="both", expand=True)
            return

        for idx, g in enumerate(gpus):
            row = tk.Frame(self.gpu_container, bg=COLORS["surface"])
            row.pack(fill="x", pady=(0, 10) if idx < len(gpus) - 1 else 0)

            name = tk.Label(row, text=g.name, font=FONTS["med"], fg=COLORS["text"], bg=COLORS["surface"], anchor="w")
            name.pack(fill="x")

            util_line = tk.Frame(row, bg=COLORS["surface"])
            util_line.pack(fill="x", pady=(6, 0))
            tk.Label(util_line, text="Util", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"]).pack(side="left")
            util_bar = MiniBar(util_line, width=280, height=6)
            util_bar.pack(side="left", padx=(8, 0))
            util_val = tk.Label(util_line, text="--%", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"])
            util_val.pack(side="left", padx=(10, 0))

            mem_line = tk.Frame(row, bg=COLORS["surface"])
            mem_line.pack(fill="x", pady=(6, 0))
            tk.Label(mem_line, text="VRAM", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"]).pack(side="left")
            mem_bar = MiniBar(mem_line, width=280, height=6)
            mem_bar.pack(side="left", padx=(8, 0))
            mem_val = tk.Label(mem_line, text="-- / --", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"])
            mem_val.pack(side="left", padx=(10, 0))

            meta = tk.Label(row, text="", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"], anchor="w")
            meta.pack(fill="x", pady=(6, 0))

            self.gpu_rows.append({
                "util_bar": util_bar,
                "util_val": util_val,
                "mem_bar": mem_bar,
                "mem_val": mem_val,
                "meta": meta,
            })

    # ---------- Disk ----------
    def _build_disk_card(self, parent) -> ModernCard:
        card = ModernCard(parent, "Disk")

        top = tk.Frame(card.content, bg=COLORS["surface"])
        top.pack(fill="x", pady=(0, 10))

        self.disk_io = tk.Label(
            top, text="↓ --/s  ↑ --/s", font=FONTS["med"], fg=COLORS["text"], bg=COLORS["surface"], anchor="w"
        )
        self.disk_io.pack(fill="x")

        tk.Label(
            top, text="Total I/O (last ~60s)", font=FONTS["small"],
            fg=COLORS["text_dim"], bg=COLORS["surface"]
        ).pack(anchor="w", pady=(8, 0))
        self.disk_spark = Sparkline(top, width=330, height=46)
        self.disk_spark.pack(anchor="w", pady=(6, 0))

        tk.Frame(card.content, height=1, bg=COLORS["surface_light"]).pack(fill="x", pady=(12, 10))

        self.disk_rows_frame = tk.Frame(card.content, bg=COLORS["surface"])
        self.disk_rows_frame.pack(fill="both", expand=True)

        self.disk_rows: Dict[str, Dict[str, object]] = {}  # mount -> widgets
        return card

    def _ensure_disk_rows(self, mounts: List[str]) -> None:
        # add new mounts
        for m in mounts:
            if m in self.disk_rows:
                continue
            row = tk.Frame(self.disk_rows_frame, bg=COLORS["surface"])
            row.pack(fill="x", pady=6)

            name = tk.Label(row, text=f"💾 {m}", font=FONTS["body"], fg=COLORS["text"], bg=COLORS["surface"])
            name.pack(anchor="w")

            bar = MiniBar(row, width=300, height=6)
            bar.pack(anchor="w", pady=(6, 0))

            info = tk.Label(row, text="-- / --", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"])
            info.pack(anchor="w", pady=(4, 0))

            self.disk_rows[m] = {"bar": bar, "info": info}

    # ---------- Network ----------
    def _build_network_card(self, parent) -> ModernCard:
        card = ModernCard(parent, "Network")

        top = tk.Frame(card.content, bg=COLORS["surface"])
        top.pack(fill="x", pady=(0, 10))

        self.net_total = tk.Label(
            top, text="↑ --/s  ↓ --/s", font=FONTS["med"], fg=COLORS["text"], bg=COLORS["surface"], anchor="w"
        )
        self.net_total.pack(fill="x")

        tk.Label(
            top, text="Total throughput (last ~60s)", font=FONTS["small"],
            fg=COLORS["text_dim"], bg=COLORS["surface"]
        ).pack(anchor="w", pady=(8, 0))
        self.net_spark = Sparkline(top, width=690, height=46)
        self.net_spark.pack(anchor="w", pady=(6, 0))

        tk.Frame(card.content, height=1, bg=COLORS["surface_light"]).pack(fill="x", pady=(12, 10))

        self.net_rows_frame = tk.Frame(card.content, bg=COLORS["surface"])
        self.net_rows_frame.pack(fill="both", expand=True)

        self.net_rows: Dict[str, Dict[str, object]] = {}  # nic -> widgets
        return card

    def _ensure_net_rows(self, nics: List[str]) -> None:
        for nic in nics:
            if nic in self.net_rows:
                continue
            row = tk.Frame(self.net_rows_frame, bg=COLORS["surface"])
            row.pack(fill="x", pady=6)

            name = tk.Label(row, text=f"🌐 {nic}", font=FONTS["body"], fg=COLORS["text"], bg=COLORS["surface"])
            name.grid(row=0, column=0, sticky="w")

            up = tk.Label(row, text="↑ --/s", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"])
            up.grid(row=0, column=1, sticky="e", padx=(20, 10))

            down = tk.Label(row, text="↓ --/s", font=FONTS["small"], fg=COLORS["text_dim"], bg=COLORS["surface"])
            down.grid(row=0, column=2, sticky="e")

            row.grid_columnconfigure(0, weight=1)

            self.net_rows[nic] = {"up": up, "down": down}

    # ---------- UI Tick ----------
    def ui_tick(self) -> None:
        # time label always
        self.time_label.config(text=datetime.now().strftime("%I:%M:%S %p"))

        with self.sampler.lock:
            snap = dict(self.sampler.snapshot)
            cpu_hist = list(self.sampler.cpu_hist)
            mem_hist = list(self.sampler.mem_hist)
            disk_hist = list(self.sampler.disk_hist)
            net_hist = list(self.sampler.net_hist)

        if snap:
            try:
                self._render(snap, cpu_hist, mem_hist, disk_hist, net_hist)
                self.status_label.config(
                    text=f"{platform.system()} {platform.release()} • Python {platform.python_version()} • psutil {psutil.__version__}"
                )
            except Exception as e:
                self.status_label.config(text=f"Render error: {e}")

        self.after(UI_INTERVAL_MS, self.ui_tick)

    def _render(
        self,
        snap: Dict[str, object],
        cpu_hist: List[float],
        mem_hist: List[float],
        disk_hist: List[float],
        net_hist: List[float],
    ) -> None:
        # CPU card
        cpu_total = float(snap.get("cpu_total", 0.0))
        self.cpu_circle.set_value(cpu_total)
        self.cpu_name.config(text=self.sampler.cpu_brand)

        freq_cur = snap.get("cpu_freq_cur", None)
        freq_max = snap.get("cpu_freq_max", None)
        if isinstance(freq_cur, (int, float)):
            if isinstance(freq_max, (int, float)) and freq_max > 0:
                self.cpu_freq.config(text=f"{freq_cur:.0f} MHz  •  Max {freq_max:.0f} MHz")
            else:
                self.cpu_freq.config(text=f"{freq_cur:.0f} MHz")
        else:
            self.cpu_freq.config(text="-- MHz")

        self.cpu_meta.config(
            text=f"Cores: {self.sampler.physical_cores} • Threads: {self.sampler.logical_cores}"
        )

        temp_c = snap.get("cpu_temp_c", None)
        if isinstance(temp_c, (int, float)):
            self.cpu_temp.config(text=f"Temp: {temp_c:.0f}°C")
        else:
            self.cpu_temp.config(text="Temp: N/A")

        self.cpu_spark.set_series(cpu_hist, 0, 100, COLORS["accent"])

        per = snap.get("cpu_per", [])
        if isinstance(per, list) and per:
            # If core count changes (rare), rebuild
            if len(per) != len(self.core_bars):
                self._build_core_bars(len(per))
            for bar, pct in zip(self.core_bars, per):
                try:
                    bar.set_value(float(pct))
                except Exception:
                    bar.set_value(0.0)

        # Memory card
        mem_used = float(snap.get("mem_used", 0.0))
        mem_total = float(snap.get("mem_total", 0.0))
        mem_pct = float(snap.get("mem_pct", 0.0))
        self.mem_bar.set_value(mem_pct)
        self.mem_label.config(text=f"{bytes2human(mem_used)} / {bytes2human(mem_total)}")
        self.mem_spark.set_series(mem_hist, 0, 100, COLORS["accent2"])

        swap_used = float(snap.get("swap_used", 0.0))
        swap_total = float(snap.get("swap_total", 0.0))
        swap_pct = float(snap.get("swap_pct", 0.0))
        self.swap_bar.set_value(swap_pct)
        self.swap_label.config(text=f"{bytes2human(swap_used)} / {bytes2human(swap_total)}")

        # GPU card
        gpus = snap.get("gpus", [])
        if isinstance(gpus, list):
            # list of GpuSample
            gpus2: List[GpuSample] = []
            for g in gpus:
                if isinstance(g, GpuSample):
                    gpus2.append(g)
            self._ensure_gpu_rows(gpus2)
            if gpus2 and len(self.gpu_rows) == len(gpus2):
                for row, g in zip(self.gpu_rows, gpus2):
                    util = g.util if g.util is not None else 0
                    row["util_bar"].set_value(float(util))
                    row["util_val"].config(text=f"{util}%")

                    if g.mem_total and g.mem_total > 0 and g.mem_used is not None:
                        pct = 100.0 * safe_div(float(g.mem_used), float(g.mem_total))
                        row["mem_bar"].set_value(pct, COLORS["accent"])
                        row["mem_val"].config(text=f"{bytes2human(g.mem_used)} / {bytes2human(g.mem_total)}")
                    else:
                        row["mem_bar"].set_value(0.0, COLORS["surface_light"])
                        row["mem_val"].config(text="-- / --")

                    meta_bits = []
                    if g.temp_c is not None:
                        meta_bits.append(f"🌡 {g.temp_c}°C")
                    if g.power_w is not None:
                        meta_bits.append(f"⚡ {g.power_w:.0f} W")
                    row["meta"].config(text=" • ".join(meta_bits))

        # Disk card
        r_bps = float(snap.get("disk_read_bps", 0.0))
        w_bps = float(snap.get("disk_write_bps", 0.0))
        self.disk_io.config(text=f"↓ {bps2human(r_bps)}   ↑ {bps2human(w_bps)}")
        # autoscale sparkline to max seen
        vmax = max(disk_hist) if disk_hist else 1.0
        self.disk_spark.set_series(disk_hist, 0, max(1.0, vmax), COLORS["accent"])

        du = snap.get("disk_usage", [])
        mounts: List[str] = []
        if isinstance(du, list):
            for item in du:
                if isinstance(item, tuple) and len(item) == 4:
                    mounts.append(str(item[0]))
        self._ensure_disk_rows(mounts)

        if isinstance(du, list):
            for item in du:
                if not (isinstance(item, tuple) and len(item) == 4):
                    continue
                m, used, total, pct = item
                m = str(m)
                row = self.disk_rows.get(m)
                if not row:
                    continue
                row["bar"].set_value(float(pct))
                row["info"].config(text=f"{bytes2human(used)} / {bytes2human(total)} ({pct:.0f}%)")

        # Network card
        up = float(snap.get("net_total_up", 0.0))
        down = float(snap.get("net_total_down", 0.0))
        self.net_total.config(text=f"↑ {bps2human(up)}   ↓ {bps2human(down)}")
        vmax = max(net_hist) if net_hist else 1.0
        self.net_spark.set_series(net_hist, 0, max(1.0, vmax), COLORS["accent2"])

        nics = snap.get("net_nics", {})
        if isinstance(nics, dict):
            # show active-ish NICs first
            ordered = sorted(
                nics.items(),
                key=lambda kv: (kv[1][0] + kv[1][1]),
                reverse=True,
            )
            # Keep top 10 to avoid clutter
            ordered = ordered[:10]
            names = [k for k, _ in ordered]
            self._ensure_net_rows(names)
            for nic, (u, d) in ordered:
                row = self.net_rows.get(nic)
                if not row:
                    continue
                row["up"].config(text=f"↑ {bps2human(float(u))}")
                row["down"].config(text=f"↓ {bps2human(float(d))}")

    def on_close(self) -> None:
        try:
            self.sampler.stop()
        except Exception:
            pass
        try:
            if GPU_SUPPORTED:
                N.nvmlShutdown()
        except Exception:
            pass
        self.destroy()


def main() -> None:
    sampler = MetricsSampler(interval_s=SAMPLE_INTERVAL_S)
    sampler.start()

    app = App(sampler)
    app.mainloop()


if __name__ == "__main__":
    main()
