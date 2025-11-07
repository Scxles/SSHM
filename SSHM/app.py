#SSHM - Sam Scales

import sys
import time
import platform
import psutil
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# Try NVIDIA NVML
gpu_supported = False
gpu_error = None
try:
    import pynvml as N
    N.nvmlInit()
    gpu_supported = True
except Exception as e:
    gpu_error = str(e)

APP_TITLE = "Hardware Monitor"
REFRESH_MS = 1000

# Modern color scheme
COLORS = {
    'bg': '#0a0e27',
    'surface': '#151b3d',
    'surface_light': '#1e2749',
    'accent': '#667eea',
    'accent_light': '#764ba2',
    'text': '#e2e8f0',
    'text_dim': '#94a3b8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'chart_bg': '#0f1629'
}

def bytes2human(n: float) -> str:
    symbols = ('B', 'KB', 'MB', 'GB', 'TB')
    i = 0
    while n >= 1024 and i < len(symbols) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {symbols[i]}"

class ModernCard(tk.Frame):
    """A modern card container with rounded corners effect"""
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        
        # Title
        if title:
            title_frame = tk.Frame(self, bg=COLORS['surface'])
            title_frame.pack(fill='x', padx=20, pady=(20, 10))
            
            tk.Label(
                title_frame,
                text=title,
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['surface']
            ).pack(side='left')
        
        # Content area
        self.content = tk.Frame(self, bg=COLORS['surface'])
        self.content.pack(fill='both', expand=True, padx=20, pady=(0, 20))

class CircularProgress(tk.Canvas):
    """Modern circular progress indicator"""
    def __init__(self, parent, size=120, width=12, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        bg=COLORS['surface'], highlightthickness=0, **kwargs)
        self.size = size
        self.width = width
        self.value = 0
        
        # Background circle
        self.create_oval(
            self.width, self.width,
            size - self.width, size - self.width,
            outline=COLORS['surface_light'], width=self.width
        )
        
        # Progress arc
        self.progress_arc = self.create_arc(
            self.width, self.width,
            size - self.width, size - self.width,
            start=90, extent=0,
            outline=COLORS['accent'], width=self.width,
            style='arc'
        )
        
        # Center text
        self.text = self.create_text(
            size // 2, size // 2,
            text="0%", font=('Segoe UI', 18, 'bold'),
            fill=COLORS['text']
        )
    
    def set_value(self, value, color=None):
        self.value = max(0, min(100, value))
        extent = -int(self.value * 3.6)
        self.itemconfig(self.progress_arc, extent=extent)
        
        if color:
            self.itemconfig(self.progress_arc, outline=color)
        else:
            # Dynamic color based on value
            if self.value < 50:
                col = COLORS['success']
            elif self.value < 80:
                col = COLORS['warning']
            else:
                col = COLORS['danger']
            self.itemconfig(self.progress_arc, outline=col)
        
        self.itemconfig(self.text, text=f"{int(self.value)}%")

class MiniBar(tk.Canvas):
    """Mini horizontal progress bar"""
    def __init__(self, parent, width=200, height=6, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['surface'], highlightthickness=0, **kwargs)
        self.w = width
        self.h = height
        
        # Background
        self.create_rectangle(0, 0, width, height, fill=COLORS['surface_light'], outline='')
        
        # Progress
        self.bar = self.create_rectangle(0, 0, 0, height, fill=COLORS['accent'], outline='')
    
    def set_value(self, value, color=None):
        value = max(0, min(100, value))
        w = int(self.w * value / 100)
        
        if color:
            col = color
        else:
            if value < 50:
                col = COLORS['success']
            elif value < 80:
                col = COLORS['warning']
            else:
                col = COLORS['danger']
        
        self.itemconfig(self.bar, fill=col)
        self.coords(self.bar, 0, 0, w, self.h)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.configure(bg=COLORS['bg'])
        
        # Remove default window styling
        try:
            self.iconbitmap(default="")
        except:
            pass
        
        # Main container with padding
        main = tk.Frame(self, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(main, bg=COLORS['bg'])
        header.pack(fill='x', pady=(0, 20))
        
        tk.Label(
            header,
            text="⚡ Hardware Monitor",
            font=('Segoe UI', 24, 'bold'),
            fg=COLORS['text'],
            bg=COLORS['bg']
        ).pack(side='left')
        
        self.time_label = tk.Label(
            header,
            text="",
            font=('Segoe UI', 11),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        self.time_label.pack(side='right')
        
        # Top row: CPU and Memory
        top_row = tk.Frame(main, bg=COLORS['bg'])
        top_row.pack(fill='both', expand=True, pady=(0, 15))
        
        top_row.grid_columnconfigure(0, weight=1)
        top_row.grid_columnconfigure(1, weight=1)
        top_row.grid_rowconfigure(0, weight=1)
        
        self.create_cpu_card(top_row).grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        self.create_memory_card(top_row).grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        
        # Middle row: GPU and Disk
        mid_row = tk.Frame(main, bg=COLORS['bg'])
        mid_row.pack(fill='both', expand=True, pady=(0, 15))
        
        mid_row.grid_columnconfigure(0, weight=1)
        mid_row.grid_columnconfigure(1, weight=1)
        mid_row.grid_rowconfigure(0, weight=1)
        
        self.create_gpu_card(mid_row).grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        self.create_disk_card(mid_row).grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        
        # Bottom row: Network
        self.create_network_card(main).pack(fill='both', expand=True)
        
        # Footer
        footer = tk.Frame(main, bg=COLORS['bg'], height=30)
        footer.pack(fill='x', pady=(15, 0))
        
        self.status_label = tk.Label(
            footer,
            text=f"{platform.system()} {platform.release()} • Python {platform.python_version()}",
            font=('Segoe UI', 9),
            fg=COLORS['text_dim'],
            bg=COLORS['bg']
        )
        self.status_label.pack(side='left')
        
        # Initialize
        self.prev_disk = psutil.disk_io_counters(perdisk=True)
        self.prev_net = psutil.net_io_counters(pernic=True)
        self.prev_net_time = time.time()
        
        self.after(100, self.refresh)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_cpu_card(self, parent):
        card = ModernCard(parent, title="CPU")
        
        # Top: circular progress
        top = tk.Frame(card.content, bg=COLORS['surface'])
        top.pack(fill='x', pady=(0, 15))
        
        self.cpu_circle = CircularProgress(top, size=120)
        self.cpu_circle.pack(side='left', padx=(0, 20))
        
        # Info
        info = tk.Frame(top, bg=COLORS['surface'])
        info.pack(side='left', fill='both', expand=True)
        
        self.cpu_name = tk.Label(
            info,
            text="Detecting...",
            font=('Segoe UI', 10),
            fg=COLORS['text_dim'],
            bg=COLORS['surface'],
            anchor='w'
        )
        self.cpu_name.pack(fill='x', pady=(10, 5))
        
        self.cpu_freq = tk.Label(
            info,
            text="-- MHz",
            font=('Segoe UI', 20, 'bold'),
            fg=COLORS['text'],
            bg=COLORS['surface'],
            anchor='w'
        )
        self.cpu_freq.pack(fill='x')
        
        self.cpu_temp = tk.Label(
            info,
            text="",
            font=('Segoe UI', 10),
            fg=COLORS['text_dim'],
            bg=COLORS['surface'],
            anchor='w'
        )
        self.cpu_temp.pack(fill='x', pady=(5, 0))
        
        # Separator
        tk.Frame(card.content, height=1, bg=COLORS['surface_light']).pack(fill='x', pady=10)
        
        # Per-core bars
        cores_label = tk.Label(
            card.content,
            text="Cores",
            font=('Segoe UI', 10, 'bold'),
            fg=COLORS['text_dim'],
            bg=COLORS['surface'],
            anchor='w'
        )
        cores_label.pack(fill='x', pady=(0, 10))
        
        self.core_frame = tk.Frame(card.content, bg=COLORS['surface'])
        self.core_frame.pack(fill='both', expand=True)
        
        self.core_bars = []
        
        return card
    
    def create_memory_card(self, parent):
        card = ModernCard(parent, title="Memory")
        
        # RAM
        ram_frame = tk.Frame(card.content, bg=COLORS['surface'])
        ram_frame.pack(fill='x', pady=(0, 20))
        
        ram_label = tk.Label(
            ram_frame,
            text="RAM",
            font=('Segoe UI', 10),
            fg=COLORS['text_dim'],
            bg=COLORS['surface']
        )
        ram_label.pack(anchor='w', pady=(0, 5))
        
        self.mem_bar = MiniBar(ram_frame, width=300, height=8)
        self.mem_bar.pack(anchor='w', pady=(0, 8))
        
        self.mem_label = tk.Label(
            ram_frame,
            text="-- / --",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text'],
            bg=COLORS['surface'],
            anchor='w'
        )
        self.mem_label.pack(fill='x')
        
        # Separator
        tk.Frame(card.content, height=1, bg=COLORS['surface_light']).pack(fill='x', pady=15)
        
        # SWAP
        swap_frame = tk.Frame(card.content, bg=COLORS['surface'])
        swap_frame.pack(fill='x')
        
        swap_label = tk.Label(
            swap_frame,
            text="Swap",
            font=('Segoe UI', 10),
            fg=COLORS['text_dim'],
            bg=COLORS['surface']
        )
        swap_label.pack(anchor='w', pady=(0, 5))
        
        self.swap_bar = MiniBar(swap_frame, width=300, height=8)
        self.swap_bar.pack(anchor='w', pady=(0, 8))
        
        self.swap_label = tk.Label(
            swap_frame,
            text="-- / --",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text'],
            bg=COLORS['surface'],
            anchor='w'
        )
        self.swap_label.pack(fill='x')
        
        return card
    
    def create_gpu_card(self, parent):
        card = ModernCard(parent, title="GPU")
        
        self.gpu_text = tk.Text(
            card.content,
            font=('Consolas', 10),
            fg=COLORS['text'],
            bg=COLORS['chart_bg'],
            relief='flat',
            wrap='word',
            height=8
        )
        self.gpu_text.pack(fill='both', expand=True)
        
        return card
    
    def create_disk_card(self, parent):
        card = ModernCard(parent, title="Disk")
        
        # Custom styled frame for disk info
        self.disk_frame = tk.Frame(card.content, bg=COLORS['surface'])
        self.disk_frame.pack(fill='both', expand=True)
        
        return card
    
    def create_network_card(self, parent):
        card = ModernCard(parent, title="Network")
        
        self.net_frame = tk.Frame(card.content, bg=COLORS['surface'])
        self.net_frame.pack(fill='both', expand=True)
        
        return card
    
    def build_core_bars(self):
        for w in self.core_frame.winfo_children():
            w.destroy()
        
        self.core_bars = []
        percpu = psutil.cpu_percent(interval=None, percpu=True)
        
        for idx in range(len(percpu)):
            row = idx // 4
            col = idx % 4
            
            frm = tk.Frame(self.core_frame, bg=COLORS['surface'])
            frm.grid(row=row, column=col, sticky='ew', padx=5, pady=3)
            
            tk.Label(
                frm,
                text=f"C{idx}",
                font=('Segoe UI', 8),
                fg=COLORS['text_dim'],
                bg=COLORS['surface'],
                width=3
            ).pack(side='left')
            
            bar = MiniBar(frm, width=100, height=4)
            bar.pack(side='left', padx=(5, 0))
            
            self.core_bars.append(bar)
    
    def refresh(self):
        try:
            # Update time
            self.time_label.config(text=datetime.now().strftime("%I:%M:%S %p"))
            
            # CPU
            cpu_pct = psutil.cpu_percent(interval=None)
            self.cpu_circle.set_value(cpu_pct)
            
            cpu_name = platform.processor() or platform.machine()
            if len(cpu_name) > 50:
                cpu_name = cpu_name[:47] + "..."
            self.cpu_name.config(text=cpu_name)
            
            freq = psutil.cpu_freq()
            if freq:
                self.cpu_freq.config(text=f"{freq.current:.0f} MHz")
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            self.cpu_temp.config(text=f"🌡 {entries[0].current:.0f}°C")
                            break
            except:
                pass
            
            # Core bars
            if not self.core_bars:
                self.build_core_bars()
            
            percpu = psutil.cpu_percent(interval=None, percpu=True)
            for bar, pct in zip(self.core_bars, percpu):
                bar.set_value(pct)
            
            # Memory
            vm = psutil.virtual_memory()
            self.mem_bar.set_value(vm.percent)
            self.mem_label.config(text=f"{bytes2human(vm.used)} / {bytes2human(vm.total)}")
            
            sw = psutil.swap_memory()
            self.swap_bar.set_value(sw.percent)
            self.swap_label.config(text=f"{bytes2human(sw.used)} / {bytes2human(sw.total)}")
            
            # GPU
            self.update_gpu()
            
            # Disk
            self.update_disks()
            
            # Network
            self.update_network()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
        finally:
            self.after(REFRESH_MS, self.refresh)
    
    def update_gpu(self):
        self.gpu_text.config(state='normal')
        self.gpu_text.delete('1.0', 'end')
        
        if not gpu_supported:
            self.gpu_text.insert('end', "🔌 NVIDIA GPU not detected\n\n")
            self.gpu_text.insert('end', "No NVIDIA drivers or NVML library found.\n")
            if gpu_error:
                self.gpu_text.insert('end', f"\nDetails: {gpu_error}")
        else:
            try:
                count = N.nvmlDeviceGetCount()
                if count == 0:
                    self.gpu_text.insert('end', "No NVIDIA GPUs detected.")
                
                for i in range(count):
                    h = N.nvmlDeviceGetHandleByIndex(i)
                    name = N.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode()
                    
                    util = N.nvmlDeviceGetUtilizationRates(h)
                    mem = N.nvmlDeviceGetMemoryInfo(h)
                    
                    temp = None
                    try:
                        temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                    except:
                        pass
                    
                    self.gpu_text.insert('end', f"🎮 {name}\n\n")
                    self.gpu_text.insert('end', f"  Utilization:  {util.gpu}%\n")
                    self.gpu_text.insert('end', f"  Memory:       {mem.used//(1024**2)} / {mem.total//(1024**2)} MB ({mem.used*100//mem.total}%)\n")
                    if temp:
                        self.gpu_text.insert('end', f"  Temperature:  {temp}°C\n")
                    
                    if i < count - 1:
                        self.gpu_text.insert('end', "\n" + "─" * 40 + "\n\n")
                        
            except Exception as e:
                self.gpu_text.insert('end', f"GPU Error: {e}")
        
        self.gpu_text.config(state='disabled')
    
    def update_disks(self):
        for w in self.disk_frame.winfo_children():
            w.destroy()
        
        cur = psutil.disk_io_counters(perdisk=True)
        partitions = psutil.disk_partitions(all=False)
        
        for idx, p in enumerate(partitions):
            if not p.fstype:
                continue
            
            try:
                usage = psutil.disk_usage(p.mountpoint)
            except:
                continue
            
            # Calculate read/write speeds
            readps = writeps = 0
            if p.device in self.prev_disk and p.device in cur:
                dt = REFRESH_MS / 1000.0
                readps = max(0, cur[p.device].read_bytes - self.prev_disk[p.device].read_bytes) / dt
                writeps = max(0, cur[p.device].write_bytes - self.prev_disk[p.device].write_bytes) / dt
            
            # Create disk item
            item = tk.Frame(self.disk_frame, bg=COLORS['surface'])
            item.pack(fill='x', pady=5)
            
            tk.Label(
                item,
                text=f"💾 {p.mountpoint}",
                font=('Segoe UI', 10),
                fg=COLORS['text'],
                bg=COLORS['surface']
            ).pack(anchor='w')
            
            bar = MiniBar(item, width=250, height=6)
            bar.set_value(usage.percent)
            bar.pack(anchor='w', pady=5)
            
            info = tk.Label(
                item,
                text=f"{bytes2human(usage.used)} / {bytes2human(usage.total)} • ↓ {bytes2human(readps)}/s ↑ {bytes2human(writeps)}/s",
                font=('Segoe UI', 9),
                fg=COLORS['text_dim'],
                bg=COLORS['surface']
            )
            info.pack(anchor='w')
        
        self.prev_disk = cur
    
    def update_network(self):
        for w in self.net_frame.winfo_children():
            w.destroy()
        
        cur = psutil.net_io_counters(pernic=True)
        dt = time.time() - self.prev_net_time
        
        for nic, counters in cur.items():
            # Skip loopback and inactive
            if nic.startswith('lo') or counters.bytes_sent == 0 and counters.bytes_recv == 0:
                continue
            
            sentps = recvps = 0
            if nic in self.prev_net and dt > 0:
                sentps = max(0, counters.bytes_sent - self.prev_net[nic].bytes_sent) / dt
                recvps = max(0, counters.bytes_recv - self.prev_net[nic].bytes_recv) / dt
            
            # Create network item
            item = tk.Frame(self.net_frame, bg=COLORS['surface'])
            item.pack(fill='x', pady=8)
            
            tk.Label(
                item,
                text=f"🌐 {nic}",
                font=('Segoe UI', 10, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['surface']
            ).pack(anchor='w')
            
            stats = tk.Frame(item, bg=COLORS['surface'])
            stats.pack(fill='x', pady=5)
            
            # Upload
            up = tk.Frame(stats, bg=COLORS['surface'])
            up.pack(side='left', padx=(0, 30))
            
            tk.Label(
                up,
                text="↑ Upload",
                font=('Segoe UI', 8),
                fg=COLORS['text_dim'],
                bg=COLORS['surface']
            ).pack(anchor='w')
            
            tk.Label(
                up,
                text=f"{bytes2human(sentps)}/s",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['accent'],
                bg=COLORS['surface']
            ).pack(anchor='w')
            
            # Download
            down = tk.Frame(stats, bg=COLORS['surface'])
            down.pack(side='left')
            
            tk.Label(
                down,
                text="↓ Download",
                font=('Segoe UI', 8),
                fg=COLORS['text_dim'],
                bg=COLORS['surface']
            ).pack(anchor='w')
            
            tk.Label(
                down,
                text=f"{bytes2human(recvps)}/s",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['success'],
                bg=COLORS['surface']
            ).pack(anchor='w')
        
        self.prev_net = cur
        self.prev_net_time = time.time()
    
    def on_close(self):
        try:
            if gpu_supported:
                N.nvmlShutdown()
        except:
            pass
        self.destroy()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()