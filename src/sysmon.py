"""
SYSTEM METRICS MONITOR
======================
Samples hardware usage in a background thread while code runs.
Used during training on Kaggle AND during inference on the Raspberry Pi.

Recorded every `interval` seconds:
    cpu_percent      overall CPU usage (%)
    ram_used_mb      system RAM in use
    proc_rss_mb      RAM used by this Python process
    gpu_util / gpu_mem_mb   via nvidia-smi (Kaggle GPU only)
    cpu_temp_c       CPU temperature (Raspberry Pi / Linux thermal zone)

Usage
-----
    mon = SystemMonitor(interval=1.0)
    mon.start()
    ... work ...
    mon.stop()
    print(mon.summary())
"""
import os
import platform
import shutil
import subprocess
import threading
import time

import psutil


def read_cpu_temp():
    """CPU temperature in Celsius (works on Raspberry Pi); None if unavailable."""
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_gpu():
    """(utilization %, memory used MB) of GPU 0 via nvidia-smi; (None, None) if no GPU."""
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=2).stdout.strip()
        util, mem = out.split(",")
        return float(util), float(mem)
    except Exception:
        return None, None


def system_info() -> dict:
    """Static description of the machine (goes into the report)."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


class SystemMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid())

    def _sample(self):
        gpu_util, gpu_mem = read_gpu()
        return {
            "t": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": psutil.virtual_memory().used / 1e6,
            "proc_rss_mb": self._proc.memory_info().rss / 1e6,
            "gpu_util": gpu_util,
            "gpu_mem_mb": gpu_mem,
            "cpu_temp_c": read_cpu_temp(),
        }

    def _run(self):
        psutil.cpu_percent(interval=None)       # first call primes the counter
        while not self._stop.wait(self.interval):
            self.samples.append(self._sample())

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self.samples:                     # very short runs: take one sample
            self.samples.append(self._sample())
        return self

    def summary(self) -> dict:
        """Mean and max of every recorded quantity."""
        out = {}
        keys = [k for k in self.samples[0] if k != "t"] if self.samples else []
        for k in keys:
            vals = [s[k] for s in self.samples if s[k] is not None]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
                out[f"{k}_max"] = max(vals)
        return out
