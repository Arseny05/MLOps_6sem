import torch
import os
import time
import threading
import psutil


class Monitor:
    def __init__(self, device="cpu", sample_interval=0.1):
        self.device = device
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.start_time, self.end_time, self.start_cpu_time, self.end_cpu_time = None, None, None, None
        self._stop_event = threading.Event()
        self._thread = None
        self.ram_bytes = []
        self.gpu_bytes = []
    
    def sample_memory(self):
        while not self._stop_event.is_set():
            try:
                self.ram_bytes.append(self.process.memory_info().rss)
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    self.gpu_bytes.append(torch.cuda.memory_allocated(self.device))   
            except Exception:
                pass
            time.sleep(self.sample_interval)
    
    def start(self):
        self.ram_bytes = []
        self.gpu_bytes = []
        self.start_time = time.perf_counter()
        cpu_times = self.process.cpu_times()
        self.start_cpu_time = cpu_times.user + cpu_times.system
        self.ram_bytes.append(self.process.memory_info().rss)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_bytes.append(torch.cuda.memory_allocated(self.device))
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.sample_memory, daemon=True)
        self._thread.start()
    
    def stop(self):
        self.end_time = time.perf_counter()
        cpu_times = self.process.cpu_times()
        self.end_cpu_time = cpu_times.user + cpu_times.system
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.ram_bytes.append(self.process.memory_info().rss)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self.gpu_bytes.append(torch.cuda.memory_allocated(self.device))

    def summary(self):
        return {
            "train_time_sec": None if self.start_time is None or self.end_time is None else self.end_time - self.start_time,
            "cpu_time_sec": None if self.start_cpu_time is None or self.end_cpu_time is None else self.end_cpu_time - self.start_cpu_time,
            "peak_ram_mb": max(self.ram_bytes) / 1024**2 if self.ram_bytes else 0.0,
            "peak_gpu_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.gpu_bytes else 0.0,
            "ram_history": [x / 1024**2 for x in self.ram_bytes],
            "gpu_history": [x / 1024**2 for x in self.gpu_bytes]
        }