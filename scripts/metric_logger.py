import pynvml
import torch
import time
import psutil
import os

class HardwareTelemetry:
    def __init__(self, use_gpu=True):
        self.start_time = None
        self.handle = None
        self.use_gpu = use_gpu
        self.process = psutil.Process(os.getpid())
        
        if self.use_gpu:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                print(f"Hardware Telemetry Initialized on: {self.gpu_name}")
            except Exception as e:
                print(f"Failed to initialize pynvml: {e}")
                self.use_gpu = False
        else:
            print("Hardware Telemetry Initialized for CPU-Only execution.")
            
        # Initialize psutil cpu percent
        psutil.cpu_percent(interval=None)

    def start_epoch(self):
        self.start_time = time.time()
        if self.use_gpu:
            torch.cuda.reset_peak_memory_stats()

    def end_epoch(self, epoch, loss, acc):
        epoch_time = time.time() - self.start_time
        
        # System RAM usage (GB)
        mem_info = self.process.memory_info()
        ram_usage = mem_info.rss / (1024 ** 3)
        sys_ram_percent = psutil.virtual_memory().percent
        
        # System CPU usage (%)
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # CPU Heat (Not always accessible on Windows without admin, default to N/A)
        cpu_temp = "N/A"
        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                cpu_temp = f"{temps['coretemp'][0].current:.1f}°C"
        except Exception:
            pass

        print(f"--- Epoch {epoch} Metrics ---")
        print(f"Time Total: {epoch_time:.2f} seconds")
        print(f"Loss: {loss:.4f} | Acc: {acc:.2f}%")
        
        print("\n[CPU & System RAM Metrics]")
        print(f"CPU Usage: {cpu_usage}% | CPU Heat: {cpu_temp}")
        print(f"RAM Usage (Process): {ram_usage:.2f} GB | System RAM Utilization: {sys_ram_percent}%")

        if self.use_gpu and self.handle:
            # GPU VRAM usage
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
            vram_utilization = (peak_vram / 8192) * 100
            
            # GPU Temp & Utilization
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = utilization.gpu
            
            print("\n[GPU & VRAM Metrics]")
            print(f"GPU Usage: {gpu_util}% | GPU Heat: {temp}°C")
            print(f"Peak VRAM Usage: {peak_vram:.2f} MB | VRAM Utilization: {vram_utilization:.2f}% (Limit: 8192 MB)")
            
        print(f"---------------------------\n")
