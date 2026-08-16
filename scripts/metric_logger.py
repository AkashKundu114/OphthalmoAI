import pynvml
import torch
import time
import psutil
import os
import json
from datetime import datetime

class HardwareTelemetry:
    def __init__(self, use_gpu=True, model_name="unknown"):
        self.start_time = None
        self.handle = None
        self.use_gpu = use_gpu
        self.process = psutil.Process(os.getpid())
        self.model_name = model_name
        self.gpu_name = "CPU Only"
        
        if self.use_gpu:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                # Decode bytes to string if needed
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode('utf-8')
                print(f"Hardware Telemetry Initialized on: {self.gpu_name}")
            except Exception as e:
                print(f"Failed to initialize pynvml: {e}")
                self.use_gpu = False
        else:
            print("Hardware Telemetry Initialized for CPU-Only execution.")
            
        # Initialize psutil cpu percent
        psutil.cpu_percent(interval=None)
        
        # Setup JSON logging
        # Save directly to the dataset folder so Docker passes it through to Windows
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"telemetry_{self.model_name}_{timestamp}.json")
        
        self.history = {
            "hardware": self.gpu_name,
            "model": self.model_name,
            "epochs": []
        }

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
        
        # CPU Heat
        cpu_temp = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
        except Exception:
            pass

        print(f"--- Epoch {epoch} Metrics ---")
        print(f"Time Total: {epoch_time:.2f} seconds")
        print(f"Loss: {loss:.4f} | Acc: {acc:.2f}%")
        
        print("\n[CPU & System RAM Metrics]")
        print(f"CPU Usage: {cpu_usage}% | CPU Heat: {cpu_temp if cpu_temp > 0 else 'N/A'}")
        print(f"RAM Usage (Process): {ram_usage:.2f} GB | System RAM Utilization: {sys_ram_percent}%")

        vram_utilization = 0.0
        peak_vram_gb = 0.0
        gpu_util = 0.0
        gpu_temp = 0.0

        if self.use_gpu and self.handle:
            # GPU VRAM usage
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
            peak_vram_gb = peak_vram / 1024
            vram_utilization = (peak_vram / 8192) * 100
            
            # GPU Temp & Utilization
            gpu_temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = utilization.gpu
            
            print("\n[GPU & VRAM Metrics]")
            print(f"GPU Usage: {gpu_util}% | GPU Heat: {gpu_temp}°C")
            print(f"Peak VRAM Usage: {peak_vram:.2f} MB | VRAM Utilization: {vram_utilization:.2f}% (Limit: 8192 MB)")
            
        print(f"---------------------------\n")
        
        # Save to JSON history
        epoch_data = {
            "epoch": epoch,
            "time_seconds": epoch_time,
            "loss": loss,
            "accuracy": acc,
            "sys_ram_gb_used": ram_usage,
            "sys_cpu_percent": cpu_usage,
            "cpu_temp_c": cpu_temp,
            "vram_gb_used": peak_vram_gb,
            "gpu_util_percent": gpu_util,
            "gpu_temp_c": gpu_temp
        }
        self.history["epochs"].append(epoch_data)
        
        # Write to disk at end of each epoch (in case of crash)
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)
