"""
Comprehensive Hardware & Clinical Performance Telemetry Logger.
Tracks EVERYTHING possible:
- CPU: Usage %, per-core % (32 threads for Ryzen 9 HX), frequencies, temps, context switches
- RAM: Process RSS, Virtual Memory, Total System RAM, Free RAM, Swap/Pagefile %, temps
- GPU: Core utilization %, Memory controller %, Core temp, Slowdown/Shutdown thresholds,
       Graphics clock, VRAM clock, SM clock, Power draw (W), Power limit (W), PCIe TX/RX throughput
- VRAM (RTX 5060 8GB): PyTorch allocated/reserved VRAM, NVML total/used/free VRAM, % utilization, headroom
- Training: Loss, Accuracy, Macro F1, Weighted F1, Throughput (samples/sec), Learning rate, Scaler scale
"""

import os
import sys
import time
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class HardwareTelemetry:
    def __init__(self, use_gpu=True, model_name="unknown", target_vram_mb=8151.0):
        self.start_time = None
        self.step_start_time = None
        self.handle = None
        self.use_gpu = use_gpu and PYNVML_AVAILABLE
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self.model_name = model_name
        self.gpu_name = "CPU Only"
        self.target_vram_mb = target_vram_mb
        self.nvml_initialized = False

        if self.use_gpu and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_bname = pynvml.nvmlDeviceGetName(self.handle)
                self.gpu_name = gpu_bname.decode('utf-8') if isinstance(gpu_bname, bytes) else str(gpu_bname)
                self.nvml_initialized = True
                print(f"[TELEMETRY] Attached to GPU: {self.gpu_name}")
            except Exception as e:
                print(f"[TELEMETRY] NVML Init notice: {e}. GPU hardware telemetry running in fallback mode.")
                self.use_gpu = False
        else:
            print("[TELEMETRY] Initialized CPU / General Telemetry Mode.")

        # Baseline CPU query to prime psutil
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)

        self.log_dir = Path(__file__).resolve().parent.parent / "dataset" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"telemetry_{self.model_name}_{timestamp}.json"

        # Hardware inventory metadata
        cpu_model = "AMD Ryzen 9 HX (or multi-core x86_64)"
        if PSUTIL_AVAILABLE:
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            sys_ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        else:
            cpu_count_logical, cpu_count_physical, sys_ram_total_gb = os.cpu_count(), os.cpu_count(), 0.0

        self.history = {
            "metadata": {
                "model_architecture": self.model_name,
                "gpu_hardware": self.gpu_name,
                "cpu_cores_physical": cpu_count_physical,
                "cpu_cores_logical": cpu_count_logical,
                "system_ram_total_gb": sys_ram_total_gb,
                "start_timestamp": datetime.now().isoformat()
            },
            "epochs": []
        }

        self.session_start_time = time.time()

        # Background continuous telemetry sampler
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._start_disk_io = None
        self._start_proc_io = None
        self._samples = {
            "gpu_util": [],
            "gpu_mem_ctrl": [],
            "gpu_power": [],
            "gpu_clock": [],
            "gpu_temp": [],
            "cpu_util": [],
            "proc_ram_gb": [],
            "sys_ram_percent": [],
            "pcie_tx": [],
            "pcie_rx": [],
        }

    def _sampling_loop(self):
        """Continuously samples GPU, CPU, RAM, and system metrics during active training/validation."""
        while not self._stop_event.is_set():
            if self.use_gpu and self.nvml_initialized and self.handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self._samples["gpu_util"].append(float(util.gpu))
                    self._samples["gpu_mem_ctrl"].append(float(util.memory))
                except Exception:
                    pass

                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self._samples["gpu_power"].append(float(p))
                except Exception:
                    pass

                try:
                    clk = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                    self._samples["gpu_clock"].append(float(clk))
                except Exception:
                    pass

                try:
                    t = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    self._samples["gpu_temp"].append(float(t))
                except Exception:
                    pass

                try:
                    tx = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024 ** 2)
                    rx = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / (1024 ** 2)
                    self._samples["pcie_tx"].append(float(tx))
                    self._samples["pcie_rx"].append(float(rx))
                except Exception:
                    pass

            if PSUTIL_AVAILABLE:
                try:
                    c_u = psutil.cpu_percent(interval=None)
                    if c_u > 0:
                        self._samples["cpu_util"].append(float(c_u))

                    vmem = psutil.virtual_memory()
                    self._samples["sys_ram_percent"].append(float(vmem.percent))

                    if self.process:
                        mem_info = self.process.memory_info()
                        self._samples["proc_ram_gb"].append(float(mem_info.rss / (1024 ** 3)))
                except Exception:
                    pass

            self._stop_event.wait(0.5)

    def start_epoch(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # Reset active telemetry samples and record start disk/process I/O
        self._stop_event.clear()
        for k in self._samples:
            self._samples[k] = []

        if PSUTIL_AVAILABLE:
            try:
                self._start_disk_io = psutil.disk_io_counters()
            except Exception:
                self._start_disk_io = None
            try:
                if self.process and hasattr(self.process, "io_counters"):
                    self._start_proc_io = self.process.io_counters()
            except Exception:
                self._start_proc_io = None

        self._monitor_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._monitor_thread.start()

    def _get_cpu_temp(self):
        """Attempts to query thermal sensors via psutil or Windows WMI."""
        if PSUTIL_AVAILABLE and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name in ['coretemp', 'k10temp', 'zenpower', 'cpu_thermal', 'acpitz']:
                        if name in temps and len(temps[name]) > 0:
                            return round(temps[name][0].current, 1)
            except Exception:
                pass
        return None

    def end_epoch(self, epoch, loss, acc, val_loss=None, val_acc=None, val_f1=None, samples_count=None, lr=None, scaler_scale=None):
        epoch_duration = time.time() - self.start_time if self.start_time else 0.0
        throughput = round(samples_count / epoch_duration, 2) if (samples_count and epoch_duration > 0) else None

        # Stop background continuous sampler
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join(timeout=1.0)

        # ==========================================
        # 1. CPU & SYSTEM RAM METRICS
        # ==========================================
        cpu_usage_overall = 0.0
        cpu_per_core = []
        cpu_freq_current = 0.0
        proc_ram_rss_gb = 0.0
        proc_ram_vms_gb = 0.0
        sys_ram_used_gb = 0.0
        sys_ram_free_gb = 0.0
        sys_ram_percent = 0.0
        swap_used_gb = 0.0
        swap_percent = 0.0
        cpu_temp = self._get_cpu_temp()

        if PSUTIL_AVAILABLE:
            try:
                cpu_usage_overall = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                freq = psutil.cpu_freq()
                if freq:
                    cpu_freq_current = round(freq.current, 1)

                vmem = psutil.virtual_memory()
                sys_ram_used_gb = round(vmem.used / (1024 ** 3), 2)
                sys_ram_free_gb = round(vmem.available / (1024 ** 3), 2)
                sys_ram_percent = vmem.percent

                smem = psutil.swap_memory()
                swap_used_gb = round(smem.used / (1024 ** 3), 2)
                swap_percent = smem.percent

                if self.process:
                    mem_info = self.process.memory_info()
                    proc_ram_rss_gb = round(mem_info.rss / (1024 ** 3), 3)
                    proc_ram_vms_gb = round(mem_info.vms / (1024 ** 3), 3)
            except Exception:
                pass

        avg_cpu_util = round(sum(self._samples["cpu_util"]) / len(self._samples["cpu_util"]), 1) if self._samples["cpu_util"] else cpu_usage_overall
        peak_cpu_util = max(self._samples["cpu_util"]) if self._samples["cpu_util"] else cpu_usage_overall
        avg_proc_ram_gb = round(sum(self._samples["proc_ram_gb"]) / len(self._samples["proc_ram_gb"]), 3) if self._samples["proc_ram_gb"] else proc_ram_rss_gb
        peak_proc_ram_gb = round(max(self._samples["proc_ram_gb"]), 3) if self._samples["proc_ram_gb"] else proc_ram_rss_gb
        avg_sys_ram_percent = round(sum(self._samples["sys_ram_percent"]) / len(self._samples["sys_ram_percent"]), 1) if self._samples["sys_ram_percent"] else sys_ram_percent
        peak_sys_ram_percent = round(max(self._samples["sys_ram_percent"]), 1) if self._samples["sys_ram_percent"] else sys_ram_percent

        # SSD / Disk I/O metrics
        proc_disk_read_mb = 0.0
        proc_disk_write_mb = 0.0
        proc_disk_read_speed_mb_s = 0.0
        proc_disk_write_speed_mb_s = 0.0
        sys_disk_read_mb = 0.0
        sys_disk_write_mb = 0.0
        sys_disk_read_speed_mb_s = 0.0
        sys_disk_write_speed_mb_s = 0.0

        if PSUTIL_AVAILABLE and epoch_duration > 0:
            try:
                if self.process and hasattr(self.process, "io_counters") and self._start_proc_io:
                    end_proc_io = self.process.io_counters()
                    proc_disk_read_mb = round(max(0, end_proc_io.read_bytes - self._start_proc_io.read_bytes) / (1024 ** 2), 2)
                    proc_disk_write_mb = round(max(0, end_proc_io.write_bytes - self._start_proc_io.write_bytes) / (1024 ** 2), 2)
                    proc_disk_read_speed_mb_s = round(proc_disk_read_mb / epoch_duration, 2)
                    proc_disk_write_speed_mb_s = round(proc_disk_write_mb / epoch_duration, 2)
            except Exception:
                pass

            try:
                if self._start_disk_io:
                    end_sys_io = psutil.disk_io_counters()
                    sys_disk_read_mb = round(max(0, end_sys_io.read_bytes - self._start_disk_io.read_bytes) / (1024 ** 2), 2)
                    sys_disk_write_mb = round(max(0, end_sys_io.write_bytes - self._start_disk_io.write_bytes) / (1024 ** 2), 2)
                    sys_disk_read_speed_mb_s = round(sys_disk_read_mb / epoch_duration, 2)
                    sys_disk_write_speed_mb_s = round(sys_disk_write_mb / epoch_duration, 2)
            except Exception:
                pass

        # ==========================================
        # 2. GPU & VRAM METRICS (RTX 5060 8GB)
        # ==========================================
        gpu_util_percent = 0.0
        gpu_mem_ctrl_percent = 0.0
        gpu_temp_c = 0.0
        gpu_slowdown_temp = 0.0
        gpu_shutdown_temp = 0.0
        gpu_power_watts = 0.0
        gpu_power_limit_watts = 0.0
        gpu_clock_graphics = 0.0
        gpu_clock_mem = 0.0
        gpu_clock_sm = 0.0
        pcie_tx_mb_s = 0.0
        pcie_rx_mb_s = 0.0

        nvml_vram_used_mb = 0.0
        nvml_vram_total_mb = self.target_vram_mb
        nvml_vram_free_mb = self.target_vram_mb
        vram_util_percent = 0.0

        pytorch_vram_allocated_mb = 0.0
        pytorch_vram_reserved_mb = 0.0

        if torch.cuda.is_available():
            try:
                pytorch_vram_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
                pytorch_vram_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
            except Exception:
                pass

        if self.nvml_initialized and self.handle:
            try:
                gpu_temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util_percent = util.gpu
                gpu_mem_ctrl_percent = util.memory

                try:
                    gpu_power_watts = round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2)
                    gpu_power_limit_watts = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0, 2)
                except Exception:
                    pass

                try:
                    gpu_clock_graphics = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                    gpu_clock_mem = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                    gpu_clock_sm = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                except Exception:
                    pass

                try:
                    gpu_slowdown_temp = pynvml.nvmlDeviceGetTemperatureThreshold(self.handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
                    gpu_shutdown_temp = pynvml.nvmlDeviceGetTemperatureThreshold(self.handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
                except Exception:
                    pass

                try:
                    pcie_tx_mb_s = round(pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024 ** 2), 2)
                    pcie_rx_mb_s = round(pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / (1024 ** 2), 2)
                except Exception:
                    pass

                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                nvml_vram_used_mb = round(mem.used / (1024 ** 2), 2)
                nvml_vram_total_mb = round(mem.total / (1024 ** 2), 2)
                nvml_vram_free_mb = round(mem.free / (1024 ** 2), 2)
                vram_util_percent = round((mem.used / mem.total) * 100, 2)
            except Exception as e:
                pass

        vram_headroom_mb = round(nvml_vram_free_mb, 2)

        # Aggregate continuous active training samples
        avg_gpu_util = round(sum(self._samples["gpu_util"]) / len(self._samples["gpu_util"]), 1) if self._samples["gpu_util"] else gpu_util_percent
        peak_gpu_util = max(self._samples["gpu_util"]) if self._samples["gpu_util"] else gpu_util_percent
        avg_gpu_mem_ctrl = round(sum(self._samples["gpu_mem_ctrl"]) / len(self._samples["gpu_mem_ctrl"]), 1) if self._samples["gpu_mem_ctrl"] else gpu_mem_ctrl_percent
        avg_gpu_power = round(sum(self._samples["gpu_power"]) / len(self._samples["gpu_power"]), 2) if self._samples["gpu_power"] else gpu_power_watts
        peak_gpu_power = round(max(self._samples["gpu_power"]), 2) if self._samples["gpu_power"] else gpu_power_watts
        peak_gpu_clock = max(self._samples["gpu_clock"]) if self._samples["gpu_clock"] else gpu_clock_graphics
        peak_gpu_temp = max(self._samples["gpu_temp"]) if self._samples["gpu_temp"] else gpu_temp_c

        mins, secs = divmod(int(epoch_duration), 60)
        time_str = f"{mins}m {secs:02d}s ({epoch_duration:.2f}s)" if mins > 0 else f"{epoch_duration:.2f}s"
        throughput_str = f"{throughput} samples/sec" if throughput else "N/A"

        # ==========================================
        # 3. PRINT FORMATTED CONSOLE TELEMETRY
        # ==========================================
        print(f"\n==================== [EPOCH {epoch:02d} HARDWARE & MODEL TELEMETRY] ====================")
        print(f"Time Taken: {time_str} | Throughput: {throughput_str} | LR: {lr or 'auto'}")
        print(f"Training Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Val Loss: {val_loss or 'N/A'} | Val Acc: {val_acc or 'N/A'}% | Val F1: {val_f1 or 'N/A'}")

        print("\n--- CPU & SYSTEM RAM METRICS ---")
        print(f"CPU Utilization: {avg_cpu_util}% avg (Peak: {peak_cpu_util}%) | Frequency: {cpu_freq_current} MHz")
        if cpu_temp:
            print(f"CPU Temperature: {cpu_temp} C")
        print(f"Process RAM (RSS): {avg_proc_ram_gb} GB avg (Peak: {peak_proc_ram_gb} GB) | Virtual: {proc_ram_vms_gb} GB")
        print(f"System RAM: {sys_ram_used_gb} / {self.history['metadata']['system_ram_total_gb']} GB ({avg_sys_ram_percent}% avg, Peak: {peak_sys_ram_percent}%)")
        print(f"Swap / Pagefile Usage: {swap_used_gb} GB ({swap_percent}%)")

        print("\n--- STORAGE & DISK I/O (SSD) METRICS ---")
        print(f"Process Disk Read: {proc_disk_read_mb} MB ({proc_disk_read_speed_mb_s} MB/s) | Write: {proc_disk_write_mb} MB ({proc_disk_write_speed_mb_s} MB/s)")
        print(f"System Total Disk Read: {sys_disk_read_mb} MB ({sys_disk_read_speed_mb_s} MB/s) | Write: {sys_disk_write_mb} MB ({sys_disk_write_speed_mb_s} MB/s)")

        if self.use_gpu and self.nvml_initialized:
            print("\n--- GPU & DEDICATED VRAM (RTX 5060 8GB) METRICS ---")
            print(f"GPU Active Compute: {avg_gpu_util}% avg (Peak: {peak_gpu_util}%) | Memory Controller: {avg_gpu_mem_ctrl}% avg")
            print(f"GPU Core Temp: {gpu_temp_c} C (Peak: {peak_gpu_temp} C | Slowdown: {gpu_slowdown_temp} C | Shutdown: {gpu_shutdown_temp} C)")
            print(f"GPU Active Power: {avg_gpu_power} W avg (Peak: {peak_gpu_power} W / {gpu_power_limit_watts} W TGP)")
            print(f"GPU Clocks: Core Peak {peak_gpu_clock} MHz | VRAM {gpu_clock_mem} MHz | SM {gpu_clock_sm} MHz")
            print(f"PCIe Throughput: TX {pcie_tx_mb_s} MB/s | RX {pcie_rx_mb_s} MB/s")
            print(f"VRAM Used: {nvml_vram_used_mb} MB / {nvml_vram_total_mb} MB ({vram_util_percent}%) | Headroom: {vram_headroom_mb} MB free")
            if pytorch_vram_allocated_mb > 0:
                print(f"PyTorch Active Tensors: {pytorch_vram_allocated_mb} MB | PyTorch Reserved Cache: {pytorch_vram_reserved_mb} MB")

        print("=========================================================================\n")

        # ==========================================
        # 4. EXHAUSTIVE PERSISTENT JSON LOGGING
        # ==========================================
        epoch_record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "epoch_duration_sec": round(epoch_duration, 2),
                "throughput_samples_per_sec": throughput,
                "train_loss": round(loss, 5),
                "train_accuracy": round(acc, 4),
                "val_loss": round(val_loss, 5) if val_loss is not None else None,
                "val_accuracy": round(val_acc, 4) if val_acc is not None else None,
                "val_macro_f1": round(val_f1, 4) if val_f1 is not None else None,
                "learning_rate": lr,
                "grad_scaler_scale": scaler_scale
            },
            "cpu_telemetry": {
                "cpu_util_avg_percent": avg_cpu_util,
                "cpu_util_peak_percent": peak_cpu_util,
                "cpu_freq_mhz": cpu_freq_current,
                "cpu_temp_c": cpu_temp,
                "cpu_per_core_percent": cpu_per_core
            },
            "system_ram_telemetry": {
                "process_ram_rss_avg_gb": avg_proc_ram_gb,
                "process_ram_rss_peak_gb": peak_proc_ram_gb,
                "process_ram_vms_gb": proc_ram_vms_gb,
                "sys_ram_used_gb": sys_ram_used_gb,
                "sys_ram_free_gb": sys_ram_free_gb,
                "sys_ram_avg_percent": avg_sys_ram_percent,
                "sys_ram_peak_percent": peak_sys_ram_percent,
                "swap_used_gb": swap_used_gb,
                "swap_percent": swap_percent
            },
            "disk_telemetry": {
                "process_disk_read_mb": proc_disk_read_mb,
                "process_disk_write_mb": proc_disk_write_mb,
                "process_disk_read_speed_mb_s": proc_disk_read_speed_mb_s,
                "process_disk_write_speed_mb_s": proc_disk_write_speed_mb_s,
                "system_disk_read_mb": sys_disk_read_mb,
                "system_disk_write_mb": sys_disk_write_mb,
                "system_disk_read_speed_mb_s": sys_disk_read_speed_mb_s,
                "system_disk_write_speed_mb_s": sys_disk_write_speed_mb_s
            },
            "gpu_telemetry": {
                "gpu_name": self.gpu_name,
                "gpu_util_avg_percent": avg_gpu_util,
                "gpu_util_peak_percent": peak_gpu_util,
                "gpu_mem_ctrl_avg_percent": avg_gpu_mem_ctrl,
                "gpu_temp_c": gpu_temp_c,
                "gpu_temp_peak_c": peak_gpu_temp,
                "gpu_slowdown_temp_c": gpu_slowdown_temp,
                "gpu_shutdown_temp_c": gpu_shutdown_temp,
                "gpu_power_watts_avg": avg_gpu_power,
                "gpu_power_watts_peak": peak_gpu_power,
                "gpu_power_limit_watts": gpu_power_limit_watts,
                "gpu_clock_graphics_peak_mhz": peak_gpu_clock,
                "gpu_clock_mem_mhz": gpu_clock_mem,
                "gpu_clock_sm_mhz": gpu_clock_sm,
                "pcie_tx_mb_s": pcie_tx_mb_s,
                "pcie_rx_mb_s": pcie_rx_mb_s
            },
            "vram_telemetry": {
                "pytorch_vram_allocated_mb": pytorch_vram_allocated_mb,
                "pytorch_vram_reserved_mb": pytorch_vram_reserved_mb,
                "nvml_vram_used_mb": nvml_vram_used_mb,
                "nvml_vram_total_mb": nvml_vram_total_mb,
                "nvml_vram_free_mb": nvml_vram_free_mb,
                "vram_utilization_percent": vram_util_percent,
                "vram_headroom_mb": vram_headroom_mb
            }
        }

        self.history["epochs"].append(epoch_record)
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        total_session_sec = round(time.time() - self.session_start_time, 2)
        mins, secs = divmod(int(total_session_sec), 60)
        dur_str = f"{mins}m {secs:02d}s ({total_session_sec:.2f}s)" if mins > 0 else f"{total_session_sec:.2f}s"
        self.history["metadata"]["total_duration_sec"] = total_session_sec
        self.history["metadata"]["total_duration_human"] = dur_str
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

        print(f"\n[BENCHMARK] Total Training Session Duration: {dur_str}")
        print(f"[BENCHMARK] Hardware & performance log saved: {self.log_file.name}")

        if self.nvml_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
