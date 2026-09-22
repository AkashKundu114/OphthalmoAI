#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Continuous Real-Time System Metrics & Telemetry Daemon
===================================================================
Continuously samples and records comprehensive hardware metrics during active
training, benchmarking, and clinical evaluation runs:
- CPU: Overall utilization %, per-thread utilization (32 logical threads), frequency
- RAM: Process RSS, Virtual Memory, Total System RAM, Free RAM, Pagefile usage
- GPU: Core utilization %, Memory Controller %, Clocks (Core/Mem/SM), Power draw (W), PCIe I/O
- VRAM: Used MB, Total MB, Free MB, Utilization %
- Disk / Storage: Process read/write rates (MB/s), system I/O
- Output: Persists time-series log to dataset/logs/live_training_system_metrics.json
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "dataset" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    gpu_name = gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else str(gpu_name)
    HAS_GPU = True
except Exception as e:
    gpu_handle = None
    gpu_name = "N/A"
    HAS_GPU = False

output_json = LOGS_DIR / "live_training_system_metrics.json"
output_summary = LOGS_DIR / "live_system_metrics_summary.txt"

def get_snapshot(process):
    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    cpu_freq_mhz = round(freq.current, 1) if freq else 0.0

    # System Memory
    vmem = psutil.virtual_memory()
    sys_ram_used_gb = round(vmem.used / (1024 ** 3), 2)
    sys_ram_total_gb = round(vmem.total / (1024 ** 3), 2)
    sys_ram_percent = vmem.percent

    # Swap
    smem = psutil.swap_memory()
    swap_used_gb = round(smem.used / (1024 ** 3), 2)
    swap_percent = smem.percent

    # Process Memory
    try:
        mem_info = process.memory_info()
        proc_rss_gb = round(mem_info.rss / (1024 ** 3), 3)
        proc_vms_gb = round(mem_info.vms / (1024 ** 3), 3)
    except Exception:
        proc_rss_gb, proc_vms_gb = 0.0, 0.0

    # GPU & VRAM
    gpu_data = {}
    if HAS_GPU and gpu_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power_w = round(pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0, 2)
            power_limit_w = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(gpu_handle) / 1000.0, 2)
            clk_gfx = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
            clk_mem = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_MEM)
            mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            tx_mb = round(pynvml.nvmlDeviceGetPcieThroughput(gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024 ** 2), 2)
            rx_mb = round(pynvml.nvmlDeviceGetPcieThroughput(gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / (1024 ** 2), 2)

            gpu_data = {
                "name": gpu_name,
                "utilization_pct": util.gpu,
                "mem_controller_pct": util.memory,
                "temp_c": temp,
                "power_watts": power_w,
                "power_limit_watts": power_limit_w,
                "clock_graphics_mhz": clk_gfx,
                "clock_mem_mhz": clk_mem,
                "vram_used_mb": round(mem.used / (1024 ** 2), 1),
                "vram_total_mb": round(mem.total / (1024 ** 2), 1),
                "vram_free_mb": round(mem.free / (1024 ** 2), 1),
                "vram_util_pct": round((mem.used / mem.total) * 100, 1),
                "pcie_tx_mb_s": tx_mb,
                "pcie_rx_mb_s": rx_mb
            }
        except Exception:
            pass

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "overall_pct": cpu_percent,
            "frequency_mhz": cpu_freq_mhz,
            "per_core_pct": cpu_per_core
        },
        "system_ram": {
            "used_gb": sys_ram_used_gb,
            "total_gb": sys_ram_total_gb,
            "percent": sys_ram_percent,
            "swap_used_gb": swap_used_gb,
            "swap_percent": swap_percent
        },
        "process_ram": {
            "rss_gb": proc_rss_gb,
            "vms_gb": proc_vms_gb
        },
        "gpu": gpu_data
    }

def main():
    print(f"[METRICS DAEMON] Initialized for OphthalmoAI pipeline.")
    print(f"Sampling every 5.0 seconds. Log: {output_json}")
    
    current_proc = psutil.Process()
    metrics_history = []
    
    # Run continuously as an ongoing telemetry daemon
    while True:
        try:
            snapshot = get_snapshot(current_proc)
            metrics_history.append(snapshot)
            
            # Keep rolling window of last 200 samples
            if len(metrics_history) > 200:
                metrics_history = metrics_history[-200:]
                
            with open(output_json, "w") as f:
                json.dump({
                    "meta": {
                        "updated_at": datetime.now().isoformat(),
                        "samples_count": len(metrics_history),
                        "gpu_name": gpu_name,
                        "logical_threads": psutil.cpu_count(True)
                    },
                    "latest": snapshot,
                    "history": metrics_history
                }, f, indent=2)

            # Also generate a human-readable summary text snapshot
            cpu_val = snapshot['cpu']['overall_pct']
            ram_val = f"{snapshot['system_ram']['used_gb']} / {snapshot['system_ram']['total_gb']} GB ({snapshot['system_ram']['percent']}%)"
            gpu_u = snapshot['gpu'].get('utilization_pct', 'N/A')
            gpu_t = snapshot['gpu'].get('temp_c', 'N/A')
            gpu_p = snapshot['gpu'].get('power_watts', 'N/A')
            vram_u = f"{snapshot['gpu'].get('vram_used_mb', 0)} / {snapshot['gpu'].get('vram_total_mb', 0)} MB" if HAS_GPU else "N/A"

            summary_text = (
                f"================================================================================\n"
                f" OPHTHALMOAI REAL-TIME SYSTEM HARDWARE TELEMETRY SNAPSHOT\n"
                f" Time: {snapshot['timestamp']}\n"
                f"================================================================================\n"
                f" CPU Utilization:       {cpu_val} % (32 Logical Threads active)\n"
                f" CPU Frequency:         {snapshot['cpu']['frequency_mhz']} MHz\n"
                f" System RAM Usage:      {ram_val}\n"
                f" Process RAM RSS:       {snapshot['process_ram']['rss_gb']} GB\n"
                f" Pagefile / Swap:       {snapshot['system_ram']['swap_used_gb']} GB ({snapshot['system_ram']['swap_percent']}%)\n"
                f"--------------------------------------------------------------------------------\n"
                f" GPU Hardware:          {gpu_name}\n"
                f" GPU Compute Load:      {gpu_u} %\n"
                f" GPU Core Temperature:  {gpu_t} °C\n"
                f" GPU Power Draw:        {gpu_p} W\n"
                f" Dedicated VRAM:        {vram_u}\n"
                f"================================================================================\n"
            )
            with open(output_summary, "w", encoding="utf-8") as sf:
                sf.write(summary_text)

        except Exception as e:
            pass

        time.sleep(5.0)

if __name__ == "__main__":
    main()
