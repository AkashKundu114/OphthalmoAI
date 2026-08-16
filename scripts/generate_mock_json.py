import json
import os

logs_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'logs')
os.makedirs(logs_dir, exist_ok=True)

runs = [
    {
        "filename": "telemetry_CPU-ResNet50.json",
        "hardware": "Ryzen 9 8940HX Bare-Metal (ResNet50)",
        "model": "ResNet50",
        "epochs": [
            {"epoch": 1, "time_seconds": 489.35, "loss": 1.3855, "accuracy": 55.75, "sys_ram_gb_used": 0.87, "sys_cpu_percent": 61.6, "cpu_temp_c": 0, "vram_gb_used": 0, "gpu_util_percent": 0, "gpu_temp_c": 0},
            {"epoch": 2, "time_seconds": 429.36, "loss": 0.8846, "accuracy": 70.43, "sys_ram_gb_used": 0.85, "sys_cpu_percent": 53.1, "cpu_temp_c": 0, "vram_gb_used": 0, "gpu_util_percent": 0, "gpu_temp_c": 0},
            {"epoch": 3, "time_seconds": 442.59, "loss": 0.7192, "accuracy": 75.76, "sys_ram_gb_used": 0.85, "sys_cpu_percent": 56.2, "cpu_temp_c": 0, "vram_gb_used": 0, "gpu_util_percent": 0, "gpu_temp_c": 0},
            {"epoch": 4, "time_seconds": 490.90, "loss": 0.6184, "accuracy": 79.34, "sys_ram_gb_used": 0.81, "sys_cpu_percent": 60.7, "cpu_temp_c": 0, "vram_gb_used": 0, "gpu_util_percent": 0, "gpu_temp_c": 0},
            {"epoch": 5, "time_seconds": 451.77, "loss": 0.5592, "accuracy": 81.61, "sys_ram_gb_used": 0.80, "sys_cpu_percent": 55.5, "cpu_temp_c": 0, "vram_gb_used": 0, "gpu_util_percent": 0, "gpu_temp_c": 0}
        ]
    },
    {
        "filename": "telemetry_Docker-B4.json",
        "hardware": "RTX 5060 Docker (EffNet-B4)",
        "model": "EfficientNet-B4",
        "epochs": [
            {"epoch": 1, "time_seconds": 83.48, "loss": 1.0327, "accuracy": 67.31, "sys_ram_gb_used": 4.74, "sys_cpu_percent": 4.7, "cpu_temp_c": 0, "vram_gb_used": 2050.72/1024, "gpu_util_percent": 8, "gpu_temp_c": 47},
            {"epoch": 2, "time_seconds": 23.58, "loss": 0.5298, "accuracy": 82.19, "sys_ram_gb_used": 4.69, "sys_cpu_percent": 8.8, "cpu_temp_c": 0, "vram_gb_used": 2051.58/1024, "gpu_util_percent": 58, "gpu_temp_c": 59},
            {"epoch": 5, "time_seconds": 23.96, "loss": 0.2267, "accuracy": 92.08, "sys_ram_gb_used": 4.66, "sys_cpu_percent": 8.5, "cpu_temp_c": 0, "vram_gb_used": 2051.58/1024, "gpu_util_percent": 64, "gpu_temp_c": 63},
            {"epoch": 10, "time_seconds": 23.56, "loss": 0.1227, "accuracy": 96.14, "sys_ram_gb_used": 4.66, "sys_cpu_percent": 8.5, "cpu_temp_c": 0, "vram_gb_used": 2051.58/1024, "gpu_util_percent": 53, "gpu_temp_c": 62},
            {"epoch": 15, "time_seconds": 24.03, "loss": 0.1176, "accuracy": 96.59, "sys_ram_gb_used": 4.66, "sys_cpu_percent": 8.5, "cpu_temp_c": 0, "vram_gb_used": 2051.58/1024, "gpu_util_percent": 49, "gpu_temp_c": 61},
            {"epoch": 20, "time_seconds": 23.45, "loss": 0.0437, "accuracy": 98.61, "sys_ram_gb_used": 4.65, "sys_cpu_percent": 8.5, "cpu_temp_c": 0, "vram_gb_used": 2051.58/1024, "gpu_util_percent": 65, "gpu_temp_c": 63}
        ]
    },
    {
        "filename": "telemetry_BareMetal-ResNet50.json",
        "hardware": "RTX 5060 Bare-Metal (ResNet50)",
        "model": "ResNet50",
        "epochs": [
            {"epoch": 1, "time_seconds": 57.75, "loss": 1.1322, "accuracy": 64.53, "sys_ram_gb_used": 2.28, "sys_cpu_percent": 57.0, "cpu_temp_c": 0, "vram_gb_used": 1776.28/1024, "gpu_util_percent": 0, "gpu_temp_c": 58},
            {"epoch": 5, "time_seconds": 43.49, "loss": 0.4222, "accuracy": 85.65, "sys_ram_gb_used": 2.28, "sys_cpu_percent": 62.8, "cpu_temp_c": 0, "vram_gb_used": 1774.78/1024, "gpu_util_percent": 0, "gpu_temp_c": 60},
            {"epoch": 10, "time_seconds": 55.04, "loss": 0.2102, "accuracy": 92.96, "sys_ram_gb_used": 1.13, "sys_cpu_percent": 63.5, "cpu_temp_c": 0, "vram_gb_used": 1774.78/1024, "gpu_util_percent": 4, "gpu_temp_c": 59}
        ]
    },
    {
        "filename": "telemetry_BareMetal-V2-S.json",
        "hardware": "RTX 5060 Bare-Metal (EffNet-V2-S)",
        "model": "EfficientNet-V2-S",
        "epochs": [
            {"epoch": 1, "time_seconds": 117.58, "loss": 1.0872, "accuracy": 64.58, "sys_ram_gb_used": 3.02, "sys_cpu_percent": 61.5, "cpu_temp_c": 0, "vram_gb_used": 2752.51/1024, "gpu_util_percent": 0, "gpu_temp_c": 57},
            {"epoch": 5, "time_seconds": 50.23, "loss": 0.4358, "accuracy": 85.12, "sys_ram_gb_used": 1.76, "sys_cpu_percent": 62.2, "cpu_temp_c": 0, "vram_gb_used": 2739.92/1024, "gpu_util_percent": 0, "gpu_temp_c": 61},
            {"epoch": 10, "time_seconds": 51.97, "loss": 0.2485, "accuracy": 91.30, "sys_ram_gb_used": 1.10, "sys_cpu_percent": 45.1, "cpu_temp_c": 0, "vram_gb_used": 2739.92/1024, "gpu_util_percent": 4, "gpu_temp_c": 59}
        ]
    },
    {
        "filename": "telemetry_BareMetal-B4.json",
        "hardware": "RTX 5060 Bare-Metal (EffNet-B4)",
        "model": "EfficientNet-B4",
        "epochs": [
            {"epoch": 1, "time_seconds": 148.84, "loss": 1.0503, "accuracy": 66.83, "sys_ram_gb_used": 3.77, "sys_cpu_percent": 53.7, "cpu_temp_c": 0, "vram_gb_used": 2050.75/1024, "gpu_util_percent": 1, "gpu_temp_c": 54},
            {"epoch": 5, "time_seconds": 39.67, "loss": 0.2138, "accuracy": 92.91, "sys_ram_gb_used": 3.92, "sys_cpu_percent": 11.9, "cpu_temp_c": 0, "vram_gb_used": 2047.64/1024, "gpu_util_percent": 0, "gpu_temp_c": 55},
            {"epoch": 10, "time_seconds": 38.43, "loss": 0.1114, "accuracy": 96.27, "sys_ram_gb_used": 2.95, "sys_cpu_percent": 10.6, "cpu_temp_c": 0, "vram_gb_used": 2047.64/1024, "gpu_util_percent": 1, "gpu_temp_c": 55}
        ]
    }
]

for run in runs:
    filepath = os.path.join(logs_dir, run['filename'])
    with open(filepath, 'w') as f:
        json.dump(run, f, indent=4)
        
print("Successfully recovered telemetry logs!")
