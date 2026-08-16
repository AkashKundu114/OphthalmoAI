import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

def load_telemetry_data(log_dir='./dataset/logs'):
    """Finds and loads all telemetry JSON files in the specified directory."""
    json_files = glob.glob(os.path.join(log_dir, 'telemetry_*.json'))
    data_runs = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data['filename'] = os.path.basename(file)
            data_runs.append(data)
    return data_runs

def create_benchmark_graphs(output_dir='docs/images'):
    os.makedirs(output_dir, exist_ok=True)
    
    runs = load_telemetry_data()
    if not runs:
        print("No telemetry_*.json files found! Run your training scripts first.")
        return

    print(f"Found {len(runs)} telemetry logs. Generating graphs...")

    # 1. Average Time Per Epoch Bar Chart
    environments = []
    time_per_epoch = []
    for run in runs:
        # Create a short name based on hardware or filename
        name = run.get('hardware', run['filename'].replace('.json', ''))
        # Calculate average time across all epochs
        avg_time = np.mean([e['time_seconds'] for e in run['epochs']])
        environments.append(name)
        time_per_epoch.append(avg_time)
        
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(environments, time_per_epoch, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(environments)])
    
    ax.set_ylabel('Avg Time per Epoch (Seconds)', fontsize=12)
    ax.set_title('Training Speed Comparison: Average Epoch Time', fontsize=14, pad=20)
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300)
    print("Generated training_time_comparison.png")
    
    # 2. System Resource Usage (VRAM & RAM) over Epochs
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for run in runs:
        name = run.get('hardware', run['filename'])
        epochs = [e['epoch'] for e in run['epochs']]
        vram = [e.get('vram_gb_used', 0) for e in run['epochs']]
        ram = [e.get('sys_ram_gb_used', 0) for e in run['epochs']]
        
        ax1.plot(epochs, vram, marker='o', linestyle='-', label=name, linewidth=2)
        ax2.plot(epochs, ram, marker='s', linestyle='-', label=name, linewidth=2)
        
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax1.set_title('GPU VRAM Consumption over Time', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('System RAM (GB)', fontsize=12)
    ax2.set_title('System Memory Consumption over Time', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
    print("Generated memory_usage_comparison.png")

    # 3. Accuracy Convergence
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for run in runs:
        name = run.get('hardware', run['filename'])
        epochs = [e['epoch'] for e in run['epochs']]
        acc = [e.get('accuracy', 0) for e in run['epochs']]
        
        ax3.plot(epochs, acc, marker='^', linestyle='-', label=name, linewidth=2)
        
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.set_title('Model Convergence Comparison', fontsize=14, pad=20)
    ax3.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), dpi=300)
    print("Generated convergence_comparison.png")
    
    # 4. Thermal Analysis
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for run in runs:
        # Only plot GPU temps if they exist and aren't 0
        temps = [e.get('gpu_temp_c', 0) for e in run['epochs']]
        if any(t > 0 for t in temps):
            name = run.get('hardware', run['filename'])
            epochs = [e['epoch'] for e in run['epochs']]
            ax4.plot(epochs, temps, marker='d', linestyle='-', label=name, linewidth=2)
            
    ax4.set_xlabel('Epochs', fontsize=12)
    ax4.set_ylabel('GPU Temperature (°C)', fontsize=12)
    ax4.set_title('Thermal Performance During Training', fontsize=14, pad=20)
    if ax4.get_legend_handles_labels()[1]: # only show legend if we plotted something
        ax4.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thermal_comparison.png'), dpi=300)
    print("Generated thermal_comparison.png")

if __name__ == '__main__':
    create_benchmark_graphs()
