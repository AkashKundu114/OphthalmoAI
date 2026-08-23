import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

def load_telemetry_data(log_dir='./dataset/logs'):
    json_files = glob.glob(os.path.join(log_dir, 'telemetry_*.json'))
    data_runs = []
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if 'epochs' in data and len(data['epochs']) > 0:
                    data['filename'] = os.path.basename(file)
                    data_runs.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return data_runs

def create_benchmark_graphs(output_dir='docs/images'):
    os.makedirs(output_dir, exist_ok=True)
    
    runs = load_telemetry_data()
    if not runs:
        print("No telemetry_*.json files found in dataset/logs!")
        return

    print(f"\n=======================================================")
    print(f" Loaded {len(runs)} Benchmark Telemetry Logs")
    print(f"=======================================================")
    
    # Print Console Summary Table
    print(f"{'Model / Run Name':<38} | {'Avg Epoch Time':<15} | {'Peak VRAM':<12} | {'Final Acc':<10} | {'Max GPU Temp'}")
    print("-" * 95)
    
    environments = []
    time_per_epoch = []
    
    for run in runs:
        model_name = run.get('model', run['filename'].replace('.json', '').replace('telemetry_', ''))
        epochs = run['epochs']
        avg_time = np.mean([e['time_seconds'] for e in epochs])
        max_vram = max([e.get('vram_gb_used', 0) for e in epochs])
        final_acc = epochs[-1].get('accuracy', 0)
        max_temp = max([e.get('gpu_temp_c', 0) for e in epochs])
        
        environments.append(model_name)
        time_per_epoch.append(avg_time)
        
        print(f"{model_name:<38} | {avg_time:>10.2f} s    | {max_vram:>8.2f} GB   | {final_acc:>8.2f} % | {max_temp:>5.0f} °C")

    print("-" * 95)

    # 1. Training Speed Bar Chart
    fig, ax = plt.subplots(figsize=(max(12, len(runs) * 1.1), 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(environments)))
    bars = ax.bar(environments, time_per_epoch, color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Average Time per Epoch (Seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Speed Comparison: Average Epoch Time', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(0.5, height * 0.02),
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'training_time_comparison.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"\n📊 Generated: {chart1_path}")
    
    # 2. VRAM & RAM Memory Consumption
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    
    for run in runs:
        name = run.get('model', run['filename'].replace('.json', '').replace('telemetry_', ''))
        epochs = [e['epoch'] for e in run['epochs']]
        vram = [e.get('vram_gb_used', 0) for e in run['epochs']]
        ram = [e.get('sys_ram_gb_used', 0) for e in run['epochs']]
        
        if max(vram) > 0:
            ax1.plot(epochs, vram, marker='o', markersize=3, linestyle='-', label=name, linewidth=1.8)
        if max(ram) > 0:
            ax2.plot(epochs, ram, marker='s', markersize=3, linestyle='-', label=name, linewidth=1.8)
        
    ax1.set_ylabel('VRAM Usage (GB)', fontsize=11, fontweight='bold')
    ax1.set_title('GPU VRAM Consumption Across Training Epochs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax2.set_ylabel('System RAM (GB)', fontsize=11, fontweight='bold')
    ax2.set_title('System Memory (RAM) Utilization', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'memory_usage_comparison.png')
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"💾 Generated: {chart2_path}")

    # 3. Model Accuracy / Loss Convergence
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for run in runs:
        name = run.get('model', run['filename'].replace('.json', '').replace('telemetry_', ''))
        epochs = [e['epoch'] for e in run['epochs']]
        acc = [e.get('accuracy', 0) for e in run['epochs']]
        
        if max(acc) > 0:
            ax3.plot(epochs, acc, marker='^', markersize=3, linestyle='-', label=name, linewidth=1.8)
        
    ax3.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Model Convergence & Accuracy Progression', fontsize=13, fontweight='bold', pad=15)
    ax3.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.01, 0))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"📈 Generated: {chart3_path}")
    
    # 4. Thermal Performance
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    
    for run in runs:
        temps = [e.get('gpu_temp_c', 0) for e in run['epochs']]
        if any(t > 0 for t in temps):
            name = run.get('model', run['filename'].replace('.json', '').replace('telemetry_', ''))
            epochs = [e['epoch'] for e in run['epochs']]
            ax4.plot(epochs, temps, marker='d', markersize=3, linestyle='-', label=name, linewidth=1.8)
            
    ax4.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax4.set_ylabel('GPU Temperature (°C)', fontsize=11, fontweight='bold')
    ax4.set_title('GPU Core Thermal Profile During Training Runs', fontsize=13, fontweight='bold', pad=15)
    if ax4.get_legend_handles_labels()[1]:
        ax4.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, 'thermal_comparison.png')
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"🌡️ Generated: {chart4_path}")
    print("=======================================================\n")

if __name__ == '__main__':
    create_benchmark_graphs()
