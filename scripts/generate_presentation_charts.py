import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

# Style settings for clean, publication-quality presentation figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

def load_all_runs(log_dir='./dataset/logs'):
    json_files = glob.glob(os.path.join(log_dir, 'telemetry_*.json'))
    runs = {}
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if 'epochs' in data and len(data['epochs']) > 0:
                    model_key = data.get('model', os.path.basename(file).replace('.json', '').replace('telemetry_', ''))
                    # Normalize keys
                    filename = os.path.basename(file)
                    if 'ConvNeXt' in filename and 'BF16' in filename:
                        model_key = 'ConvNeXt-Small (BF16)'
                    elif 'ConvNeXt' in filename and 'FP16' in filename:
                        model_key = 'ConvNeXt-Small (FP16)'
                    elif 'DenseNet' in filename and 'BF16' in filename:
                        model_key = 'DenseNet-201 (BF16)'
                    elif 'DenseNet' in filename and 'FP16' in filename:
                        model_key = 'DenseNet-201 (FP16)'
                    elif 'EfficientNet-V2-M' in filename and 'BF16' in filename:
                        model_key = 'EfficientNet-V2-M (BF16)'
                    elif 'EfficientNet-V2-M' in filename and 'FP16' in filename:
                        model_key = 'EfficientNet-V2-M (FP16)'
                    elif 'MetaClassifier' in filename and 'BF16' in filename:
                        model_key = 'Meta-Classifier (BF16, BS32)'
                    elif 'MetaClassifier' in filename and 'FP16' in filename:
                        model_key = 'Meta-Classifier (FP16, BS32)'
                    elif 'MetaEnsemble_2026' in filename:
                        model_key = 'Meta-Ensemble (FP16, BS4 Baseline)'
                    elif 'Docker-B4' in filename:
                        model_key = 'EfficientNet-B4 (Docker Single)'
                    elif 'CPU-ResNet50' in filename:
                        model_key = 'ResNet50 (CPU Baseline)'
                        
                    runs[model_key] = data
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return runs

def plot_base_monolith_models(runs, output_dir='docs/images'):
    target_models = [
        'ConvNeXt-Small (FP16)', 'ConvNeXt-Small (BF16)',
        'DenseNet-201 (FP16)', 'DenseNet-201 (BF16)',
        'EfficientNet-V2-M (FP16)', 'EfficientNet-V2-M (BF16)'
    ]
    
    valid_runs = {k: v for k, v in runs.items() if k in target_models}
    if not valid_runs:
        print("Base monolith logs not found!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
    
    names = list(valid_runs.keys())
    avg_times = [np.mean([e['time_seconds'] for e in valid_runs[k]['epochs']]) for k in names]
    peak_vrams = [max([e.get('vram_gb_used', 0) for e in valid_runs[k]['epochs']]) for k in names]
    colors = ['#2b5c8f', '#4a90e2', '#2e7d32', '#4caf50', '#c25e00', '#ff9800']
    
    # 1. Throughput / Time per epoch
    bars1 = ax1.bar(names, avg_times, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Avg Time per Epoch (Seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Training Throughput: 3 Base Monolith Models (BS=32)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Peak VRAM Footprint
    bars2 = ax2.bar(names, peak_vrams, color=colors, edgecolor='black', alpha=0.85)
    ax2.axhline(8.0, color='red', linestyle='--', linewidth=1.5, label='RTX 5060 Limit (8.0 GB)')
    ax2.set_ylabel('Peak VRAM (GB)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) GPU Memory Consumption (VRAM Limit: 8 GB)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax2.set_ylim(0, 9.0)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.2, f'{h:.2f} GB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Accuracy Convergence
    for i, k in enumerate(names):
        epochs = [e['epoch'] for e in valid_runs[k]['epochs']]
        accs = [e.get('accuracy', 0) for e in valid_runs[k]['epochs']]
        ax3.plot(epochs, accs, label=k, color=colors[i], linewidth=2.0, marker='o' if i%2==0 else 's', markersize=3)
    ax3.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Accuracy Convergence Across 40 Epochs', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. Loss Trajectory
    for i, k in enumerate(names):
        epochs = [e['epoch'] for e in valid_runs[k]['epochs']]
        losses = [e.get('loss', 0) for e in valid_runs[k]['epochs']]
        ax4.plot(epochs, losses, label=k, color=colors[i], linewidth=2.0, marker='^' if i%2==0 else 'd', markersize=3)
    ax4.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Loss Minimization Trajectories', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("OphthalmoAI: Base Monolith Vision Backbones Comparison\n(ConvNeXt-Small vs DenseNet-201 vs EfficientNet-V2-M | RTX 5060 GPU)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    path = os.path.join(output_dir, 'base_monolith_models_comparison.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Generated: {path}")

def plot_meta_classifier(runs, output_dir='docs/images'):
    target_meta = [
        'Meta-Ensemble (FP16, BS4 Baseline)',
        'Meta-Classifier (FP16, BS32)',
        'Meta-Classifier (BF16, BS32)'
    ]
    valid_runs = {k: v for k, v in runs.items() if k in target_meta}
    if not valid_runs:
        print("Meta classifier logs not found!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#7f8c8d', '#2980b9', '#27ae60']
    names = list(valid_runs.keys())
    
    # 1. Accuracy vs Time Speedup
    times = [np.mean([e['time_seconds'] for e in valid_runs[k]['epochs']]) for k in names]
    final_accs = [valid_runs[k]['epochs'][-1].get('accuracy', 0) for k in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1_twin = ax1.twinx()
    bars_t = ax1.bar(x - width/2, times, width, label='Avg Epoch Time (s)', color='#e74c3c', alpha=0.85, edgecolor='black')
    bars_a = ax1_twin.bar(x + width/2, final_accs, width, label='Final Accuracy (%)', color='#2ecc71', alpha=0.85, edgecolor='black')
    
    ax1.set_ylabel('Avg Time per Epoch (Seconds) [Lower is better]', color='#c0392b', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('Final Accuracy (%) [Higher is better]', color='#27ae60', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Meta-Classifier Scaling & Optimization (BS4 vs BS32)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax1_twin.set_ylim(90, 101)
    
    for bar in bars_t:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_a:
        h = bar.get_height()
        ax1_twin.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Convergence Curve of Meta-Classifier
    for i, k in enumerate(names):
        epochs = [e['epoch'] for e in valid_runs[k]['epochs']]
        accs = [e.get('accuracy', 0) for e in valid_runs[k]['epochs']]
        ax2.plot(epochs, accs, label=k, color=colors[i], linewidth=2.2, marker='o', markersize=3)
        
    ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ensemble Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Meta-Classifier Convergence (Reaching 99.72%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("OphthalmoAI: Meta-Classifier Ensemble Fusion Benchmark", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'meta_classifier_comparison.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Generated: {path}")

def plot_architecture_evolution(runs, output_dir='docs/images'):
    # Comparison stages
    stages = [
        ('CPU Baseline\n(ResNet50)', 460.79, 81.61, '#95a5a6'),
        ('GPU Monolith\n(ResNet50)', 52.09, 92.96, '#3498db'),
        ('GPU Monolith\n(EfficientNet-B4)', 33.68, 98.61, '#9b59b6'),
        ('GPU Monolith\n(ConvNeXt-Small)', 19.32, 99.32, '#e67e22'),
        ('GPU Monolith\n(DenseNet-201)', 24.74, 99.49, '#1abc9c'),
        ('Meta-Ensemble\n(Final Fusion [SOTA])', 20.62, 99.72, '#2ecc71')
    ]
    
    labels = [s[0] for s in stages]
    times = [s[1] for s in stages]
    accs = [s[2] for s in stages]
    colors = [s[3] for s in stages]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Speedup
    bars1 = ax1.bar(labels, times, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Time per Epoch (Seconds) [Log Scale]', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_title('Evolution of Training Speed (From CPU to Ensemble)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h * 1.1, f'{h:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Diagnostic Accuracy
    bars2 = ax2.bar(labels, accs, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Diagnostic Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(75, 102)
    ax2.set_title('Evolution of Diagnostic Screening Accuracy (Up to 99.72%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("OphthalmoAI: Full Architectural Evolution (Baseline vs Monoliths vs Meta-Ensemble)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_evolution_summary.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Generated: {path}")

if __name__ == '__main__':
    all_runs = load_all_runs()
    os.makedirs('docs/images', exist_ok=True)
    plot_base_monolith_models(all_runs)
    plot_meta_classifier(all_runs)
    plot_architecture_evolution(all_runs)
