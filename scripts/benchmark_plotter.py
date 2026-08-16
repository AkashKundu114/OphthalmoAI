import matplotlib.pyplot as plt
import numpy as np
import os

def create_benchmark_graphs(output_dir='docs/images'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Data for Training Time Comparison
    environments = ['CPU Only', 'Docker (GPU)', 'Bare-Metal GPU (RTX 5060)']
    # Placeholders for seconds per epoch
    time_per_epoch = [320.5, 42.1, 38.4] 
    
    # 1. Bar Chart: Time Per Epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(environments, time_per_epoch, color=['#e74c3c', '#f39c12', '#2ecc71'])
    
    ax.set_ylabel('Time per Epoch (Seconds)', fontsize=12)
    ax.set_title('Training Speed Comparison: CPU vs GPU vs Docker', fontsize=14, pad=20)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300)
    print("Generated training_time_comparison.png")
    
    # 2. Line Chart: Loss convergence speed
    epochs = np.arange(1, 11)
    # Simulated loss values
    cpu_loss = np.exp(-epochs * 0.1) + 0.5 
    gpu_loss = np.exp(-epochs * 0.3) + 0.2
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs, cpu_loss, marker='o', linestyle='-', color='#e74c3c', label='CPU Only', linewidth=2)
    ax2.plot(epochs, gpu_loss, marker='s', linestyle='-', color='#2ecc71', label='Bare-Metal GPU', linewidth=2)
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Convergence: CPU vs GPU Acceleration', fontsize=14, pad=20)
    ax2.set_xticks(epochs)
    ax2.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), dpi=300)
    print("Generated convergence_comparison.png")

if __name__ == '__main__':
    create_benchmark_graphs()
