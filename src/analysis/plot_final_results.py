"""
Plot Final Results - With Real Data
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Real data from experiments
CLOUD_EDGE_BO = {
    'MNIST': {'mean': 99.19, 'std': 0.13, 'best': 99.31},
    'FashionMNIST': {'mean': 88.08, 'std': 0.86, 'best': 89.09},
    'CIFAR-10': {'mean': 70.83, 'std': 3.14, 'best': 74.46}
}

BASELINE = {'MNIST': 98.50, 'FashionMNIST': 89.96, 'CIFAR-10': 65.00}

METHOD_RESULTS = {
    'MNIST': {'Random Search': 99.33, 'Grid Search': 99.47, 'Sequential BO': 99.23, 'Cloud-Edge BO': 99.19},
    'FashionMNIST': {'Random Search': 83.28, 'Grid Search': 90.04, 'Sequential BO': 89.24, 'Cloud-Edge BO': 88.08},
    'CIFAR-10': {'Random Search': 63.65, 'Grid Search': 77.57, 'Sequential BO': 66.82, 'Cloud-Edge BO': 70.83}
}

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = ['MNIST', 'FashionMNIST', 'CIFAR-10']
colors = {'Random Search': '#95A5A6', 'Grid Search': '#5D6D7E', 
          'Sequential BO': '#F39C12', 'Cloud-Edge BO': '#2E86AB'}

for idx, dataset in enumerate(datasets):
    methods = list(METHOD_RESULTS[dataset].keys())
    values = list(METHOD_RESULTS[dataset].values())
    
    axes[idx].bar(methods, values, color=[colors[m] for m in methods], 
                  edgecolor='black', linewidth=1.2)
    
    # Highlight Cloud-Edge BO
    for i, (method, val) in enumerate(zip(methods, values)):
        if method == 'Cloud-Edge BO':
            axes[idx].bar(method, val, color='#2E86AB', edgecolor='black', linewidth=2)
    
    axes[idx].axhline(y=BASELINE[dataset], color='#E74C3C', linestyle='--', 
                      linewidth=2, label=f'Baseline: {BASELINE[dataset]:.2f}%')
    
    axes[idx].set_xlabel('Method', fontsize=11)
    axes[idx].set_ylabel('Accuracy (%)', fontsize=11)
    axes[idx].set_title(f'{dataset}\nBest: {CLOUD_EDGE_BO[dataset]["best"]:.2f}%', fontsize=12)
    axes[idx].set_ylim(60, 102)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].tick_params(axis='x', rotation=15)
    
    for bar, val in zip(axes[idx].patches, values):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                      f'{val:.2f}%', ha='center', fontsize=8, fontweight='bold')

# Add legend for baseline
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=10)

plt.suptitle('Performance Comparison - Cloud-Edge Bayesian Optimization', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
print(" Graph saved to results/performance_comparison.png")
plt.show()