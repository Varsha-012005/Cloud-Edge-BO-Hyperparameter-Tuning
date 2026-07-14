import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Patch

os.makedirs('paper/figures', exist_ok=True)
os.makedirs('paper/figures/mnist', exist_ok=True)
os.makedirs('paper/figures/fashionmnist', exist_ok=True)
os.makedirs('paper/figures/cifar10', exist_ok=True)
os.makedirs('paper/tables', exist_ok=True)
os.makedirs('report_graphs', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (8, 5.5),
    'axes.linewidth': 1.2,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

CLOUD_EDGE_BO = {
    'MNIST': {'mean': 99.19, 'std': 0.13, 'best': 99.31, 'seeds': [98.98, 99.25, 99.25, 99.31, 99.15]},
    'FashionMNIST': {'mean': 88.08, 'std': 0.86, 'best': 89.09, 'seeds': [87.08, 87.93, 88.86, 89.09, 87.42]},
    'CIFAR-10': {'mean': 70.83, 'std': 3.14, 'best': 74.46, 'seeds': [70.15, 73.80, 74.46, 69.49, 66.26]}
}

BASELINE = {'MNIST': 99.47, 'FashionMNIST': 90.04, 'CIFAR-10': 77.57}

METHOD_RESULTS = {
    'MNIST': [99.33, 99.47, 99.23, 99.19],
    'FashionMNIST': [83.28, 90.04, 89.24, 88.08],
    'CIFAR-10': [63.65, 77.57, 66.82, 70.83]
}

METHOD_NAMES = ['Random Search', 'Grid Search', 'Sequential BO', 'Proposed Method']

CONVERGENCE = {
    'mnist': {
        'best': [98.49, 98.62, 98.74, 98.78, 98.84, 98.88, 98.97, 98.98, 99.25, 99.31],
        'baseline': 99.47,
        'best_acc': 99.31,
        'name': 'MNIST',
        'color': '#3498DB',
        'marker': 'o',
        'folder': 'mnist'
    },
    'fashionmnist': {
        'best': [83.15, 83.65, 86.38, 86.47, 86.64, 87.08, 87.93, 88.86, 89.09, 89.09],
        'baseline': 90.04,
        'best_acc': 89.09,
        'name': 'FashionMNIST',
        'color': '#2ECC71',
        'marker': 's',
        'folder': 'fashionmnist'
    },
    'cifar10': {
        'best': [64.50, 65.27, 65.43, 67.76, 68.02, 70.15, 73.80, 74.46, 74.46, 74.46],
        'baseline': 77.57,
        'best_acc': 74.46,
        'name': 'CIFAR-10',
        'color': '#E74C3C',
        'marker': '^',
        'folder': 'cifar10'
    }
}

LR_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LR_DATA = {
    'mnist': [95.35, 97.84, 98.49, 98.84, 98.78, 97.95, 11.35],
    'fashionmnist': [74.34, 78.99, 83.15, 86.38, 87.08, 81.77, 77.06],
    'cifar10': [46.87, 63.69, 64.50, 67.76, 68.02, 40.63, 10.0]
}

BATCH_SIZES = [16, 32, 64, 128, 256]
BATCH_DATA = {
    'mnist': [98.98, 99.15, 98.88, 99.25, 98.50],
    'fashionmnist': [87.08, 83.69, 89.09, 87.93, 86.93],
    'cifar10': [70.15, 64.50, 68.69, 74.46, 65.27]
}

LIT_METHODS = ['LeNet-5\n(1998)', 'PyTorch\n(2020)', 'Keras\n(2021)', 'ResNet-50\n(2016)', 'PROPOSED\n(2026)']
LIT_ACCURACIES = [99.05, 99.47, 98.40, 99.10, 99.31]
LIT_COLORS = ['#95A5A6', '#5D6D7E', '#85929E', '#717D7E', '#27AE60']

WORKERS = [1, 2, 3, 4, 5, 6]
SPEEDUP = [1.0, 1.9, 2.8, 3.4, 3.9, 4.2]

datasets = ['MNIST', 'FashionMNIST', 'CIFAR-10']
dataset_keys = ['mnist', 'fashionmnist', 'cifar10']

def save_figure(fig, path_png, path_pdf):
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')

print("=" * 70)
print("GENERATING ALL PAPER FIGURES (PNG + PDF)")
print("=" * 70)

print("\n[1/20] Figure 1: Final Results Bar Chart")
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, [99.47, 90.04, 77.57], width, label='Grid Search (Best)', 
               color='#95A5A6', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, [99.19, 88.08, 70.83], width, label='Proposed Method (Mean)', 
               color='#2E86C1', edgecolor='black', linewidth=1.5)

ax.errorbar(x + width/2, [99.19, 88.08, 70.83], yerr=[0.13, 0.86, 3.14], 
            fmt='none', ecolor='black', capsize=5, linewidth=1.5)

for i, (xi, best) in enumerate(zip(x + width/2, [99.31, 89.09, 74.46])):
    ax.scatter(xi, best, color='#27AE60', s=150, marker='*', 
               edgecolor='black', linewidth=0.5, zorder=10, label='Best' if i == 0 else '')

ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Figure 1: Final Results - Proposed Bayesian Optimization Method', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(55, 102)
ax.grid(True, alpha=0.25, axis='y')
ax.legend(loc='lower right', fontsize=10)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.8, 
                f'{height:.2f}%', ha='center', fontsize=9, fontweight='bold')

save_figure(fig, 'paper/figures/Figure1_Final_Results.png', 'paper/figures/Figure1_Final_Results.pdf')
save_figure(fig, 'report_graphs/Figure1_Final_Results.png', 'report_graphs/Figure1_Final_Results.pdf')
plt.close()

print("[2/20] Figure 2-4: Performance Comparison (Per Dataset)")
for i, dataset_key in enumerate(dataset_keys):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(METHOD_NAMES))
    results = METHOD_RESULTS[datasets[i]]
    bar_colors = ['#95A5A6', '#5D6D7E', '#F39C12', '#2E86AB']
    bars = ax.bar(x, results, 0.6, color=bar_colors, edgecolor='black', linewidth=1.2)
    bars[3].set_color('#2E86AB')
    bars[3].set_linewidth(2)
    
    for bar, val in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure {i+2}: {datasets[i]} Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=9, rotation=15, ha='right')
    
    if dataset_key == 'mnist':
        ax.set_ylim(95, 100.5)
    elif dataset_key == 'fashionmnist':
        ax.set_ylim(80, 92)
    else:
        ax.set_ylim(55, 80)
    
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    save_figure(fig, f'paper/figures/{dataset_key}/Figure{i+2}_{datasets[i]}_Performance.png', 
                f'paper/figures/{dataset_key}/Figure{i+2}_{datasets[i]}_Performance.pdf')
    plt.close()

print("[5/20] Figure 5: Improvement Over Baseline")
fig, ax = plt.subplots(figsize=(8, 5))
improvements = [99.19-99.47, 88.08-90.04, 70.83-77.57]
colors_imp = ['#3498DB' if x > 0 else '#E74C3C' for x in improvements]
bars = ax.bar(datasets, improvements, color=colors_imp, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement Over Grid Search Best (%)', fontsize=13, fontweight='bold')
ax.set_title('Figure 5: Accuracy Improvement Over Baseline', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.25, axis='y')

for bar, imp in zip(bars, improvements):
    color = 'green' if imp > 0 else 'red'
    sign = '+' if imp > 0 else ''
    y_pos = bar.get_height() + (0.8 if imp > 0 else -2.5)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{sign}{imp:.2f} pp', ha='center', fontsize=11, fontweight='bold', color=color)

save_figure(fig, 'paper/figures/Figure5_Improvement_Over_Baseline.png', 'paper/figures/Figure5_Improvement_Over_Baseline.pdf')
save_figure(fig, 'report_graphs/Figure5_Improvement_Over_Baseline.png', 'report_graphs/Figure5_Improvement_Over_Baseline.pdf')
plt.close()

print("[6/20] Figure 6-8: Convergence Curves")
fig_count = 6
for key, data in CONVERGENCE.items():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    trials = list(range(1, len(data['best']) + 1))
    
    ax.plot(trials, data['best'], color='#2C3E50', marker=data['marker'],
            markersize=9, markerfacecolor=data['color'], 
            markeredgecolor='white', markeredgewidth=1.5,
            linewidth=3, label='Best so far', zorder=4)
    
    ax.axhline(y=data['baseline'], color='#E74C3C', linestyle='--',
               linewidth=2.5, label=f'Grid Search Best ({data["baseline"]:.2f}%)', zorder=1)
    
    ax.scatter([len(data['best'])], [data['best_acc']], 
               color='#27AE60', s=200, zorder=5, marker='*', 
               edgecolor='white', linewidth=1,
               label=f'Proposed Method Best: {data["best_acc"]:.2f}%')
    
    ax.set_xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure {fig_count}: {data["name"]} Convergence', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, len(data['best']) + 0.5)
    ax.set_xticks(range(1, len(data['best']) + 1))
    
    if key == 'mnist':
        ax.set_ylim(97.5, 100)
    elif key == 'fashionmnist':
        ax.set_ylim(80, 92)
    else:
        ax.set_ylim(60, 78)
    
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    save_figure(fig, f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_Convergence.png',
                f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_Convergence.pdf')
    plt.close()
    fig_count += 1

print("[9/20] Figure 9-11: Learning Rate Sensitivity")
fig_count = 9
for key, data in CONVERGENCE.items():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lr_acc = LR_DATA[key]
    
    ax.semilogx(LR_VALUES, lr_acc, 'o-', color=data['color'], linewidth=3,
                markersize=10, markerfacecolor=data['color'], markeredgecolor='white',
                markeredgewidth=2, label='Validation Accuracy')
    opt_idx = np.argmax(lr_acc)
    ax.plot(LR_VALUES[opt_idx], lr_acc[opt_idx], 's', color='#27AE60',
            markersize=14, markerfacecolor='#27AE60', markeredgecolor='white',
            markeredgewidth=2, label=f'Optimal: {LR_VALUES[opt_idx]:.0e}')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure {fig_count}: {data["name"]} Learning Rate Sensitivity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=10)
    
    if key == 'mnist':
        ax.set_ylim(0, 100)
    elif key == 'fashionmnist':
        ax.set_ylim(70, 90)
    else:
        ax.set_ylim(0, 80)
    
    plt.tight_layout()
    save_figure(fig, f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_LR_Sensitivity.png',
                f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_LR_Sensitivity.pdf')
    plt.close()
    fig_count += 1

print("[12/20] Figure 12-14: Batch Size Sensitivity")
fig_count = 12
for key, data in CONVERGENCE.items():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bs_acc = BATCH_DATA[key]
    
    ax.plot(BATCH_SIZES, bs_acc, 's-', color=data['color'], linewidth=3,
            markersize=10, markerfacecolor=data['color'], markeredgecolor='white',
            markeredgewidth=2, label='Validation Accuracy')
    opt_idx = np.argmax(bs_acc)
    ax.plot(BATCH_SIZES[opt_idx], bs_acc[opt_idx], 'o', color='#27AE60',
            markersize=14, markerfacecolor='#27AE60', markeredgecolor='white',
            markeredgewidth=2, label=f'Optimal: {BATCH_SIZES[opt_idx]}')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure {fig_count}: {data["name"]} Batch Size Effect', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(10, 270)
    
    if key == 'mnist':
        ax.set_ylim(98, 100)
    elif key == 'fashionmnist':
        ax.set_ylim(80, 91)
    else:
        ax.set_ylim(60, 76)
    
    plt.tight_layout()
    save_figure(fig, f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_Batch_Size.png',
                f'paper/figures/{data["folder"]}/Figure{fig_count}_{data["name"]}_Batch_Size.pdf')
    plt.close()
    fig_count += 1

print("[15/20] Figure 15: Speedup Analysis")
fig, ax = plt.subplots(figsize=(8, 5.5))
seq_time = 36
ce_time = 12
methods = ['Sequential BO\n(1 worker)', 'Proposed Method\n(3 workers)']
times = [seq_time, ce_time]
bars = ax.bar(methods, times, color=['#E74C3C', '#27AE60'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Time to Complete (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Figure 15: Speedup Analysis (3× Speedup)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.25, axis='y')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{t} min', ha='center', fontsize=10, fontweight='bold')
ax.text(0.5, 25, '3× Speedup', ha='center', fontsize=12, fontweight='bold', color='#27AE60')
save_figure(fig, 'paper/figures/Figure15_Speedup_Analysis.png', 'paper/figures/Figure15_Speedup_Analysis.pdf')
save_figure(fig, 'report_graphs/Figure15_Speedup_Analysis.png', 'report_graphs/Figure15_Speedup_Analysis.pdf')
plt.close()

print("[16/20] Figure 16: Worker Scalability")
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(WORKERS, SPEEDUP, 'o-', color='#2E86AB', linewidth=3, markersize=10,
        markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
ax.axvline(x=3, color='#E74C3C', linestyle='--', linewidth=2.5, label='Optimal (3 workers)')
ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax.set_title('Figure 16: Worker Scalability', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0, 5.5)
ax.grid(True, alpha=0.25)
ax.legend(loc='best', fontsize=10)
for x, y in zip(WORKERS, SPEEDUP):
    ax.text(x, y + 0.2, f'{y:.1f}×', ha='center', fontsize=9, fontweight='bold')
save_figure(fig, 'paper/figures/Figure16_Worker_Scalability.png', 'paper/figures/Figure16_Worker_Scalability.pdf')
save_figure(fig, 'report_graphs/Figure16_Worker_Scalability.png', 'report_graphs/Figure16_Worker_Scalability.pdf')
plt.close()

print("[17/20] Figure 17: Heterogeneity Compensation")
fig, ax = plt.subplots(figsize=(8, 5.5))
workers = ['Slow\n(t3.micro)', 'Medium\n(t3.small)', 'Fast\n(t3.medium)']
tasks = [45, 35, 20]
colors = ['#F39C12', '#3498DB', '#2ECC71']
bars = ax.bar(workers, tasks, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Task Distribution (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 17: Heterogeneity Compensation Effect', fontsize=14, fontweight='bold')
ax.set_ylim(0, 55)
ax.grid(True, alpha=0.25, axis='y')
for bar, t in zip(bars, tasks):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{t}%', ha='center', fontsize=10, fontweight='bold')
ax.text(0.5, 48, 'Load Balancing Strategy', ha='center', fontsize=10, style='italic')
save_figure(fig, 'paper/figures/Figure17_Heterogeneity_Compensation.png', 'paper/figures/Figure17_Heterogeneity_Compensation.pdf')
save_figure(fig, 'report_graphs/Figure17_Heterogeneity_Compensation.png', 'report_graphs/Figure17_Heterogeneity_Compensation.pdf')
plt.close()

print("[18/20] Figure 18: MNIST Literature Comparison")
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(LIT_METHODS, LIT_ACCURACIES, color=LIT_COLORS, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Figure 18: MNIST Literature Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(97.5, 100.5)
ax.grid(True, alpha=0.25, axis='y')
for bar, acc in zip(bars, LIT_ACCURACIES):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{acc:.2f}%', ha='center', fontsize=9, fontweight='bold')
legend_elements = [
    Patch(facecolor='#95A5A6', edgecolor='black', label='Other Methods'),
    Patch(facecolor='#27AE60', edgecolor='black', label='PROPOSED METHOD')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
save_figure(fig, 'paper/figures/Figure18_MNIST_Literature_Comparison.png', 'paper/figures/Figure18_MNIST_Literature_Comparison.pdf')
save_figure(fig, 'report_graphs/Figure18_MNIST_Literature_Comparison.png', 'report_graphs/Figure18_MNIST_Literature_Comparison.pdf')
plt.close()

print("[19/20] Figure 19: Results Summary Table")
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
table_data = [
    ['Dataset', 'Grid Search Best', 'Proposed Method Mean', 'Std', 'Proposed Method Best', 'Improvement'],
    ['MNIST', '99.47%', '99.19%', '0.13%', '99.31%', '-0.28 pp'],
    ['FashionMNIST', '90.04%', '88.08%', '0.86%', '89.09%', '-1.96 pp'],
    ['CIFAR-10', '77.57%', '70.83%', '3.14%', '74.46%', '-6.74 pp']
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', 
                  colWidths=[0.14, 0.14, 0.18, 0.10, 0.14, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)
for i in range(6):
    table[(0, i)].set_facecolor('#2C3E50')
    table[(0, i)].set_text_props(color='white', fontweight='bold')
for i in range(1, 4):
    table[(i, 2)].set_facecolor('#D6EAF8')
    table[(i, 4)].set_facecolor('#D5F5E3')
ax.set_title('Figure 19: Results Summary Table', fontsize=14, fontweight='bold', pad=20)
save_figure(fig, 'paper/figures/Figure19_Results_Table.png', 'paper/figures/Figure19_Results_Table.pdf')
save_figure(fig, 'report_graphs/Figure19_Results_Table.png', 'report_graphs/Figure19_Results_Table.pdf')
plt.close()

print("[20/20] Figure 20: Complete Results Dashboard")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(datasets))
width = 0.35
bars1 = ax1.bar(x - width/2, [99.47, 90.04, 77.57], width, label='Grid Search', color='#95A5A6')
bars2 = ax1.bar(x + width/2, [99.19, 88.08, 70.83], width, label='Proposed Method', color='#2E86AB')
ax1.errorbar(x + width/2, [99.19, 88.08, 70.83], yerr=[0.13, 0.86, 3.14], fmt='none', ecolor='black', capsize=3)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=9)
ax1.set_ylabel('Accuracy (%)', fontsize=10)
ax1.set_title('A) Final Results', fontsize=11, fontweight='bold')
ax1.set_ylim(55, 102)
ax1.grid(True, alpha=0.25, axis='y')
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
for i, dataset in enumerate(['MNIST', 'FashionMNIST', 'CIFAR-10']):
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    ax2.barh(i, CLOUD_EDGE_BO[dataset]['mean'], xerr=CLOUD_EDGE_BO[dataset]['std'], 
             color=colors[i], edgecolor='black', capsize=3)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['MNIST', 'FashionMNIST', 'CIFAR-10'], fontsize=9)
ax2.set_xlabel('Accuracy (%)', fontsize=10)
ax2.set_title('B) Performance by Dataset', fontsize=11, fontweight='bold')
ax2.set_xlim(60, 102)
ax2.grid(True, alpha=0.25, axis='x')

ax3 = fig.add_subplot(gs[0, 2])
methods = ['Rand', 'Grid', 'Seq BO', 'Prop']
mnist_acc = [99.33, 99.47, 99.23, 99.19]
colors = ['#95A5A6', '#5D6D7E', '#F39C12', '#2E86AB']
bars = ax3.bar(methods, mnist_acc, color=colors, edgecolor='black', linewidth=1)
ax3.set_ylabel('Accuracy (%)', fontsize=10)
ax3.set_title('C) MNIST Methods', fontsize=11, fontweight='bold')
ax3.set_ylim(98.5, 100)
ax3.grid(True, alpha=0.25, axis='y')
for bar, val in zip(bars, mnist_acc):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', ha='center', fontsize=7)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(WORKERS, SPEEDUP, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax4.axvline(x=3, color='#E74C3C', linestyle='--', linewidth=2)
ax4.set_xlabel('Workers', fontsize=10)
ax4.set_ylabel('Speedup', fontsize=10)
ax4.set_title('D) Scalability', fontsize=11, fontweight='bold')
ax4.set_xlim(0.5, 6.5)
ax4.set_ylim(0, 5)
ax4.grid(True, alpha=0.25)

ax5 = fig.add_subplot(gs[1, 1])
methods = ['LeNet-5', 'PyTorch', 'Keras', 'ResNet', 'Proposed']
acc = [99.05, 99.47, 98.40, 99.10, 99.31]
colors = ['#95A5A6', '#5D6D7E', '#85929E', '#717D7E', '#27AE60']
bars = ax5.bar(methods, acc, color=colors, edgecolor='black', linewidth=1)
ax5.set_ylabel('Accuracy (%)', fontsize=10)
ax5.set_title('E) Literature Comparison', fontsize=11, fontweight='bold')
ax5.set_ylim(97.5, 100.5)
ax5.grid(True, alpha=0.25, axis='y')
for bar, val in zip(bars, acc):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', ha='center', fontsize=7)

ax6 = fig.add_subplot(gs[1, 2])
seeds = ['42', '123', '456', '789', '1010']
ce_acc = [98.98, 99.25, 99.25, 99.31, 99.15]
ax6.plot(seeds, ce_acc, 'o-', color='#3498DB', linewidth=2, markersize=8)
ax6.axhline(y=99.47, color='#E74C3C', linestyle='--', label='Grid Search: 99.47%')
ax6.set_xlabel('Random Seed', fontsize=10)
ax6.set_ylabel('Accuracy (%)', fontsize=10)
ax6.set_title('F) MNIST Stability', fontsize=11, fontweight='bold')
ax6.set_ylim(98.5, 100)
ax6.grid(True, alpha=0.25)
ax6.legend(fontsize=8)

fig.suptitle('Figure 20: Proposed Bayesian Optimization Method - Complete Results Dashboard', 
             fontsize=14, fontweight='bold')
save_figure(fig, 'paper/figures/Figure20_Complete_Dashboard.png', 'paper/figures/Figure20_Complete_Dashboard.pdf')
save_figure(fig, 'report_graphs/Figure20_Complete_Dashboard.png', 'report_graphs/Figure20_Complete_Dashboard.pdf')
plt.close()

print("\n" + "=" * 70)
print("ALL 20 FIGURES GENERATED SUCCESSFULLY (PNG + PDF)!")
print("=" * 70)
print("\nFigure Locations:")
print("  PNG: paper/figures/ and report_graphs/")
print("  PDF: paper/figures/ and report_graphs/")
print("\n" + "=" * 70)