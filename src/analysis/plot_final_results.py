# src/plot_final_results.py
import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
with open('results/sequential_bo_mnist.json', 'r') as f:
    mnist = json.load(f)
with open('results/sequential_bo_fashionmnist.json', 'r') as f:
    fashion = json.load(f)
with open('results/sequential_bo_cifar10.json', 'r') as f:
    cifar = json.load(f)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Dataset 1: MNIST
trials_mnist = [t['accuracy'] for t in mnist['all_trials']]
best_mnist = np.maximum.accumulate(trials_mnist)
axes[0].plot(range(1, 11), trials_mnist, 'bo-', alpha=0.5, label='Individual')
axes[0].plot(range(1, 11), best_mnist, 'r-', linewidth=2, label='Best so far')
axes[0].axhline(y=98.50, color='gray', linestyle='--', label='Baseline (98.50%)')
axes[0].set_xlabel('Trial Number')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title(f'MNIST - Best: {mnist["best_accuracy"]:.2f}%')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dataset 2: FashionMNIST
trials_fashion = [t['accuracy'] for t in fashion['all_trials']]
best_fashion = np.maximum.accumulate(trials_fashion)
axes[1].plot(range(1, 11), trials_fashion, 'go-', alpha=0.5, label='Individual')
axes[1].plot(range(1, 11), best_fashion, 'r-', linewidth=2, label='Best so far')
axes[1].axhline(y=89.96, color='gray', linestyle='--', label='Baseline (89.96%)')
axes[1].set_xlabel('Trial Number')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title(f'FashionMNIST - Best: {fashion["best_accuracy"]:.2f}%')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Dataset 3: CIFAR-10
trials_cifar = [t['accuracy'] for t in cifar['all_trials']]
best_cifar = np.maximum.accumulate(trials_cifar)
axes[2].plot(range(1, 11), trials_cifar, 'mo-', alpha=0.5, label='Individual')
axes[2].plot(range(1, 11), best_cifar, 'r-', linewidth=2, label='Best so far')
axes[2].axhline(y=65.00, color='gray', linestyle='--', label='Baseline (~65%)')
axes[2].set_xlabel('Trial Number')
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_title(f'CIFAR-10 - Best: {cifar["best_accuracy"]:.2f}%')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Sequential Bayesian Optimization Results on 3 Datasets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/three_datasets_results.png', dpi=150, bbox_inches='tight')
print(" Graph saved to results/three_datasets_results.png")
plt.show()