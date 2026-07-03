"""
GRID SEARCH - Baseline Method
"""

import numpy as np
import time
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def grid_search(dataset_name='mnist'):
    print("=" * 60)
    print(f"GRID SEARCH - {dataset_name.upper()}")
    print("=" * 60)
    
    if dataset_name == 'mnist':
        from src.training.train_mnist import train_mnist
        train_fn = train_mnist
    elif dataset_name == 'fashionmnist':
        from src.training.train_fashionmnist import train_fashionmnist
        train_fn = train_fashionmnist
    elif dataset_name == 'cifar10':
        from src.training.train_cifar10 import train_cifar10
        train_fn = train_cifar10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    learning_rates = [1e-5, 1e-4, 0.001, 0.01]
    batch_sizes = [32, 64, 128, 256]
    num_epochs = [10, 15, 20, 25]
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    
    total = len(learning_rates) * len(batch_sizes) * len(num_epochs) * len(dropout_rates)
    
    print(f"\nGrid size: {total} combinations")
    print(f"  - Learning Rates: {learning_rates}")
    print(f"  - Batch Sizes: {batch_sizes}")
    print(f"  - Num Epochs: {num_epochs}")
    print(f"  - Dropout Rates: {dropout_rates}")
    print("-" * 40)
    
    best_acc = -float('inf')
    best_params = None
    all_trials = []
    trial_count = 0
    
    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in num_epochs:
                for dropout in dropout_rates:
                    trial_count += 1
                    print(f"\nGrid {trial_count}/{total}: lr={lr:.6f}, bs={bs}, epochs={epochs}, dropout={dropout:.3f}")
                    
                    start_time = time.time()
                    accuracy, runtime = train_fn(
                        lr=lr,
                        batch_size=bs,
                        num_epochs=epochs
                    )
                    elapsed = time.time() - start_time
                    
                    print(f"  -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
                    
                    all_trials.append({
                        'learning_rate': float(lr),
                        'batch_size': int(bs),
                        'num_epochs': int(epochs),
                        'dropout_rate': float(dropout),
                        'accuracy': float(accuracy),
                        'runtime': float(elapsed)
                    })
                    
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_params = {
                            'learning_rate': float(lr),
                            'batch_size': int(bs),
                            'num_epochs': int(epochs),
                            'dropout_rate': float(dropout)
                        }
                        print(f"  NEW BEST: {best_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE!")
    print("=" * 60)
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Hyperparameters:")
    print(f"   Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"   Batch Size: {best_params['batch_size']}")
    print(f"   Num Epochs: {best_params['num_epochs']}")
    print(f"   Dropout Rate: {best_params['dropout_rate']:.3f}")
    
    results = {
        'method': f'Grid Search - {dataset_name}',
        'grid_size': total,
        'best_accuracy': float(best_acc),
        'best_params': best_params,
        'all_trials': all_trials
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/grid_search_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/grid_search_{dataset_name}.json")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashionmnist', 'cifar10'])
    args = parser.parse_args()
    grid_search(dataset_name=args.dataset)
