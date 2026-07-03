"""
RANDOM SEARCH - Baseline Method
"""

import numpy as np
import random
import time
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def random_search(dataset_name='mnist', n_trials=25):
    print("=" * 60)
    print(f"RANDOM SEARCH - {dataset_name.upper()}")
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
    
    print(f"\nRunning {n_trials} random trials...")
    print(f"Search space:")
    print(f"  - Learning Rate: [1e-5, 5e-3] (log scale)")
    print(f"  - Batch Size: [32, 256]")
    print(f"  - Num Epochs: [10, 25]")
    print(f"  - Dropout Rate: [0.0, 0.3]")
    print("-" * 40)
    
    best_acc = -float('inf')
    best_params = None
    all_trials = []
    
    for i in range(n_trials):
        lr = float(10 ** np.random.uniform(-5, -2))
        batch_size = int(np.random.choice([32, 64, 128, 256]))
        num_epochs = int(np.random.randint(10, 26))
        dropout_rate = float(np.random.uniform(0.0, 0.3))
        
        print(f"\nTrial {i+1}/{n_trials}: lr={lr:.6f}, bs={batch_size}, epochs={num_epochs}, dropout={dropout_rate:.3f}")
        
        start_time = time.time()
        accuracy, runtime = train_fn(
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        elapsed = time.time() - start_time
        
        print(f"  -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        all_trials.append({
            'learning_rate': float(lr),
            'batch_size': int(batch_size),
            'num_epochs': int(num_epochs),
            'dropout_rate': float(dropout_rate),
            'accuracy': float(accuracy),
            'runtime': float(elapsed)
        })
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = {
                'learning_rate': float(lr),
                'batch_size': int(batch_size),
                'num_epochs': int(num_epochs),
                'dropout_rate': float(dropout_rate)
            }
            print(f"  NEW BEST: {best_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("RANDOM SEARCH COMPLETE!")
    print("=" * 60)
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Hyperparameters:")
    print(f"   Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"   Batch Size: {best_params['batch_size']}")
    print(f"   Num Epochs: {best_params['num_epochs']}")
    print(f"   Dropout Rate: {best_params['dropout_rate']:.3f}")
    
    results = {
        'method': f'Random Search - {dataset_name}',
        'n_trials': n_trials,
        'best_accuracy': float(best_acc),
        'best_params': best_params,
        'all_trials': all_trials
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/random_search_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/random_search_{dataset_name}.json")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--trials', type=int, default=25)
    args = parser.parse_args()
    random_search(dataset_name=args.dataset, n_trials=args.trials)
