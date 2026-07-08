"""
Random Search - 3D Support
Optimizes: learning_rate, batch_size, num_epochs
"""

import numpy as np
import time
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def random_search(dataset='mnist', n_trials=12, seed=42):
    np.random.seed(seed)
    
    print("=" * 70)
    print(f"RANDOM SEARCH (3D) - {dataset.upper()}")
    print("=" * 70)
    
    if dataset == 'mnist':
        from src.training.train_mnist import train_mnist
        train_fn = train_mnist
    elif dataset == 'fashionmnist':
        from src.training.train_fashionmnist import train_fashionmnist
        train_fn = train_fashionmnist
    elif dataset == 'cifar10':
        from src.training.train_cifar10 import train_cifar10
        train_fn = train_cifar10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"\nRunning {n_trials} random trials...")
    print(f"Search space:")
    print(f"  - Learning Rate: [1e-5, 1e-2] (log scale)")
    print(f"  - Batch Size: [16, 32, 64, 128]")
    print(f"  - Num Epochs: [2, 6]")
    print(f"  - Seed: {seed}")
    print("-" * 70)
    
    best_acc = -float('inf')
    best_params = None
    all_trials = []
    
    for i in range(n_trials):
        lr = float(10 ** np.random.uniform(-5, -2))
        bs = int(np.random.choice([16, 32, 64, 128]))
        ep = int(np.random.choice([2, 6]))
        
        print(f"\nTrial {i+1}/{n_trials}: lr={lr:.6f}, bs={bs}, epochs={ep}")
        
        start_time = time.time()
        accuracy, runtime = train_fn(lr=lr, batch_size=bs, num_epochs=ep)
        elapsed = time.time() - start_time
        
        print(f"  -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        all_trials.append({
            'learning_rate': float(lr),
            'batch_size': int(bs),
            'num_epochs': int(ep),
            'accuracy': float(accuracy),
            'runtime': float(elapsed)
        })
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = {'learning_rate': float(lr), 'batch_size': int(bs), 'num_epochs': int(ep)}
            print(f"  NEW BEST: {best_acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("RANDOM SEARCH COMPLETE!")
    print("=" * 70)
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Best: lr={best_params['learning_rate']:.6f}, bs={best_params['batch_size']}, epochs={best_params['num_epochs']}")
    
    results = {
        'method': f'Random Search - {dataset}',
        'seed': seed,
        'n_trials': n_trials,
        'best_accuracy': float(best_acc),
        'best_hyperparameters': best_params,
        'all_trials': all_trials
    }
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/random_search_{dataset}_seed{seed}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random_search(dataset=args.dataset, n_trials=args.trials, seed=args.seed)
