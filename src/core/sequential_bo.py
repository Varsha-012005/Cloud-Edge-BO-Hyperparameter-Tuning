import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
import json
import sys
import os

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_objective(dataset_name):
    if dataset_name == 'mnist':
        from train_mnist import train_mnist
        def objective(lr, batch_size, num_epochs=15, dropout_rate=0.25):
            print(f"\n  Training MNIST: lr={lr:.6f}, bs={int(batch_size)}, epochs={int(num_epochs)}")
            accuracy, runtime = train_mnist(lr=lr, batch_size=int(batch_size), num_epochs=int(num_epochs))
            print(f"  -> Accuracy: {accuracy:.2f}%, Time: {runtime:.1f}s")
            return -accuracy
    elif dataset_name == 'fashionmnist':
        from train_fashionmnist import train_fashionmnist
        def objective(lr, batch_size, num_epochs=15, dropout_rate=0.25):
            print(f"\n  Training FashionMNIST: lr={lr:.6f}, bs={int(batch_size)}, epochs={int(num_epochs)}")
            accuracy, runtime = train_fashionmnist(lr=lr, batch_size=int(batch_size), num_epochs=int(num_epochs))
            print(f"  -> Accuracy: {accuracy:.2f}%, Time: {runtime:.1f}s")
            return -accuracy
    elif dataset_name == 'cifar10':
        from train_cifar10 import train_cifar10
        def objective(lr, batch_size, num_epochs=15, dropout_rate=0.25):
            print(f"\n  Training CIFAR-10: lr={lr:.6f}, bs={int(batch_size)}, epochs={int(num_epochs)}")
            accuracy, runtime = train_cifar10(lr=lr, batch_size=int(batch_size), num_epochs=int(num_epochs))
            print(f"  -> Accuracy: {accuracy:.2f}%, Time: {runtime:.1f}s")
            return -accuracy
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return objective

def run_sequential_bo(dataset_name='mnist', n_trials=25):
    print("=" * 60)
    print(f"SEQUENTIAL BAYESIAN OPTIMIZATION - {dataset_name.upper()}")
    print("=" * 60)
    
    # Add training folder to path
    training_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training')
    sys.path.insert(0, training_path)
    
    space = [
        Real(1e-5, 5e-3, "log-uniform", name="learning_rate"),
        Integer(32, 256, name="batch_size"),
        Integer(10, 25, name="num_epochs"),  # ✅ INCREASED TO 10-25!
        Real(0.0, 0.3, name="dropout_rate"),  # ✅ Added dropout
    ]
    
    objective = get_objective(dataset_name)
    
    @use_named_args(space)
    def objective_wrapper(**params):
        return objective(
            params["learning_rate"], 
            params["batch_size"],
            params["num_epochs"],
            params["dropout_rate"]
        )
    
    print(f"\nRunning {n_trials} sequential trials...")
    print(f"Search space:")
    print(f"  - Learning Rate: [1e-5, 5e-3] (log scale)")
    print(f"  - Batch Size: [32, 256]")
    print(f"  - Num Epochs: [10, 25]")  # ✅ Updated!
    print(f"  - Dropout Rate: [0.0, 0.3]")  # ✅ Added!
    print("-" * 40)
    
    result = gp_minimize(
        func=objective_wrapper,
        dimensions=space,
        n_calls=n_trials,
        n_initial_points=min(5, n_trials),
        random_state=42,
        verbose=True
    )
    
    best_lr = result.x[0]
    best_bs = result.x[1]
    best_epochs = result.x[2]
    best_dropout = result.x[3]
    best_acc = -result.fun
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Hyperparameters:")
    print(f"   Learning Rate: {best_lr:.6f}")
    print(f"   Batch Size: {int(best_bs)}")
    print(f"   Num Epochs: {int(best_epochs)}")
    print(f"   Dropout Rate: {best_dropout:.3f}")
    
    results = {
        'method': f'Sequential BO - {dataset_name}',
        'n_trials': n_trials,
        'best_accuracy': float(best_acc),
        'best_params': {
            'learning_rate': float(best_lr),
            'batch_size': int(best_bs),
            'num_epochs': int(best_epochs),
            'dropout_rate': float(best_dropout)
        },
        'all_trials': [
            {
                'learning_rate': float(x[0]),
                'batch_size': int(x[1]),
                'num_epochs': int(x[2]),
                'dropout_rate': float(x[3]),
                'accuracy': float(-y)
            }
            for x, y in zip(result.x_iters, result.func_vals)
        ]
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/sequential_bo_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/sequential_bo_{dataset_name}.json")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--trials', type=int, default=25)
    args = parser.parse_args()
    run_sequential_bo(dataset_name=args.dataset, n_trials=args.trials)
