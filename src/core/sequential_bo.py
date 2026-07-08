"""
Sequential Bayesian Optimization - 3D Support
Optimizes: learning_rate, batch_size, num_epochs
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import time
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')


class SequentialBO:
    """Sequential Bayesian Optimization with 3D search space"""
    
    def __init__(self, bounds, beta=2.0, n_restarts=10, seed=42):
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.beta = beta
        self.seed = seed
        np.random.seed(seed)
        
        self.kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed
        )
        
        self.X = []
        self.y = []
        self.best_X = None
        self.best_y = -np.inf
        self.initial_design_size = max(5, 2 * self.dim)
    
    def _latin_hypercube(self, n_points):
        samples = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] + np.arange(n_points)) / n_points
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        np.random.shuffle(samples)
        return samples
    
    def _snap_to_valid(self, params):
        """Snap batch_size and epochs to valid discrete values"""
        lr = params[0]
        bs_val = params[1]
        ep_val = params[2]
        
        # Valid batch sizes: 16, 32, 64, 128
        valid_bs = [16, 32, 64, 128]
        bs = valid_bs[np.argmin([abs(bs_val - v) for v in valid_bs])]
        
        # Valid epochs: 2, 6
        valid_ep = [2, 6]
        ep = valid_ep[np.argmin([abs(ep_val - v) for v in valid_ep])]
        
        return [float(lr), int(bs), int(ep)]
    
    def _acquisition_ucb(self, X):
        X = np.atleast_2d(X)
        mean, std = self.gp.predict(X, return_std=True)
        return mean + self.beta * std
    
    def propose_next(self):
        if len(self.X) < self.initial_design_size:
            points = self._latin_hypercube(1)
            return self._snap_to_valid(points[0])
        
        n_candidates = 1000
        candidates = np.random.rand(n_candidates, self.dim)
        for i in range(self.dim):
            candidates[:, i] = self.bounds[i, 0] + candidates[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        
        acq_values = np.array([self._acquisition_ucb(c)[0] for c in candidates])
        best_idx = np.argmax(acq_values)
        return self._snap_to_valid(candidates[best_idx])
    
    def evaluate(self, params, dataset='mnist'):
        from src.training.train_mnist import train_mnist
        
        lr = params[0]
        bs = int(params[1])
        ep = int(params[2])
        
        print(f"  Training: lr={lr:.6f}, bs={bs}, epochs={ep}")
        
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
        
        start_time = time.time()
        accuracy, runtime = train_fn(lr=lr, batch_size=bs, num_epochs=ep)
        elapsed = time.time() - start_time
        
        print(f"  -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        return accuracy, elapsed
    
    def run_optimization(self, n_trials=12, dataset='mnist', seed=42):
        print("=" * 70)
        print("SEQUENTIAL BAYESIAN OPTIMIZATION (3D)")
        print("=" * 70)
        
        print(f"\nConfiguration:")
        print(f"  Search space: {self.dim} dimensions")
        print(f"  - Learning Rate: [{self.bounds[0][0]:.0e}, {self.bounds[0][1]:.0e}]")
        print(f"  - Batch Size: [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        print(f"  - Epochs: [{self.bounds[2][0]}, {self.bounds[2][1]}]")
        print(f"  Beta (UCB): {self.beta}")
        print(f"  Dataset: {dataset}")
        print(f"  Total trials: {n_trials}")
        print(f"  Seed: {seed}")
        print("-" * 70)
        
        all_trials = []
        
        for trial_idx in range(n_trials):
            print(f"\nTrial {trial_idx + 1}/{n_trials}")
            print("-" * 40)
            
            params = self.propose_next()
            print(f"  Proposed: lr={params[0]:.6f}, bs={params[1]}, epochs={params[2]}")
            
            accuracy, runtime = self.evaluate(params, dataset)
            
            self.X.append(params)
            self.y.append(accuracy)
            
            if accuracy > self.best_y:
                self.best_y = accuracy
                self.best_X = params
            
            if len(self.X) > 1:
                self.gp.fit(np.array(self.X), np.array(self.y))
            
            all_trials.append({
                'learning_rate': float(params[0]),
                'batch_size': int(params[1]),
                'num_epochs': int(params[2]),
                'accuracy': float(accuracy),
                'runtime': float(runtime)
            })
            
            print(f"  Best so far: {self.best_y:.2f}%")
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nBest Validation Accuracy: {self.best_y:.2f}%")
        if self.best_X:
            print(f"Best: lr={self.best_X[0]:.6f}, bs={self.best_X[1]}, epochs={self.best_X[2]}")
        
        results = {
            'method': 'Sequential BO',
            'dataset': dataset,
            'seed': seed,
            'n_trials': n_trials,
            'best_accuracy': float(self.best_y),
            'best_hyperparameters': {
                'learning_rate': float(self.best_X[0]) if self.best_X else None,
                'batch_size': int(self.best_X[1]) if self.best_X else None,
                'num_epochs': int(self.best_X[2]) if self.best_X else None
            },
            'all_trials': all_trials
        }
        
        os.makedirs('results', exist_ok=True)
        filename = f'results/sequential_bo_{dataset}_seed{seed}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        return results


def run_sequential_bo(dataset='mnist', n_trials=12, seed=42):
    bounds = [(1e-5, 1e-2), (16, 128), (2, 6)]
    optimizer = SequentialBO(bounds, beta=2.0, seed=seed)
    return optimizer.run_optimization(n_trials=n_trials, dataset=dataset, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_sequential_bo(dataset=args.dataset, n_trials=args.trials, seed=args.seed)
