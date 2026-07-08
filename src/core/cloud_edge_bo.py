"""
Cloud-Edge Bayesian Optimization - 3D Support
Optimizes: learning_rate, batch_size, num_epochs
"""

import numpy as np
import time
import json
import sys
import os
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class HeterogeneityAwareBatchBO:
    def __init__(self, bounds, n_workers=3, beta=2.0, gamma=0.2, n_restarts=10, seed=42):
        self.bounds = np.array(bounds)
        self.n_workers = n_workers
        self.dim = len(bounds)
        self.beta = beta
        self.gamma = gamma
        self.n_restarts = n_restarts
        self.seed = seed
        np.random.seed(seed)
        
        self.worker_configs = {
            0: {"name": "slow", "speed_factor": 0.5, "instance_type": "t3.micro"},
            1: {"name": "medium", "speed_factor": 1.0, "instance_type": "t3.small"},
            2: {"name": "fast", "speed_factor": 2.0, "instance_type": "t3.medium"}
        }
        
        self.kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed
        )
        
        self.X = []
        self.y = []
        self.worker_assignment = []
        self.evaluation_time = []
        self.batch_history = []
        self.iteration_count = 0
        self.best_X = None
        self.best_y = -np.inf
        self.initial_design_size = max(5, 2 * self.dim)
        
        print("Initialized Heterogeneity-Aware Batch BO (3D)")
        print(f"  Dimensions: {self.dim}")
        print(f"  Workers: {n_workers}")
        print(f"  Beta: {beta}, Gamma: {gamma}")
        print(f"  Kernel: Matérn 5/2")
        print(f"  Seed: {seed}")
    
    def _latin_hypercube(self, n_points):
        samples = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] + np.arange(n_points)) / n_points
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        np.random.shuffle(samples)
        return samples
    
    def _snap_to_valid(self, params):
        lr = params[0]
        bs_val = params[1]
        ep_val = params[2]
        
        valid_bs = [16, 32, 64, 128]
        bs = valid_bs[np.argmin([abs(bs_val - v) for v in valid_bs])]
        
        valid_ep = [2, 6]
        ep = valid_ep[np.argmin([abs(ep_val - v) for v in valid_ep])]
        
        return [float(lr), int(bs), int(ep)]
    
    def _normalize(self, X):
        X = np.atleast_2d(X)
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0] + 1e-10)
    
    def _distance(self, x1, x2):
        x1_norm = self._normalize(np.array(x1).reshape(1, -1))
        x2_norm = self._normalize(np.array(x2).reshape(1, -1))
        return float(np.linalg.norm(x1_norm - x2_norm))
    
    def _local_penalization(self, candidates, selected_batch, penalization_radius=0.05):
        if len(selected_batch) == 0:
            return np.ones(len(candidates))
        
        penalties = np.ones(len(candidates))
        for selected in selected_batch:
            for i, candidate in enumerate(candidates):
                dist = self._distance(candidate, selected)
                if dist < penalization_radius:
                    penalties[i] *= (dist / penalization_radius) ** 2
        
        return penalties
    
    def _acquisition_ucb(self, X):
        X = np.atleast_2d(X)
        mean, std = self.gp.predict(X, return_std=True)
        return mean + self.beta * std
    
    def _acquisition_compensated(self, x, worker_id):
        ucb_value = float(self._acquisition_ucb(x)[0])
        speed_factor = self.worker_configs[worker_id]['speed_factor']
        estimated_time = 1.0 / speed_factor
        compensation = self.gamma * (1.0 / estimated_time)
        return ucb_value + compensation
    
    def propose_batch(self):
        if len(self.X) < self.initial_design_size:
            n_points = min(self.initial_design_size, self.n_workers)
            batch = self._latin_hypercube(n_points)
            snapped = [self._snap_to_valid(p) for p in batch]
            return [(snapped[i], i % self.n_workers) for i in range(len(snapped))]
        
        n_candidates = 2000
        candidates = np.random.rand(n_candidates, self.dim)
        for i in range(self.dim):
            candidates[:, i] = self.bounds[i, 0] + candidates[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        
        if self.best_X is not None:
            best_repeated = np.tile(self.best_X, (n_candidates // 2, 1))
            noise = np.random.randn(n_candidates // 2, self.dim) * 0.05
            candidates_around_best = best_repeated + noise * (self.bounds[:, 1] - self.bounds[:, 0])
            candidates_around_best = np.clip(candidates_around_best, self.bounds[:, 0], self.bounds[:, 1])
            candidates = np.vstack([candidates, candidates_around_best])
        
        batch = []
        for worker_id in range(self.n_workers):
            acq_values = np.array([
                self._acquisition_compensated(c, worker_id)
                for c in candidates
            ])
            
            penalties = self._local_penalization(candidates, [b[0] for b in batch])
            penalized_acq = acq_values * penalties
            
            best_idx = np.argmax(penalized_acq)
            best_candidate = candidates[best_idx].copy()
            
            batch.append((self._snap_to_valid(best_candidate), worker_id))
            candidates = np.delete(candidates, best_idx, axis=0)
        
        return batch
    
    def train_worker(self, params, worker_id, dataset='mnist'):
        learning_rate = params[0]
        batch_size = int(params[1])
        num_epochs = int(params[2])
        
        worker_name = self.worker_configs[worker_id]['name']
        speed_factor = self.worker_configs[worker_id]['speed_factor']
        
        print(f"  [{worker_name.upper()}] Training: lr={learning_rate:.6f}, bs={batch_size}, epochs={num_epochs}")
        
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
        # Pass the seed to training function
        accuracy, runtime = train_fn(
            lr=learning_rate, 
            batch_size=batch_size, 
            num_epochs=num_epochs,
            seed=self.seed  # FIXED: seed passed to training
        )
        elapsed = time.time() - start_time
        
        print(f"  [{worker_name.upper()}] -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        return {
            'worker_id': worker_id,
            'worker_name': worker_name,
            'speed_factor': speed_factor,
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'num_epochs': int(num_epochs),
            'accuracy': float(accuracy),
            'runtime': float(elapsed)
        }
    
    def run_optimization(self, n_batches=4, dataset='mnist', seed=42):
        print("=" * 70)
        print("CLOUD-EDGE BAYESIAN OPTIMIZATION (3D)")
        print("=" * 70)
        
        print(f"\nConfiguration:")
        print(f"  Search space: {self.dim} dimensions")
        print(f"  - Learning Rate: [{self.bounds[0][0]:.0e}, {self.bounds[0][1]:.0e}]")
        print(f"  - Batch Size: [{self.bounds[1][0]}, {self.bounds[1][1]}]")
        print(f"  - Epochs: [{self.bounds[2][0]}, {self.bounds[2][1]}]")
        print(f"  Workers: {self.n_workers}")
        print(f"  Beta: {self.beta}, Gamma: {self.gamma}")
        print(f"  Dataset: {dataset}")
        print(f"  Total batches: {n_batches}")
        print(f"  Total trials: {n_batches * self.n_workers}")
        print(f"  Seed: {seed}")
        print("-" * 70)
        
        os.makedirs('results', exist_ok=True)
        
        for batch_idx in range(n_batches):
            self.iteration_count += 1
            
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print("=" * 70)
            
            batch = self.propose_batch()
            
            print(f"\n  Proposed batch:")
            for i, (params, worker_id) in enumerate(batch):
                worker_name = self.worker_configs[worker_id]['name']
                speed_factor = self.worker_configs[worker_id]['speed_factor']
                print(f"    Worker {i+1} ({worker_name}, speed={speed_factor:.1f}x): lr={params[0]:.6f}, bs={params[1]}, epochs={params[2]}")
            
            print(f"\n  Training on edge workers...")
            print("-" * 50)
            
            batch_results = []
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self.train_worker, params, worker_id, dataset): (params, worker_id)
                    for params, worker_id in batch
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    batch_results.append(result)
                    print(f"  [{result['worker_name'].upper()}] Completed: {result['accuracy']:.2f}%")
            
            X_batch = [[r['learning_rate'], r['batch_size'], r['num_epochs']] for r in batch_results]
            y_batch = [r['accuracy'] for r in batch_results]
            times_batch = [r['runtime'] for r in batch_results]
            
            for i in range(len(batch_results)):
                self.X.append(X_batch[i])
                self.y.append(y_batch[i])
                self.worker_assignment.append(batch_results[i]['worker_id'])
                self.evaluation_time.append(times_batch[i])
            
            if len(self.X) > 0:
                self.gp.fit(np.array(self.X), np.array(self.y))
            
            best_in_batch_idx = np.argmax(y_batch)
            best_in_batch = batch_results[best_in_batch_idx]
            
            if best_in_batch['accuracy'] > self.best_y:
                self.best_y = best_in_batch['accuracy']
                self.best_X = [best_in_batch['learning_rate'], best_in_batch['batch_size'], best_in_batch['num_epochs']]
            
            print(f"\n  Batch Summary:")
            print(f"    Best in batch: {best_in_batch['worker_name']} with {best_in_batch['accuracy']:.2f}%")
            print(f"    Best overall: {self.best_y:.2f}%")
            print(f"    Batch times: min={min(times_batch):.1f}s, max={max(times_batch):.1f}s, avg={np.mean(times_batch):.1f}s")
            
            self.batch_history.append({
                'batch_id': batch_idx + 1,
                'timestamp': datetime.now().isoformat(),
                'results': batch_results,
                'best_accuracy': float(self.best_y),
                'best_lr': float(self.best_X[0]) if self.best_X else None,
                'best_bs': int(self.best_X[1]) if self.best_X else None,
                'best_ep': int(self.best_X[2]) if self.best_X else None
            })
            
            print(f"\n  Progress: {((batch_idx + 1) / n_batches) * 100:.0f}%")
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nBest Validation Accuracy: {self.best_y:.2f}%")
        if self.best_X:
            print(f"Best: lr={self.best_X[0]:.6f}, bs={self.best_X[1]}, epochs={self.best_X[2]}")
        print(f"\nTotal trials: {len(self.y)}")
        print(f"  Batches: {len(self.batch_history)}")
        print(f"  Workers: {self.n_workers}")
        
        return self.get_results()
    
    def get_results(self):
        results = {
            'method': 'Cloud-Edge Bayesian Optimization',
            'timestamp': datetime.now().isoformat(),
            'n_workers': self.n_workers,
            'worker_configs': self.worker_configs,
            'gp_kernel': str(self.kernel),
            'beta': self.beta,
            'gamma': self.gamma,
            'n_batches': len(self.batch_history),
            'n_trials': len(self.y),
            'best_accuracy': float(self.best_y) if self.best_y != -np.inf else None,
            'best_hyperparameters': {
                'learning_rate': float(self.best_X[0]) if self.best_X else None,
                'batch_size': int(self.best_X[1]) if self.best_X else None,
                'num_epochs': int(self.best_X[2]) if self.best_X else None
            },
            'all_trials': [
                {
                    'learning_rate': float(self.X[i][0]),
                    'batch_size': int(self.X[i][1]),
                    'num_epochs': int(self.X[i][2]),
                    'accuracy': float(self.y[i]),
                    'worker_id': self.worker_assignment[i],
                    'worker_name': self.worker_configs[self.worker_assignment[i]]['name'],
                    'runtime': float(self.evaluation_time[i])
                }
                for i in range(len(self.X))
            ],
            'batch_history': self.batch_history
        }
        
        with open('results/cloud_edge_bo_fixed_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to results/cloud_edge_bo_fixed_results.json")
        return results


def run_cloud_edge_bo_experiment(n_batches=4, beta=2.0, gamma=0.2, dataset='mnist', seed=42):
    bounds = [(1e-5, 1e-2), (16, 128), (2, 6)]
    optimizer = HeterogeneityAwareBatchBO(
        bounds=bounds,
        n_workers=3,
        beta=beta,
        gamma=gamma,
        seed=seed
    )
    return optimizer.run_optimization(n_batches=n_batches, dataset=dataset, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_cloud_edge_bo_experiment(
        n_batches=args.batches,
        beta=args.beta,
        gamma=args.gamma,
        dataset=args.dataset,
        seed=args.seed
    )
