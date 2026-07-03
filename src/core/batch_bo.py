"""
Batch Bayesian Optimization with q-UCB
For parallel hyperparameter tuning on multiple edge devices
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import matplotlib.pyplot as plt
import json
import time

class BatchBO:
    """Batch Bayesian Optimization with q-UCB acquisition"""
    
    def __init__(self, bounds, n_workers=3):
        """
        Args:
            bounds: List of [(min, max)] for each parameter
            n_workers: Number of parallel workers (edge devices)
        """
        self.bounds = np.array(bounds)
        self.n_workers = n_workers
        self.dim = len(bounds)
        
        # Initialize GP with Matrn 5/2 kernel
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        self.X = []  # Evaluated points
        self.y = []  # Observed values
        self.batch_history = []
        
    def acquisition(self, X, beta=2.0):
        """
        q-UCB acquisition function for a single point
        """
        X = np.atleast_2d(X)
        mean, std = self.gp.predict(X, return_std=True)
        ucb = mean + beta * std
        return ucb
    
    def propose_batch(self, n_points=None):
        """
        Propose a batch of hyperparameters for parallel evaluation
        Uses local penalization to avoid redundant points
        """
        if n_points is None:
            n_points = self.n_workers
            
        if len(self.X) == 0:
            # First batch: Latin Hypercube Sampling
            batch = self._latin_hypercube(n_points)
        else:
            # Sequential greedy selection with penalization
            batch = []
            candidates = self._generate_candidates(1000)
            
            for _ in range(n_points):
                # Calculate acquisition values
                acq_values = np.array([self.acquisition(c) for c in candidates])
                
                # Apply local penalization
                penalties = self._local_penalization(candidates, batch)
                penalized_acq = acq_values * penalties
                
                # Select best candidate
                best_idx = np.argmax(penalized_acq)
                best_point = candidates[best_idx]
                batch.append(best_point)
                
                # Remove selected point and nearby points
                candidates = self._remove_nearby(candidates, best_point)
            
        return np.array(batch)
    
    def _local_penalization(self, candidates, batch):
        """
        Apply local penalization to avoid selecting points too close
        """
        if len(self.X) == 0:
            return np.ones(len(candidates))
        
        # Get minimum distance from previous points
        min_dist = np.min([self._distance(x, y) for x in self.X for y in candidates], axis=0)
        
        # Penalty based on distance
        penalty = 1 - np.exp(-min_dist ** 2 / (0.1 ** 2))
        return penalty
    
    def _latin_hypercube(self, n_points):
        """Generate Latin Hypercube samples"""
        samples = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] + np.arange(n_points)) / n_points
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        return samples
    
    def _generate_candidates(self, n_candidates=1000):
        """Generate random candidates for optimization"""
        candidates = np.random.rand(n_candidates, self.dim)
        for i in range(self.dim):
            candidates[:, i] = self.bounds[i, 0] + candidates[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        return candidates
    
    def _distance(self, x1, x2):
        """Euclidean distance in normalized space"""
        x1_norm = (x1 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        x2_norm = (x2 - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return np.linalg.norm(x1_norm - x2_norm)
    
    def _remove_nearby(self, candidates, point, threshold=0.05):
        """Remove candidates too close to selected point"""
        distances = np.array([self._distance(p, point) for p in candidates])
        keep = distances > threshold
        return candidates[keep]
    
    def update(self, X_batch, y_batch):
        """
        Update GP with batch evaluation results
        """
        self.X.extend(X_batch)
        self.y.extend(y_batch)
        
        # Update Gaussian Process
        self.gp.fit(np.array(self.X), np.array(self.y))
        
        self.batch_history.append({
            'batch_size': len(X_batch),
            'best_y': max(y_batch),
            'X': X_batch,
            'y': y_batch
        })
    
    def get_best(self):
        """Get best observed point and value"""
        if len(self.y) == 0:
            return None, None
        best_idx = np.argmax(self.y)
        return self.X[best_idx], self.y[best_idx]
    
    def plot_convergence(self):
        """Plot optimization progress"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Best value over iterations
        best_values = []
        current_best = -np.inf
        for y_batch in self.y:
            current_best = max(current_best, y_batch)
            best_values.append(current_best)
        
        axes[0].plot(range(1, len(best_values)+1), best_values, 'b-', linewidth=2)
        axes[0].set_xlabel('Evaluation')
        axes[0].set_ylabel('Best Accuracy')
        axes[0].set_title('Batch BO Convergence')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Batch sizes
        batch_sizes = [b['batch_size'] for b in self.batch_history]
        axes[1].bar(range(1, len(batch_sizes)+1), batch_sizes)
        axes[1].set_xlabel('Batch Iteration')
        axes[1].set_ylabel('Number of Parallel Evaluations')
        axes[1].set_title('Parallel Batch Size')
        
        plt.tight_layout()
        plt.savefig('results/batch_bo_convergence.png', dpi=150)
        plt.show()
        
    def save_results(self, filename='results/batch_bo_results.json'):
        """Save optimization results"""
        best_X, best_y = self.get_best()
        results = {
            'best_params': {f'dim_{i}': float(best_X[i]) for i in range(self.dim)} if best_X is not None else None,
            'best_value': float(best_y) if best_y is not None else None,
            'n_evaluations': len(self.y),
            'n_batches': len(self.batch_history),
            'batch_history': self.batch_history
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f" Results saved to {filename}")


def objective_2d(x):
    """2D test objective function (mock for demonstration)"""
    return -((x[0] - 0.5)**2 + (x[1] - 0.5)**2) + np.random.normal(0, 0.05)

if __name__ == "__main__":
    print("=" * 60)
    print("BATCH BAYESIAN OPTIMIZATION with q-UCB")
    print("For Cloud-Edge Parallel Hyperparameter Tuning")
    print("=" * 60)
    
    # Define search space for 2D test
    bounds = [(0, 1), (0, 1)]  # Normalized space
    
    # Create Batch BO with 3 workers (for cloud-edge)
    batch_bo = BatchBO(bounds, n_workers=3)
    
    print(f"\nRunning with {batch_bo.n_workers} parallel workers...")
    
    # Run optimization for 5 batches (15 total evaluations)
    for batch_idx in range(5):
        print(f"\n Batch {batch_idx + 1}/5")
        print("-" * 40)
        
        # Propose batch of hyperparameters
        batch = batch_bo.propose_batch()
        
        # Simulate parallel evaluation on edge devices
        results = []
        for i, params in enumerate(batch):
            # In real scenario: send to edge device, get accuracy
            value = objective_2d(params)
            results.append(value)
            print(f"  Worker {i+1}: params={params}, value={value:.4f}")
        
        # Update model with batch results
        batch_bo.update(batch, results)
        
        # Show best so far
        best_X, best_y = batch_bo.get_best()
        print(f"  Best so far: {best_y:.4f}")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    best_X, best_y = batch_bo.get_best()
    print(f"\n Best value: {best_y:.4f}")
    print(f" Best parameters: {best_X}")
    
    batch_bo.plot_convergence()
    batch_bo.save_results()
