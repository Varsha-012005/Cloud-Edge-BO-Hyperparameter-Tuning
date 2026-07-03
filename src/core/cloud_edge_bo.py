"""
Cloud-Edge Bayesian Optimization - Fixed
"""

import numpy as np
import time
import json
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

class CloudEdgeBO:
    def __init__(self, bounds, n_workers=3):
        self.bounds = np.array(bounds)
        self.n_workers = n_workers
        self.dim = len(bounds)
        
        self.worker_configs = {
            0: {"name": "slow", "speed_factor": 0.5},
            1: {"name": "medium", "speed_factor": 1.0},
            2: {"name": "fast", "speed_factor": 2.0}
        }
        
        self.X = []
        self.y = []
        self.execution_log = []
        
    def propose_batch(self):
        if len(self.X) == 0:
            batch = []
            for i in range(self.n_workers):
                lr = 10 ** np.random.uniform(-5, -2)
                batch.append([lr])
            return np.array(batch)
        else:
            best_idx = np.argmax(self.y)
            best_lr = self.X[best_idx][0]
            batch = []
            for i in range(self.n_workers):
                noise = 10 ** np.random.uniform(-0.3, 0.3)
                lr = best_lr * noise
                lr = np.clip(lr, self.bounds[0][0], self.bounds[0][1])
                batch.append([lr])
            return np.array(batch)
    
    def train_on_edge(self, params, worker_id):
        """Train on edge worker with real training"""
        from src.training.train_mnist import train_mnist
        
        lr = params[0]
        worker = self.worker_configs[worker_id]
        print(f"     [{worker['name'].upper()}] Training: lr={lr:.6f}")
        
        start = time.time()
        
        # train_mnist returns (accuracy, runtime)
        accuracy, runtime = train_mnist(lr, 64, num_epochs=2)
        
        elapsed = time.time() - start
        
        print(f"        -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        return accuracy, elapsed
    
    def run_optimization(self, n_batches=4):
        print("=" * 70)
        print("CLOUD-EDGE BAYESIAN OPTIMIZATION")
        print("With REAL CNN Training on MNIST")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"   Workers: {self.n_workers} (slow, medium, fast)")
        print(f"   Dataset: MNIST")
        print(f"   Epochs per trial: 2")
        print(f"   Total batches: {n_batches}")
        print(f"   Total trials: {n_batches * self.n_workers}")
        
        for batch_idx in range(n_batches):
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print("=" * 70)
            
            batch = self.propose_batch()
            
            print("\n  Cloud proposes:")
            for i, params in enumerate(batch):
                print(f"     Worker {i+1} ({self.worker_configs[i]['name']}): lr={params[0]:.6f}")
            
            print("\n  Edge Workers Training...")
            print("-" * 50)
            
            results = []
            for worker_id, params in enumerate(batch):
                accuracy, runtime = self.train_on_edge(params, worker_id)
                results.append({
                    'worker_id': worker_id,
                    'worker_name': self.worker_configs[worker_id]['name'],
                    'learning_rate': float(params[0]),
                    'accuracy': float(accuracy),
                    'runtime': float(runtime)
                })
            
            y_batch = [r['accuracy'] for r in results]
            self.X.extend(batch.tolist())
            self.y.extend(y_batch)
            
            best_in_batch = max(results, key=lambda x: x['accuracy'])
            best_overall = max(self.y) if self.y else 0
            
            print(f"\n  Batch Summary:")
            print(f"     Best in batch: {best_in_batch['worker_name']} with {best_in_batch['accuracy']:.2f}%")
            print(f"     Best overall: {best_overall:.2f}%")
            
            self.execution_log.append({
                'batch_id': batch_idx + 1,
                'timestamp': datetime.now().isoformat(),
                'proposed_lrs': [float(p[0]) for p in batch],
                'results': results,
                'best_accuracy': float(best_in_batch['accuracy'])
            })
            
            print(f"\n  Progress: {((batch_idx + 1) / n_batches) * 100:.0f}%")
        
        return results
    
    def save_results(self):
        if not self.y:
            print("No results to save")
            return
            
        best_idx = int(np.argmax(self.y))
        best_accuracy = float(self.y[best_idx])
        best_lr = float(self.X[best_idx][0])
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Cloud-Edge Bayesian Optimization',
            'n_workers': self.n_workers,
            'worker_configs': {
                str(k): v for k, v in self.worker_configs.items()
            },
            'n_batches': len(self.execution_log),
            'n_trials': len(self.y),
            'best_accuracy': best_accuracy,
            'best_hyperparameters': {
                'learning_rate': best_lr
            },
            'all_trials': [
                {
                    'learning_rate': float(self.X[i][0]),
                    'accuracy': float(self.y[i])
                }
                for i in range(len(self.X))
            ],
            'execution_log': self.execution_log
        }
        
        os.makedirs('results', exist_ok=True)
        
        with open('results/cloud_edge_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to results/cloud_edge_results.json")
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
        print(f"Best Learning Rate: {best_lr:.6f}")
        print(f"\nTotal trials: {len(self.y)}")
        print(f"   Total batches: {len(self.execution_log)}")
        print(f"   Parallel workers: {self.n_workers}")

if __name__ == "__main__":
    bounds = [(1e-5, 1e-2)]
    
    print("This will run REAL CNN training (2 epochs per trial)")
    print("   12 total trials = ~18 minutes\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        cloud_edge = CloudEdgeBO(bounds, n_workers=3)
        cloud_edge.run_optimization(n_batches=4)
        cloud_edge.save_results()
    else:
        print("Cancelled")
