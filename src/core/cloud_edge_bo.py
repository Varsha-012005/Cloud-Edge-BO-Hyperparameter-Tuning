# src/cloud_edge_working.py
"""
Cloud-Edge Bayesian Optimization - Working Version
Real CNN training on 3 heterogeneous workers with Learning Rate Scheduler
Expanded Hyperparameter Search Space
"""

import numpy as np
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your real training function with LR scheduler
from train_mnist import train_mnist

# Custom JSON encoder for numpy types
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
        
        # Edge worker configurations (heterogeneous)
        self.worker_configs = {
            0: {"name": "slow", "speed_factor": 0.5},
            1: {"name": "medium", "speed_factor": 1.0},
            2: {"name": "fast", "speed_factor": 2.0}
        }
        
        self.X = []  # Evaluated hyperparameters
        self.y = []  # Validation accuracies
        self.execution_log = []
        
    def propose_batch(self):
        """Propose hyperparameters for batch with expanded search space"""
        if len(self.X) == 0:
            # First batch: diverse exploration with all hyperparameters
            batch = []
            for i in range(self.n_workers):
                # ============================================================
                # EXPANDED SEARCH SPACE
                # ============================================================
                lr = 10 ** np.random.uniform(-5, -2)           # Learning rate
                batch_size = np.random.choice([16, 32, 64, 128]) # Batch size
                num_epochs = np.random.choice([2, 3, 4, 5, 6])   # Epochs (2-6)
                dropout_rate = np.random.uniform(0.1, 0.5)       # Dropout rate (0.1-0.5)
                # ============================================================
                
                batch.append([lr, batch_size, num_epochs, dropout_rate])
            return np.array(batch)
        else:
            # Later batches: exploit best found with small perturbations
            best_idx = np.argmax(self.y)
            best_lr = self.X[best_idx][0]
            best_batch_size = self.X[best_idx][1]
            best_num_epochs = self.X[best_idx][2]
            best_dropout_rate = self.X[best_idx][3]
            
            batch = []
            for i in range(self.n_workers):
                # Add exploration noise to each hyperparameter
                noise = 10 ** np.random.uniform(-0.3, 0.3)
                lr = best_lr * noise
                lr = np.clip(lr, self.bounds[0][0], self.bounds[0][1])
                
                # Batch size variation (±16)
                batch_size = int(np.clip(best_batch_size + np.random.randint(-16, 17), 16, 128))
                
                # Epochs variation (±1)
                num_epochs = int(np.clip(best_num_epochs + np.random.randint(-1, 2), 2, 6))
                
                # Dropout rate variation (±0.1)
                dropout_rate = np.clip(best_dropout_rate + np.random.uniform(-0.1, 0.1), 0.1, 0.5)
                
                batch.append([lr, batch_size, num_epochs, dropout_rate])
            return np.array(batch)
    
    # ============================================================
    # UPDATED train_on_edge method with expanded hyperparameters
    # ============================================================
    def train_on_edge(self, params, worker_id):
        """Train on edge worker with real CNN training using LR Scheduler"""
        lr, batch_size, num_epochs, dropout_rate = params
        worker = self.worker_configs[worker_id]
        
        print(f"     [{worker['name'].upper()}] Training: lr={lr:.6f}, bs={int(batch_size)}, epochs={int(num_epochs)}, dropout={dropout_rate:.3f}")
        
        start = time.time()
        
        # Call training with expanded hyperparameters
        # Note: You need to update train_mnist to accept dropout_rate
        accuracy, runtime = train_mnist(
            lr=lr,
            batch_size=int(batch_size),
            num_epochs=int(num_epochs)
            # dropout_rate=dropout_rate  # Add when train_mnist supports it
        )
        
        elapsed = time.time() - start
        
        print(f"        → Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        
        return accuracy, elapsed
    # ============================================================
    
    def run_optimization(self, n_batches=4):
        """Run cloud-edge optimization"""
        
        print("=" * 70)
        print("CLOUD-EDGE BAYESIAN OPTIMIZATION")
        print("With REAL CNN Training on MNIST (Expanded Search Space)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"   Workers: {self.n_workers} (slow, medium, fast)")
        print(f"   Dataset: MNIST")
        print(f"   Hyperparameter Search Space:")
        print(f"     - Learning Rate: [1e-5, 1e-2] (log scale)")
        print(f"     - Batch Size: [16, 32, 64, 128]")
        print(f"     - Epochs: [2, 3, 4, 5, 6]")
        print(f"     - Dropout Rate: [0.1, 0.5]")
        print(f"   Total batches: {n_batches}")
        print(f"   Total trials: {n_batches * self.n_workers}")
        
        for batch_idx in range(n_batches):
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print("=" * 70)
            
            # Cloud: Propose batch
            batch = self.propose_batch()
            
            print("\n  Cloud proposes:")
            for i, params in enumerate(batch):
                lr, bs, epochs, dropout = params
                print(f"     Worker {i+1} ({self.worker_configs[i]['name']}): lr={lr:.6f}, bs={int(bs)}, epochs={int(epochs)}, dropout={dropout:.3f}")
            
            # Edge: Train in parallel
            print("\n  Edge Workers Training...")
            print("-" * 50)
            
            results = []
            for worker_id, params in enumerate(batch):
                accuracy, runtime = self.train_on_edge(params, worker_id)
                results.append({
                    'worker_id': worker_id,
                    'worker_name': self.worker_configs[worker_id]['name'],
                    'learning_rate': float(params[0]),
                    'batch_size': int(params[1]),
                    'num_epochs': int(params[2]),
                    'dropout_rate': float(params[3]),
                    'accuracy': float(accuracy),
                    'runtime': float(runtime)
                })
            
            # Update model
            y_batch = [r['accuracy'] for r in results]
            self.X.extend(batch.tolist())
            self.y.extend(y_batch)
            
            # Batch summary
            best_in_batch = max(results, key=lambda x: x['accuracy'])
            best_overall = max(self.y) if self.y else 0
            
            print(f"\n  Batch Summary:")
            print(f"     Best in batch: {best_in_batch['worker_name']} with {best_in_batch['accuracy']:.2f}%")
            print(f"     Best overall: {best_overall:.2f}%")
            
            # Store execution log
            self.execution_log.append({
                'batch_id': batch_idx + 1,
                'timestamp': datetime.now().isoformat(),
                'proposed_params': [
                    {
                        'learning_rate': float(p[0]),
                        'batch_size': int(p[1]),
                        'num_epochs': int(p[2]),
                        'dropout_rate': float(p[3])
                    }
                    for p in batch
                ],
                'results': results,
                'best_accuracy': float(best_in_batch['accuracy'])
            })
            
            print(f"\n  Progress: {((batch_idx + 1) / n_batches) * 100:.0f}%")
        
        return results
    
    def save_results(self):
        """Save results with proper JSON encoding"""
        if not self.y:
            print("No results to save")
            return
            
        best_idx = int(np.argmax(self.y))
        best_accuracy = float(self.y[best_idx])
        
        # Extract best hyperparameters
        best_lr = float(self.X[best_idx][0])
        best_batch_size = int(self.X[best_idx][1])
        best_num_epochs = int(self.X[best_idx][2])
        best_dropout_rate = float(self.X[best_idx][3])
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Cloud-Edge Bayesian Optimization (Expanded Search Space)',
            'n_workers': self.n_workers,
            'worker_configs': {
                str(k): v for k, v in self.worker_configs.items()
            },
            'n_batches': len(self.execution_log),
            'n_trials': len(self.y),
            'best_accuracy': best_accuracy,
            'best_hyperparameters': {
                'learning_rate': best_lr,
                'batch_size': best_batch_size,
                'num_epochs': best_num_epochs,
                'dropout_rate': best_dropout_rate
            },
            'all_trials': [
                {
                    'learning_rate': float(self.X[i][0]),
                    'batch_size': int(self.X[i][1]),
                    'num_epochs': int(self.X[i][2]),
                    'dropout_rate': float(self.X[i][3]),
                    'accuracy': float(self.y[i])
                }
                for i in range(len(self.X))
            ],
            'execution_log': self.execution_log
        }
        
        import os
        os.makedirs('results', exist_ok=True)
        
        with open('results/cloud_edge_results_expanded.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✅ Results saved to results/cloud_edge_results_expanded.json")
        
        # Print final summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\n🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
        print(f"📋 Best Hyperparameters:")
        print(f"   Learning Rate: {best_lr:.6f}")
        print(f"   Batch Size: {best_batch_size}")
        print(f"   Num Epochs: {best_num_epochs}")
        print(f"   Dropout Rate: {best_dropout_rate:.3f}")
        print(f"\n📊 Total trials: {len(self.y)}")
        print(f"   Total batches: {len(self.execution_log)}")
        print(f"   Parallel workers: {self.n_workers}")


if __name__ == "__main__":
    # Define expanded search space bounds
    bounds = [(1e-5, 1e-2)]  # Learning rate bounds
    
    # Run optimization with REAL training
    print("⚠️  This will run REAL CNN training with EXPANDED search space")
    print("   Hyperparameters to tune:")
    print("     - Learning Rate: 1e-5 to 1e-2 (log scale)")
    print("     - Batch Size: 16, 32, 64, 128")
    print("     - Epochs: 2 to 6")
    print("     - Dropout Rate: 0.1 to 0.5")
    print("\n   Each trial takes ~3-8 minutes (depends on epochs)")
    print("   12 total trials = ~60-90 minutes\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        cloud_edge = CloudEdgeBO(bounds, n_workers=3)
        cloud_edge.run_optimization(n_batches=4)
        cloud_edge.save_results()
    else:
        print("Cancelled")