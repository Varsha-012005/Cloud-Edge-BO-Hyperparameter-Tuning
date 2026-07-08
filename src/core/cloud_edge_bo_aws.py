"""
Cloud-Edge Bayesian Optimization with AWS Infrastructure
Full implementation matching paper description:
- Matérn 5/2 Gaussian Process surrogate (Equation 2)
- q-UCB batch acquisition with heterogeneity compensation (Equation 8)
- True parallel execution across heterogeneous edge workers via SQS
- Worker-speed-aware task routing with gamma compensation term
- S3 for result storage, SQS for task distribution
"""

import numpy as np
import time
import json
import sys
import os
import boto3
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'training'))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class CloudEdgeBOAWS:
    """
    Cloud-Edge Bayesian Optimization with AWS Infrastructure
    
    Implements the full method from the paper:
    - Matérn 5/2 GP surrogate (Equation 2)
    - q-UCB batch acquisition with heterogeneity compensation (Equation 8)
    - Worker-speed-aware task routing
    - True parallel execution using SQS queues
    - S3 for result storage
    """
    
    def __init__(self, bounds, config_path='aws_config.json', 
                 n_workers=3, beta=2.0, gamma=0.2, n_restarts=10):
        
        self.bounds = np.array(bounds)
        self.n_workers = n_workers
        self.dim = len(bounds)
        self.beta = beta
        self.gamma = gamma
        self.n_restarts = n_restarts
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.bucket_name = self.config['bucket_name']
        self.queue_url = self.config['queue_url']
        self.region = self.config.get('region', 'us-east-1')
        
        self.s3 = boto3.client('s3', region_name=self.region)
        self.sqs = boto3.client('sqs', region_name=self.region)
        
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
            random_state=42
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
        self.results = None
        
        print("=" * 70)
        print("CLOUD-EDGE BAYESIAN OPTIMIZATION WITH AWS")
        print("=" * 70)
        print(f"  Bucket: {self.bucket_name}")
        print(f"  Queue: {self.queue_url}")
        print(f"  Search dimensions: {self.dim}")
        print(f"  Workers: {n_workers}")
        print(f"  Worker speeds: {[self.worker_configs[i]['speed_factor'] for i in range(n_workers)]}")
        print(f"  Beta (UCB): {beta}")
        print(f"  Gamma (heterogeneity compensation): {gamma}")
        print(f"  Kernel: Matérn 5/2 (Equation 2)")
        print(f"  Acquisition: q-UCB with heterogeneity compensation (Equation 8)")
        print("=" * 70)
    
    def _latin_hypercube(self, n_points):
        samples = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            samples[:, i] = (samples[:, i] + np.arange(n_points)) / n_points
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        np.random.shuffle(samples)
        return samples
    
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
            return [(params, i % self.n_workers) for i, params in enumerate(batch)]
        
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
            
            batch.append((best_candidate, worker_id))
            candidates = np.delete(candidates, best_idx, axis=0)
        
        return batch
    
    def dispatch_task_to_sqs(self, params, worker_id, task_id, dataset='mnist'):
        worker_name = self.worker_configs[worker_id]['name']
        
        task = {
            'task_id': task_id,
            'worker_id': worker_id,
            'worker_name': worker_name,
            'dataset': dataset,
            'learning_rate': float(params[0]),
            'batch_size': 64,
            'num_epochs': 2,
            'timestamp': time.time(),
            's3_bucket': self.bucket_name,
            's3_results_prefix': f'results/task_{task_id}/'
        }
        
        try:
            response = self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(task, cls=NumpyEncoder),
                MessageAttributes={
                    'WorkerId': {
                        'DataType': 'String',
                        'StringValue': str(worker_id)
                    },
                    'TaskId': {
                        'DataType': 'String',
                        'StringValue': str(task_id)
                    }
                }
            )
            print(f"  Dispatched task {task_id} to {worker_name} worker")
            return True, response.get('MessageId')
        except Exception as e:
            print(f"  Failed to dispatch task {task_id}: {e}")
            return False, None
    
    def poll_results_from_s3(self, task_ids, timeout=600, poll_interval=10):
        results = {}
        start_time = time.time()
        
        print(f"  Polling for results from {len(task_ids)} tasks...")
        
        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            for task_id in task_ids:
                if task_id in results:
                    continue
                
                key = f'results/task_{task_id}/result.json'
                try:
                    response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
                    result = json.loads(response['Body'].read().decode('utf-8'))
                    results[task_id] = result
                    print(f"  Received result from task {task_id}: {result['accuracy']:.2f}%")
                except self.s3.exceptions.NoSuchKey:
                    continue
                except Exception as e:
                    print(f"  Error reading result for task {task_id}: {e}")
            
            if len(results) < len(task_ids):
                time.sleep(poll_interval)
        
        for task_id in task_ids:
            if task_id not in results:
                print(f"  WARNING: Task {task_id} timed out")
                results[task_id] = {
                    'accuracy': 0.0,
                    'runtime': 0.0,
                    'error': 'Timeout'
                }
        
        return [results[tid] for tid in task_ids]
    
    def run_optimization(self, n_batches=4, dataset='mnist'):
        print("\n" + "=" * 70)
        print("CLOUD-EDGE BAYESIAN OPTIMIZATION ON AWS")
        print("Matérn 5/2 GP + q-UCB + Heterogeneity Compensation")
        print("=" * 70)
        
        print(f"\nAWS Configuration:")
        print(f"  S3 Bucket: {self.bucket_name}")
        print(f"  SQS Queue: {self.queue_url}")
        print(f"\nOptimization Configuration:")
        print(f"  Search space: {self.dim} dimensions")
        print(f"  Workers: {self.n_workers} (slow, medium, fast)")
        print(f"  Speed factors: {[self.worker_configs[i]['speed_factor'] for i in range(self.n_workers)]}")
        print(f"  Beta (UCB exploration): {self.beta}")
        print(f"  Gamma (heterogeneity compensation): {self.gamma}")
        print(f"  Kernel: Matérn 5/2 (Equation 2)")
        print(f"  Acquisition: q-UCB with heterogeneity compensation (Equation 8)")
        print(f"  Dataset: {dataset}")
        print(f"  Total batches: {n_batches}")
        print(f"  Total trials: {n_batches * self.n_workers}")
        print("-" * 70)
        
        os.makedirs('results', exist_ok=True)
        
        for batch_idx in range(n_batches):
            self.iteration_count += 1
            
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print("=" * 70)
            
            batch = self.propose_batch()
            print(f"\n  q-UCB with heterogeneity compensation proposed:")
            for i, (params, worker_id) in enumerate(batch):
                worker_name = self.worker_configs[worker_id]['name']
                speed_factor = self.worker_configs[worker_id]['speed_factor']
                print(f"    Worker {i+1} ({worker_name}, speed={speed_factor:.1f}x): lr={params[0]:.6f}")
            
            print(f"\n  Dispatching tasks to SQS...")
            print("-" * 50)
            
            task_ids = []
            for i, (params, worker_id) in enumerate(batch):
                task_id = self.iteration_count * 100 + i
                success, msg_id = self.dispatch_task_to_sqs(params, worker_id, task_id, dataset)
                if success:
                    task_ids.append(task_id)
            
            print(f"\n  Waiting for edge workers to complete...")
            print("-" * 50)
            
            results = self.poll_results_from_s3(task_ids, timeout=600, poll_interval=10)
            
            print(f"\n  Processing results...")
            print("-" * 50)
            
            for result in results:
                if 'error' in result:
                    print(f"    Task {result.get('task_id', 'unknown')}: ERROR - {result['error']}")
                    continue
                
                accuracy = result['accuracy']
                runtime = result['runtime']
                learning_rate = result['learning_rate']
                worker_id = result['worker_id']
                worker_name = self.worker_configs[worker_id]['name']
                
                print(f"    [{worker_name.upper()}] lr={learning_rate:.6f}, acc={accuracy:.2f}%, time={runtime:.1f}s")
                
                self.X.append([learning_rate])
                self.y.append(accuracy)
                self.worker_assignment.append(worker_id)
                self.evaluation_time.append(runtime)
            
            if len(self.X) > 0:
                self.gp.fit(np.array(self.X), np.array(self.y))
            
            valid_results = [r for r in results if 'error' not in r]
            if valid_results:
                best_in_batch = max(valid_results, key=lambda x: x['accuracy'])
                if best_in_batch['accuracy'] > self.best_y:
                    self.best_y = best_in_batch['accuracy']
                    self.best_X = [best_in_batch['learning_rate']]
            
            print(f"\n  Batch Summary:")
            if valid_results:
                best_in_batch = max(valid_results, key=lambda x: x['accuracy'])
                print(f"    Best in batch: {self.worker_configs[best_in_batch['worker_id']]['name']} with {best_in_batch['accuracy']:.2f}%")
            print(f"    Best overall: {self.best_y:.2f}%")
            
            self.batch_history.append({
                'batch_id': batch_idx + 1,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'best_accuracy': float(self.best_y),
                'best_lr': float(self.best_X[0]) if self.best_X else None
            })
            
            print(f"\n  Progress: {((batch_idx + 1) / n_batches) * 100:.0f}%")
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nBest Validation Accuracy: {self.best_y:.2f}%")
        print(f"Best Learning Rate: {self.best_X[0]:.6f}" if self.best_X else "N/A")
        print(f"\nTotal trials: {len(self.y)}")
        print(f"  Batches: {len(self.batch_history)}")
        print(f"  Workers: {self.n_workers}")
        
        return self.get_results()
    
    def get_results(self):
        results = {
            'method': 'Cloud-Edge Bayesian Optimization with AWS',
            'timestamp': datetime.now().isoformat(),
            'aws_bucket': self.bucket_name,
            'aws_queue': self.queue_url,
            'n_workers': self.n_workers,
            'worker_configs': {
                str(k): {
                    'name': v['name'],
                    'speed_factor': v['speed_factor'],
                    'instance_type': v['instance_type']
                }
                for k, v in self.worker_configs.items()
            },
            'gp_kernel': str(self.kernel),
            'beta': self.beta,
            'gamma': self.gamma,
            'n_batches': len(self.batch_history),
            'n_trials': len(self.y),
            'best_accuracy': float(self.best_y) if self.best_y != -np.inf else None,
            'best_hyperparameters': {
                'learning_rate': float(self.best_X[0]) if self.best_X else None
            },
            'all_trials': [
                {
                    'learning_rate': float(self.X[i][0]),
                    'accuracy': float(self.y[i]),
                    'worker_id': self.worker_assignment[i],
                    'worker_name': self.worker_configs[self.worker_assignment[i]]['name'],
                    'runtime': float(self.evaluation_time[i])
                }
                for i in range(len(self.X))
            ],
            'batch_history': self.batch_history
        }
        
        with open('results/cloud_edge_bo_aws_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key='results/cloud_edge_bo_results.json',
                Body=json.dumps(results, indent=2, cls=NumpyEncoder)
            )
            print(f"\nResults saved to S3: s3://{self.bucket_name}/results/cloud_edge_bo_results.json")
        except Exception as e:
            print(f"\nCould not save to S3: {e}")
        
        print(f"\nResults saved locally to results/cloud_edge_bo_aws_results.json")
        return results


def run_cloud_edge_bo_aws(n_batches=4, beta=2.0, gamma=0.2, dataset='mnist', config_path='aws_config.json'):
    bounds = [(1e-5, 1e-2)]
    optimizer = CloudEdgeBOAWS(
        bounds=bounds,
        config_path=config_path,
        n_workers=3,
        beta=beta,
        gamma=gamma
    )
    return optimizer.run_optimization(n_batches=n_batches, dataset=dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Cloud-Edge Bayesian Optimization with AWS')
    parser.add_argument('--batches', type=int, default=4, help='Number of batch iterations')
    parser.add_argument('--beta', type=float, default=2.0, help='UCB exploration parameter')
    parser.add_argument('--gamma', type=float, default=0.2, help='Heterogeneity compensation weight')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--config', type=str, default='aws_config.json', help='Path to AWS config file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLOUD-EDGE BAYESIAN OPTIMIZATION ON AWS")
    print("Matérn 5/2 GP + q-UCB + Heterogeneity Compensation")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Batches: {args.batches} (={args.batches * 3} total trials)")
    print(f"  Beta: {args.beta}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Dataset: {args.dataset}")
    
    results = run_cloud_edge_bo_aws(
        n_batches=args.batches,
        beta=args.beta,
        gamma=args.gamma,
        dataset=args.dataset,
        config_path=args.config
    )
