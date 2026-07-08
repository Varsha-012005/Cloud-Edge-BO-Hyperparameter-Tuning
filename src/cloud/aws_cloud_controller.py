"""
AWS Cloud Controller for Cloud-Edge Bayesian Optimization
"""

import boto3
import json
import time
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class CloudController:
    def __init__(self, config_path='aws_config.json', beta=2.0, gamma=0.2, seed=42):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.bucket_name = self.config['bucket_name']
        self.queue_url = self.config['queue_url']
        self.region = self.config.get('region', 'ap-south-1')
        self.seed = seed

        np.random.seed(seed)

        self.s3 = boto3.client('s3', region_name=self.region)
        self.sqs = boto3.client('sqs', region_name=self.region)

        self.kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed
        )

        self.beta = beta
        self.gamma = gamma

        self.worker_configs = {
            0: {"name": "slow", "speed_factor": 0.5},
            1: {"name": "medium", "speed_factor": 1.0},
            2: {"name": "fast", "speed_factor": 2.0}
        }

        self.X = []
        self.y = []
        self.task_counter = 0
        self.best_X = None

        self.valid_bs = [16, 32, 64, 128]
        self.valid_ep = [2, 4]

        print("=" * 70)
        print("AWS CLOUD CONTROLLER")
        print("Matern 5/2 GP + q-UCB + Heterogeneity Compensation")
        print("=" * 70)
        print(f"  S3 Bucket: {self.bucket_name}")
        print(f"  SQS Queue: {self.queue_url}")
        print(f"  Beta (UCB): {beta}")
        print(f"  Gamma (heterogeneity): {gamma}")
        print(f"  Seed: {seed}")
        print("=" * 70)

    def _snap_to_valid(self, params):
        lr = params[0]
        bs_val = params[1]
        ep_val = params[2]

        bs = self.valid_bs[np.argmin([abs(bs_val - v) for v in self.valid_bs])]
        ep = self.valid_ep[np.argmin([abs(ep_val - v) for v in self.valid_ep])]

        return [float(lr), int(bs), int(ep)]

    def propose_batch(self, n_workers=3):
        if len(self.X) < 5:
            batch = []
            for i in range(n_workers):
                lr = 10 ** np.random.uniform(-5, -2)
                bs = np.random.choice(self.valid_bs)
                ep = np.random.choice(self.valid_ep)
                batch.append(([lr, bs, ep], i))
            return batch

        n_candidates = 1000
        candidates = np.random.rand(n_candidates, 3)
        candidates[:, 0] = candidates[:, 0] * (1e-2 - 1e-5) + 1e-5
        candidates[:, 1] = np.random.choice(self.valid_bs, n_candidates)
        candidates[:, 2] = np.random.choice(self.valid_ep, n_candidates)

        if self.best_X is not None:
            best_repeated = np.tile(self.best_X, (n_candidates // 2, 1))
            noise = np.random.randn(n_candidates // 2, 3) * 0.05
            noise[:, 0] = noise[:, 0] * (1e-2 - 1e-5)
            candidates_around_best = best_repeated + noise
            candidates_around_best[:, 0] = np.clip(candidates_around_best[:, 0], 1e-5, 1e-2)
            candidates_around_best[:, 1] = np.clip(candidates_around_best[:, 1], 16, 128)
            candidates_around_best[:, 2] = np.clip(candidates_around_best[:, 2], 2, 4)
            for i in range(len(candidates_around_best)):
                candidates_around_best[i] = self._snap_to_valid(candidates_around_best[i])
            candidates = np.vstack([candidates, candidates_around_best])

        batch = []
        for worker_id in range(n_workers):
            acq_values = []
            for c in candidates:
                mean, std = self.gp.predict([c], return_std=True)
                ucb = float(mean[0]) + self.beta * float(std[0])
                speed_factor = self.worker_configs[worker_id]['speed_factor']
                estimated_time = 1.0 / speed_factor
                compensation = self.gamma * (1.0 / estimated_time)
                acq_values.append(ucb + compensation)

            best_idx = np.argmax(acq_values)
            best_candidate = candidates[best_idx].copy()
            batch.append(([float(best_candidate[0]), int(best_candidate[1]), int(best_candidate[2])], worker_id))
            candidates = np.delete(candidates, best_idx, axis=0)

        return batch

    def dispatch_batch(self, batch, dataset='mnist', run_id=None):
        if run_id is None:
            run_id = f"{dataset}_seed{self.seed}_{int(time.time())}"
        
        print(f"\nDispatching {len(batch)} tasks...")
        print(f"  Run ID: {run_id}")
        print("-" * 50)

        task_ids = []

        for params, worker_id in batch:
            worker_name = self.worker_configs[worker_id]['name']
            task_id = self.task_counter
            self.task_counter += 1

            task = {
                'task_id': task_id,
                'worker_id': worker_id,
                'worker_name': worker_name,
                'dataset': dataset,
                'learning_rate': float(params[0]),
                'batch_size': int(params[1]),
                'num_epochs': int(params[2]),
                'seed': self.seed,
                'run_id': run_id,
                'timestamp': time.time(),
                's3_bucket': self.bucket_name,
                's3_results_prefix': f'results/{run_id}/task_{task_id}/'
            }

            try:
                self.sqs.send_message(
                    QueueUrl=self.queue_url,
                    MessageBody=json.dumps(task, cls=NumpyEncoder),
                    MessageAttributes={
                        'WorkerId': {'DataType': 'String', 'StringValue': str(worker_id)},
                        'TaskId': {'DataType': 'String', 'StringValue': str(task_id)}
                    }
                )
                print(f"  Task {task_id}: lr={params[0]:.6f}, bs={params[1]}, epochs={params[2]} -> {worker_name} worker")
                task_ids.append(task_id)
            except Exception as e:
                print(f"  Failed to dispatch task {task_id}: {e}")

        return task_ids

    def poll_results(self, task_ids, dataset='mnist', run_id=None, timeout=1800, poll_interval=15):
        if run_id is None:
            run_id = f"{dataset}_seed{self.seed}_{int(time.time())}"
        
        results = {}
        start_time = time.time()

        print(f"\nPolling for results from {len(task_ids)} tasks...")
        print(f"  Looking in: results/{run_id}/")
        print("-" * 50)

        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            for task_id in task_ids:
                if task_id in results:
                    continue

                key = f'results/{run_id}/task_{task_id}/result.json'
                try:
                    response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
                    result = json.loads(response['Body'].read().decode('utf-8'))
                    results[task_id] = result
                    print(f"  Task {task_id}: {result['accuracy']:.2f}% in {result['runtime']:.1f}s")
                except self.s3.exceptions.NoSuchKey:
                    continue
                except Exception as e:
                    print(f"  Error reading task {task_id}: {e}")

            if len(results) < len(task_ids):
                time.sleep(poll_interval)

        for task_id in task_ids:
            if task_id not in results:
                key = f'results/{run_id}/task_{task_id}/result.json'
                print(f"  WARNING: Task {task_id} timed out (checked key: {key})")
                results[task_id] = {'task_id': task_id, 'accuracy': 0.0, 'runtime': 0.0, 'error': 'Timeout'}

        return results

    def update_gp(self, results):
        for task_id, result in results.items():
            if 'error' in result:
                continue

            learning_rate = result.get('learning_rate', 0.001)
            batch_size = result.get('batch_size', 64)
            num_epochs = result.get('num_epochs', 2)
            accuracy = result.get('accuracy', 0.0)

            self.X.append([learning_rate, batch_size, num_epochs])
            self.y.append(accuracy)

        if len(self.X) > 1:
            self.gp.fit(np.array(self.X), np.array(self.y))

        best_idx = np.argmax(self.y) if self.y else None
        if best_idx is not None:
            best_accuracy = self.y[best_idx]
            best_params = self.X[best_idx]
        else:
            best_accuracy = 0.0
            best_params = [0.001, 64, 2]

        return best_accuracy, best_params

    def run_optimization(self, n_batches=4, dataset='mnist'):
        run_id = f"{dataset}_seed{self.seed}_{int(time.time())}"
        
        print("\n" + "=" * 70)
        print(f"RUNNING CLOUD-EDGE OPTIMIZATION FOR {dataset.upper()}")
        print("=" * 70)
        print(f"  Run ID: {run_id}")
        print("=" * 70)

        best_accuracy = 0.0
        best_params = [0.001, 64, 2]
        all_results = []

        for batch_idx in range(n_batches):
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print("=" * 70)

            batch = self.propose_batch(n_workers=3)

            print("\nProposed configurations:")
            for i, (params, worker_id) in enumerate(batch):
                worker_name = self.worker_configs[worker_id]['name']
                print(f"  {i+1}. {worker_name}: lr={params[0]:.6f}, bs={params[1]}, epochs={params[2]}")

            task_ids = self.dispatch_batch(batch, dataset, run_id)

            if not task_ids:
                print("No tasks dispatched, continuing...")
                continue

            results = self.poll_results(task_ids, dataset, run_id)

            batch_best, batch_best_params = self.update_gp(results)

            if batch_best > best_accuracy:
                best_accuracy = batch_best
                best_params = batch_best_params
                self.best_X = batch_best_params
                print(f"\n  NEW BEST: {best_accuracy:.2f}% with lr={best_params[0]:.6f}, bs={best_params[1]}, epochs={best_params[2]}")

            all_results.extend(results.values())

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
        print(f"Best: lr={best_params[0]:.6f}, bs={best_params[1]}, epochs={best_params[2]}")

        results_json = {
            'method': 'AWS Cloud-Edge BO',
            'dataset': dataset,
            'seed': self.seed,
            'run_id': run_id,
            'n_batches': n_batches,
            'best_accuracy': float(best_accuracy),
            'best_hyperparameters': {
                'learning_rate': float(best_params[0]),
                'batch_size': int(best_params[1]),
                'num_epochs': int(best_params[2])
            },
            'all_results': all_results
        }

        os.makedirs('results', exist_ok=True)
        filename = f'results/aws_cloud_edge_bo_{dataset}_seed{self.seed}.json'
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to {filename}")

        return best_accuracy, best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AWS Cloud Controller')
    parser.add_argument('--batches', type=int, default=4, help='Number of batches')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--beta', type=float, default=2.0, help='UCB parameter')
    parser.add_argument('--gamma', type=float, default=0.2, help='Heterogeneity compensation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    controller = CloudController(beta=args.beta, gamma=args.gamma, seed=args.seed)
    best_acc, best_params = controller.run_optimization(n_batches=args.batches, dataset=args.dataset)

    print(f"\nFinal Results:")
    print(f"  Best Accuracy: {best_acc:.2f}%")
    print(f"  Best Learning Rate: {best_params[0]:.6f}")
    print(f"  Best Batch Size: {best_params[1]}")
    print(f"  Best Epochs: {best_params[2]}")