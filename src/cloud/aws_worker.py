"""
AWS Edge Worker - With Heartbeat-based SQS Visibility Extension
"""

import boto3
import json
import time
import os
import sys
import traceback
import threading
from datetime import datetime

print("=" * 60)
print("AWS EDGE WORKER - HEARTBEAT VERSION")
print("=" * 60)

try:
    TOKEN = os.popen('curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"').read().strip()
    INSTANCE_ID = os.popen(f'curl -s -H "X-aws-ec2-metadata-token: {TOKEN}" http://169.254.169.254/latest/meta-data/instance-id').read().strip()
    print(f"Instance ID: {INSTANCE_ID}")
except:
    INSTANCE_ID = "unknown"
    print("Could not get instance ID")

try:
    with open('/app/aws_config.json', 'r') as f:
        config = json.load(f)
    BUCKET_NAME = config['bucket_name']
    QUEUE_URL = config['queue_url']
    REGION = config.get('region', 'ap-south-1')
except Exception as e:
    print(f"Error loading config: {e}")
    BUCKET_NAME = os.environ.get('S3_BUCKET', 'cloud-edge-bo-ap-south')
    QUEUE_URL = os.environ.get('SQS_QUEUE_URL', 'https://ap-south-1.queue.amazonaws.com/987604288290/cloud-edge-bo-tasks')
    REGION = os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')

print(f"  Bucket: {BUCKET_NAME}")
print(f"  Queue: {QUEUE_URL}")
print(f"  Region: {REGION}")

s3 = boto3.client('s3', region_name=REGION)
sqs = boto3.client('sqs', region_name=REGION)

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/src/training')
sys.path.insert(0, os.getcwd())

def heartbeat_worker(receipt_handle, queue_url, stop_event, task_id):
    print(f"  [HEARTBEAT] Started for task {task_id}")
    while not stop_event.is_set():
        time.sleep(60)
        try:
            sqs.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=600
            )
            print(f"  [HEARTBEAT] Extended visibility for task {task_id}")
        except Exception as e:
            print(f"  [HEARTBEAT ERROR] {e}")
    print(f"  [HEARTBEAT] Stopped for task {task_id}")

def sync_cifar10_data():
    """Sync CIFAR-10 dataset from S3 to local disk"""
    data_dir = './data/cifar-10-batches-py/'
    if not os.path.exists(data_dir):
        print("  Syncing CIFAR-10 data from S3...")
        os.makedirs('./data', exist_ok=True)
        result = os.system(f'aws s3 sync s3://cloud-edge-bo-ap-south/datasets/cifar-10-batches-py/ ./data/cifar-10-batches-py/ --region ap-south-1')
        if result == 0:
            print("  CIFAR-10 data synced successfully!")
        else:
            print("  WARNING: CIFAR-10 sync may have failed")
    else:
        print("  CIFAR-10 data already exists locally")

def process_task(task, receipt_handle):
    try:
        task_id = task.get('task_id', 'unknown')
        worker_id = task.get('worker_id', 0)
        worker_name = task.get('worker_name', 'unknown')
        learning_rate = task.get('learning_rate', 0.001)
        batch_size = task.get('batch_size', 64)
        num_epochs = task.get('num_epochs', 2)
        dataset = task.get('dataset', 'mnist')
        seed = task.get('seed', 42)
        run_id = task.get('run_id', 'default')

        print(f"\n{'='*50}")
        print(f"TASK {task_id} - {worker_name.upper()}")
        print(f"{'='*50}")
        print(f"  Learning Rate: {learning_rate:.6f}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Dataset: {dataset}")
        print(f"  Seed: {seed}")
        print(f"  Run ID: {run_id}")

        # Sync CIFAR-10 data if needed
        if dataset == 'cifar10':
            sync_cifar10_data()

        # Import training function
        if dataset == 'mnist':
            from train_mnist import train_mnist
            train_fn = train_mnist
        elif dataset == 'fashionmnist':
            from train_fashionmnist import train_fashionmnist
            train_fn = train_fashionmnist
        elif dataset == 'cifar10':
            from train_cifar10 import train_cifar10
            train_fn = train_cifar10
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"  Training function loaded: {train_fn.__name__}")
        
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=heartbeat_worker,
            args=(receipt_handle, QUEUE_URL, stop_heartbeat, task_id)
        )
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        start_time = time.time()
        accuracy, runtime = train_fn(
            lr=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            seed=seed
        )
        total_time = time.time() - start_time

        stop_heartbeat.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=5)

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Runtime: {total_time:.1f}s")

        result = {
            'task_id': task_id,
            'worker_id': worker_id,
            'worker_name': worker_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'dataset': dataset,
            'seed': seed,
            'run_id': run_id,
            'accuracy': accuracy,
            'runtime': total_time,
            'instance_id': INSTANCE_ID,
            'timestamp': datetime.now().isoformat()
        }

        result_key = f'results/{run_id}/task_{task_id}/result.json'
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=result_key,
            Body=json.dumps(result, indent=2)
        )
        print(f"  Result uploaded to: {result_key}")

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        
        try:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=5)
        except:
            pass
        
        try:
            error_result = {
                'task_id': task.get('task_id', 'unknown'),
                'worker_id': task.get('worker_id', 0),
                'error': str(e),
                'instance_id': INSTANCE_ID,
                'timestamp': datetime.now().isoformat()
            }
            run_id = task.get('run_id', 'default')
            result_key = f'results/{run_id}/task_{task.get("task_id", "unknown")}/result.json'
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=result_key,
                Body=json.dumps(error_result, indent=2)
            )
        except:
            pass
        return None

def main_loop():
    print("\n" + "=" * 60)
    print("WORKER LISTENING FOR TASKS (HEARTBEAT ENABLED)")
    print("=" * 60)

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=600
            )

            messages = response.get('Messages', [])

            if not messages:
                print(".", end='', flush=True)
                continue

            for message in messages:
                receipt_handle = message['ReceiptHandle']
                body = json.loads(message['Body'])

                result = process_task(body, receipt_handle)

                if result:
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    print(f"[OK] Task {result['task_id']} completed")
                else:
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    print(f"[FAIL] Task {body.get('task_id', 'unknown')} failed - deleted")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main_loop()