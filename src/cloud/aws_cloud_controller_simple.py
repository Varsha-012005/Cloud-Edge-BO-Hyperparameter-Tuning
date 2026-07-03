import boto3
import json
import time
import numpy as np

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        return super().default(obj)

print("=" * 60)
print("  AWS CLOUD CONTROLLER (FIXED)")
print("Cloud-Edge Bayesian Optimization")
print("=" * 60)

# Load config
try:
    with open('aws_config.json', 'r') as f:
        config = json.load(f)
    print(f"\n Config loaded")
    print(f"   Bucket: {config['bucket_name']}")
    print(f"   Queue: {config.get('queue_name', 'cloud-edge-bo-tasks')}")
except Exception as e:
    print(f" Error loading config: {e}")
    exit(1)

# Initialize AWS clients
s3 = boto3.client('s3')
sqs = boto3.client('sqs')
queue_url = config['queue_url']
bucket_name = config['bucket_name']

class CloudController:
    def __init__(self):
        self.X = []
        self.y = []
        self.task_counter = 0
        
    def propose_hyperparameters(self):
        """Propose hyperparameters"""
        lr = float(10 ** np.random.uniform(-5, -2))  # Convert to Python float
        bs = int(np.random.choice([32, 64, 128]))     # Convert to Python int
        return lr, bs
    
    def dispatch_task(self, learning_rate, batch_size):
        """Send task to SQS"""
        task = {
            'task_id': int(self.task_counter),
            'lr': float(learning_rate),
            'batch_size': int(batch_size),
            'timestamp': float(time.time())
        }
        
        try:
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(task, cls=NumpyEncoder)
            )
            print(f"   Task {self.task_counter}: lr={learning_rate:.6f}, bs={batch_size}")
            self.task_counter += 1
            return True
        except Exception as e:
            print(f"   Error dispatching task: {e}")
            return False
        
    def run(self, n_tasks=12):
        """Run optimization"""
        
        print(f"\n Running {n_tasks} tasks on workers...")
        print("-" * 40)
        
        success_count = 0
        for i in range(n_tasks):
            lr, bs = self.propose_hyperparameters()
            if self.dispatch_task(lr, bs):
                success_count += 1
            time.sleep(0.5)  # Small delay between dispatches
        
        print(f"\n Dispatched {success_count}/{n_tasks} tasks successfully!")
        print("\n Check results in S3 bucket:")
        print(f"   s3://{bucket_name}/results/")
        
        # Save summary
        summary = {
            'timestamp': time.time(),
            'total_tasks': n_tasks,
            'successful_tasks': success_count,
            'bucket': bucket_name,
            'queue_url': queue_url
        }
        
        try:
            s3.put_object(
                Bucket=bucket_name,
                Key='results/controller_summary.json',
                Body=json.dumps(summary, indent=2, cls=NumpyEncoder)
            )
            print("\n Controller summary saved to S3")
        except Exception as e:
            print(f"\n Could not save summary: {e}")

if __name__ == "__main__":
    controller = CloudController()
    controller.run(n_tasks=12)