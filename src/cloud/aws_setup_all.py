"""
Complete AWS Setup for Cloud-Edge Research
Run this to set up everything for the cloud-edge BO experiment
"""

import subprocess
import time
import json
import os
import boto3
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)

print("=" * 70)
print("COMPLETE AWS SETUP FOR CLOUD-EDGE RESEARCH")
print("=" * 70)
print("\nThis script will:")
print("  1. Create S3 bucket")
print("  2. Create SQS queue")
print("  3. Launch EC2 workers (optional)")
print("\n  Make sure you have:")
print("  - AWS credentials configured (aws configure)")
print("  - Valid AWS account with credits")
print("\n" + "=" * 70)

response = input("\nContinue? (y/n): ")
if response.lower() != 'y':
    print("Setup cancelled")
    exit()

# Step 1: Create S3 bucket
print("\n" + "=" * 70)
print("STEP 1: Creating S3 Bucket")
print("=" * 70)

try:
    # Import and run S3 setup
    from src.cloud.aws_setup_s3 import setup_s3
    setup_s3()
except ImportError:
    # Direct S3 creation if import fails
    print("Direct S3 creation...")
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bucket_name = f'cloud-edge-bo-{timestamp}'
    
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"   Bucket created: {bucket_name}")
        
        # Create folder structure
        folders = ['datasets/', 'results/', 'models/', 'logs/']
        for folder in folders:
            s3.put_object(Bucket=bucket_name, Key=folder)
            print(f"   Created: {folder}")
        
        # Save config
        config = {
            'bucket_name': bucket_name,
            'region': 'us-east-1',
            'created_at': timestamp
        }
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   Configuration saved to aws_config.json")
    except Exception as e:
        print(f"   Error creating bucket: {e}")

time.sleep(2)

# Step 2: Create SQS queue
print("\n" + "=" * 70)
print("STEP 2: Creating SQS Queue")
print("=" * 70)

try:
    from src.cloud.aws_setup_sqs import setup_sqs
    setup_sqs()
except:
    print("Direct SQS creation...")
    sqs = boto3.client('sqs')
    queue_name = 'cloud-edge-bo-tasks'
    
    try:
        # Load existing config
        with open('aws_config.json', 'r') as f:
            config = json.load(f)
        
        response = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'VisibilityTimeout': '600',
                'MessageRetentionPeriod': '86400'
            }
        )
        
        queue_url = response['QueueUrl']
        config['queue_url'] = queue_url
        config['queue_name'] = queue_name
        
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   Queue created: {queue_url}")
    except Exception as e:
        print(f"   Error creating queue: {e}")

time.sleep(2)

# Step 3: Launch EC2 workers (optional)
print("\n" + "=" * 70)
print("STEP 3: Launching EC2 Workers")
print("=" * 70)
print("\n  This will incur AWS charges (~$0.50 per hour)")

response = input("Launch EC2 workers? (y/n): ")
if response.lower() == 'y':
    try:
        from src.cloud.aws_launch_ec2 import launch_workers
        launch_workers()
    except:
        print("Direct EC2 launch...")
        ec2 = boto3.client('ec2')
        AMI_ID = 'ami-046b36f55e189564a'
        
        workers = [
            {'name': 'slow', 'instance_type': 't3.micro'},
            {'name': 'medium', 'instance_type': 't3.small'},
            {'name': 'fast', 'instance_type': 't3.medium'}
        ]
        
        instance_ids = []
        for worker in workers:
            try:
                response = ec2.run_instances(
                    ImageId=AMI_ID,
                    InstanceType=worker['instance_type'],
                    MinCount=1,
                    MaxCount=1,
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'edge-worker-{worker["name"]}'},
                            {'Key': 'Project', 'Value': 'Cloud-Edge-BO'}
                        ]
                    }]
                )
                instance_id = response['Instances'][0]['InstanceId']
                instance_ids.append(instance_id)
                print(f"   Launched {worker['name']}: {instance_id}")
            except Exception as e:
                print(f"   Error launching {worker['name']}: {e}")
        
        # Update config
        with open('aws_config.json', 'r') as f:
            config = json.load(f)
        config['worker_instances'] = instance_ids
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
else:
    print("Skipped EC2 launch")

print("\n" + "=" * 70)
print("AWS SETUP COMPLETE!")
print("=" * 70)
print("\nConfiguration saved to aws_config.json")
print("\nNext steps:")
print("  1. Check AWS Console for resources")
print("  2. Run cloud-edge BO: python run_aws_experiment.py")
print("  3. Clean up: python src/cloud/aws_terminate_ec2.py")
