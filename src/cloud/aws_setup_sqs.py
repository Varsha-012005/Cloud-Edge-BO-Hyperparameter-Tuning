# src/aws_setup_sqs.py
"""
Create SQS queue for task distribution
"""
import boto3
import json

print("=" * 60)
print("AWS SQS SETUP FOR CLOUD-EDGE RESEARCH")
print("=" * 60)

# Load config
try:
    with open('aws_config.json', 'r') as f:
        config = json.load(f)
except:
    print("❌ Run aws_setup_s3.py first!")
    exit(1)

sqs = boto3.client('sqs')
queue_name = 'cloud-edge-bo-tasks'

print(f"\n📨 Creating SQS queue: {queue_name}")

try:
    response = sqs.create_queue(
        QueueName=queue_name,
        Attributes={
            'VisibilityTimeout': '600',  # 10 minutes
            'MessageRetentionPeriod': '86400'  # 1 day
        }
    )
    
    queue_url = response['QueueUrl']
    print(f"  ✅ Queue created")
    print(f"  📍 URL: {queue_url}")
    
    # Save queue URL
    config['queue_url'] = queue_url
    config['queue_name'] = queue_name
    
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration updated")
    
except Exception as e:
    print(f"❌ Error: {e}")