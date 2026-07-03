# src/aws_setup_s3.py
import boto3
import json
from datetime import datetime
import os

print("=" * 60)
print("AWS S3 SETUP FOR CLOUD-EDGE RESEARCH")
print("=" * 60)

# Initialize S3
s3 = boto3.client('s3')

# Create unique bucket name
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
bucket_name = f'cloud-edge-bo-{timestamp}'

print(f"\n📦 Creating bucket: {bucket_name}")

try:
    # Create bucket
    s3.create_bucket(Bucket=bucket_name)
    print(f"  ✅ Bucket created")
    
    # Create folder structure
    folders = [
        'datasets/mnist/',
        'datasets/fashionmnist/',
        'datasets/cifar10/',
        'results/',
        'models/',
        'logs/',
        'scripts/'
    ]
    
    for folder in folders:
        s3.put_object(Bucket=bucket_name, Key=folder)
        print(f"  📁 Created: {folder}")
    
    # Save configuration
    config = {
        'bucket_name': bucket_name,
        'region': 'us-east-1',
        'created_at': timestamp
    }
    
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to aws_config.json")
    print(f"\n📋 BUCKET NAME: {bucket_name}")
    print(f"   Save this for later use!")
    
except Exception as e:
    print(f"❌ Error: {e}")