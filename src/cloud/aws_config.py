# src/aws_config.py
import os
import json

# AWS Configuration
AWS_REGION = 'us-east-1'  # Change to ap-south-1 if you want Mumbai
BUCKET_NAME = None  # Will be created dynamically
QUEUE_NAME = 'cloud-edge-bo-tasks'

# Paths
LOCAL_DATA_PATH = './data'
S3_DATASET_PATH = 'datasets'
S3_RESULTS_PATH = 'results'
S3_MODELS_PATH = 'models'

# Worker configurations
WORKER_TYPES = {
    'slow': {'instance_type': 't3.micro', 'cpu': 1, 'ram_gb': 1, 'speed_factor': 0.5},
    'medium': {'instance_type': 't3.small', 'cpu': 1, 'ram_gb': 2, 'speed_factor': 1.0},
    'fast': {'instance_type': 't3.medium', 'cpu': 2, 'ram_gb': 4, 'speed_factor': 2.0}
}

def save_config(config):
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def load_config():
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except:
        return {}