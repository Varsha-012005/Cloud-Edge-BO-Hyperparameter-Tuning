# src/aws_terminate_ec2.py
"""
Terminate all EC2 edge workers
"""
import boto3
import json

print("=" * 60)
print("AWS EC2 TERMINATOR")
print("=" * 60)

# Load config
try:
    with open('aws_config.json', 'r') as f:
        config = json.load(f)
except:
    print(" No config file found!")
    exit(1)

if 'worker_instances' not in config or not config['worker_instances']:
    print(" No worker instances found in config!")
    exit(1)

ec2 = boto3.client('ec2')
instance_ids = config['worker_instances']

print(f"\n  Terminating {len(instance_ids)} instances...")
print("-" * 40)

for inst_id in instance_ids:
    print(f"  Terminating: {inst_id}")

try:
    ec2.terminate_instances(InstanceIds=instance_ids)
    print(f"\n Termination request sent!")
    print("   Instances will be terminated shortly")
    
    # Remove from config
    config['worker_instances'] = []
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
except Exception as e:
    print(f" Error: {e}")