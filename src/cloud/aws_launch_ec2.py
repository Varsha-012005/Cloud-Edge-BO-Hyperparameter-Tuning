# src/aws_launch_ec2_correct.py
import boto3
import json
import time

print("=" * 60)
print("AWS EC2 LAUNCHER FOR EDGE WORKERS (CORRECT AMI)")
print("=" * 60)

# Load config
try:
    with open('aws_config.json', 'r') as f:
        config = json.load(f)
        print(f"\n Config loaded")
        print(f"   Bucket: {config['bucket_name']}")
except:
    print(" Run aws_setup_s3.py first!")
    exit(1)

ec2 = boto3.client('ec2')

# CORRECT AMI ID for Amazon Linux 2023 in us-east-1
AMI_ID = 'ami-046b36f55e189564a'  # Amazon Linux 2023 kernel-6.18
print(f"\n Using AMI: {AMI_ID}")

# Worker configurations
workers = [
    {'name': 'slow', 'instance_type': 't3.micro', 'cpu': 1, 'ram_gb': 1},
    {'name': 'medium', 'instance_type': 't3.small', 'cpu': 1, 'ram_gb': 2},
    {'name': 'fast', 'instance_type': 't3.medium', 'cpu': 2, 'ram_gb': 4}
]

print("\n Launching edge workers...")
print("-" * 40)

instance_ids = []

for worker in workers:
    print(f"\n  Launching {worker['name']} worker ({worker['instance_type']})...")
    
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
                    {'Key': 'Project', 'Value': 'Cloud-Edge-BO'},
                    {'Key': 'Researcher', 'Value': 'Varsha'}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        instance_ids.append(instance_id)
        print(f"     Launched: {instance_id}")
        print(f"     Type: {worker['instance_type']}")
        time.sleep(5)
        
    except Exception as e:
        print(f"     Error: {e}")

# Save instance IDs
config['worker_instances'] = instance_ids
config['worker_types'] = workers
config['ami_id'] = AMI_ID

with open('aws_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n" + "=" * 60)
print(f" {len(instance_ids)} edge workers launched!")
print("=" * 60)

if instance_ids:
    print("\n Instance IDs:")
    for i, inst_id in enumerate(instance_ids):
        print(f"   {i+1}. {inst_id}")
    
    print("\n To check status:")
    for inst_id in instance_ids:
        print(f"   aws ec2 describe-instances --instance-ids {inst_id}")
    
    print("\n To terminate when done:")
    terminate_cmd = "aws ec2 terminate-instances --instance-ids " + " ".join(instance_ids)
    print(f"   {terminate_cmd}")
else:
    print("\n No instances launched. Check your AWS permissions.")