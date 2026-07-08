"""
Launch EC2 Workers with Key Pair 
"""

import boto3
import json
import time

def launch_ec2_paper():
    with open('aws_config.json', 'r') as f:
        config = json.load(f)

    bucket_name = config['bucket_name']
    region = config.get('region', 'ap-south-1')

    ec2 = boto3.client('ec2', region_name=region)

    ami_response = ec2.describe_images(
        Owners=['amazon'],
        Filters=[
            {'Name': 'name', 'Values': ['al2023-ami-*-x86_64']},
            {'Name': 'state', 'Values': ['available']}
        ]
    )
    AMI_ID = sorted(ami_response['Images'], key=lambda x: x['CreationDate'])[-1]['ImageId']
    print(f"Using AMI: {AMI_ID} (Amazon Linux 2023)")

    user_data_template = """#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

BUCKET={bucket_name}
WORKER_NAME={worker_name}
REGION=ap-south-1

echo "=========================================="
echo "CLOUD-EDGE WORKER - AUTO START ($WORKER_NAME)"
echo "=========================================="

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
echo "Instance ID: $INSTANCE_ID"

( while true; do
    aws s3 cp /var/log/user-data.log "s3://$BUCKET/logs/${{INSTANCE_ID}}_userdata.log" --region $REGION 2>/dev/null
    if [ -f /app/worker.log ]; then
        aws s3 cp /app/worker.log "s3://$BUCKET/logs/${{INSTANCE_ID}}_worker.log" --region $REGION 2>/dev/null
    fi
    sleep 15
  done ) &

if [ ! -f /swapfile ]; then
    dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

dnf update -y
dnf install -y python3 python3-pip git awscli

mkdir -p /app
cd /app

echo "Downloading scripts from s3://$BUCKET/scripts/"
for i in $(seq 1 5); do
    aws s3 sync "s3://$BUCKET/scripts/" /app/ --region $REGION
    if [ -f /app/requirements.txt ]; then
        echo "Files downloaded successfully"
        break
    fi
    echo "Attempt $i failed, retrying in 10 seconds..."
    sleep 10
done

if [ ! -f /app/requirements.txt ]; then
    echo "ERROR: Failed to download scripts from S3"
    exit 1
fi

ls -la /app/

python3 -m venv /app/venv
/app/venv/bin/pip install --upgrade pip
/app/venv/bin/pip install -r requirements.txt

nohup /app/venv/bin/python3 aws_worker.py > /app/worker.log 2>&1 &
WORKER_PID=$!
echo "Started aws_worker.py with PID $WORKER_PID"

sleep 15
if ps -p $WORKER_PID > /dev/null; then
    echo "READY" > /tmp/ready.txt
    aws s3 cp /tmp/ready.txt "s3://$BUCKET/workers/${{INSTANCE_ID}}_ready.txt" --region $REGION
    echo "Worker confirmed running, ready marker uploaded"
else
    echo "WORKER FAILED TO START - check worker.log"
    cat /app/worker.log
    exit 1
fi
"""

    workers = [
        {'name': 'slow', 'instance_type': 't3.micro'},
        {'name': 'medium', 'instance_type': 't3.small'},
        {'name': 'fast', 'instance_type': 't3.medium'}
    ]

    print("=" * 70)
    print("LAUNCHING HETEROGENEOUS EC2 WORKERS")
    print("=" * 70)
    print("  slow: t3.micro  (0.5x speed)")
    print("  medium: t3.small (1.0x speed)")
    print("  fast: t3.medium (2.0x speed)")
    print("=" * 70)

    instance_ids = []
    for worker in workers:
        user_data = user_data_template.format(bucket_name=bucket_name, worker_name=worker['name'])
        print(f"\nLaunching {worker['name']} ({worker['instance_type']})...")
        try:
            response = ec2.run_instances(
                ImageId=AMI_ID,
                InstanceType=worker['instance_type'],
                MinCount=1,
                MaxCount=1,
                BlockDeviceMappings=[{
                    'DeviceName': '/dev/xvda',
                    'Ebs': {'VolumeSize': 30, 'VolumeType': 'gp3'}
                }],
                KeyName='cloud-edge-key',
                UserData=user_data,
                IamInstanceProfile={'Name': 'EC2-S3-SQS-Profile'},
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
            print(f"  Launched: {instance_id}")
        except Exception as e:
            print(f"  Error: {e}")

    config['worker_instances'] = instance_ids
    with open('aws_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"LAUNCHED {len(instance_ids)} WORKERS")
    print("=" * 70)

    print("\nWaiting for ready markers (checking every 20s, up to 8 minutes)...")
    s3 = boto3.client('s3', region_name=region)
    deadline = time.time() + 8 * 60
    ready = set()
    while time.time() < deadline and len(ready) < len(instance_ids):
        for inst_id in instance_ids:
            if inst_id in ready:
                continue
            try:
                s3.head_object(Bucket=bucket_name, Key=f'workers/{inst_id}_ready.txt')
                ready.add(inst_id)
                print(f"  {inst_id}: READY")
            except Exception:
                pass
        if len(ready) < len(instance_ids):
            time.sleep(20)

    not_ready = [i for i in instance_ids if i not in ready]
    if not_ready:
        print(f"\nWARNING: instances not ready: {not_ready}")
        print("Check logs with:")
        for i in not_ready:
            print(f"  aws s3 cp s3://{bucket_name}/logs/{i}_userdata.log - --region {region}")
            print(f"  aws s3 cp s3://{bucket_name}/logs/{i}_worker.log - --region {region}")
    else:
        print("\nAll workers ready!")

    return instance_ids

if __name__ == "__main__":
    launch_ec2_paper()