# src/aws_setup_all.py
"""
Complete AWS Setup for Cloud-Edge Research
Run this to set up everything
"""
import subprocess
import time

print("=" * 60)
print("COMPLETE AWS SETUP FOR CLOUD-EDGE RESEARCH")
print("=" * 60)
print("\nThis script will:")
print("  1. Create S3 bucket")
print("  2. Create SQS queue")
print("  3. Launch EC2 workers")
print("\n  Make sure you have:")
print("  - AWS credentials configured (aws configure)")
print("  - Valid AWS account with credits")
print("\n" + "=" * 60)

input("\nPress Enter to continue...")

# Step 1: Create S3 bucket
print("\n" + "=" * 60)
print("STEP 1: Creating S3 Bucket")
print("=" * 60)
subprocess.run(["python", "src/aws_setup_s3.py"])
time.sleep(2)

# Step 2: Create SQS queue
print("\n" + "=" * 60)
print("STEP 2: Creating SQS Queue")
print("=" * 60)
subprocess.run(["python", "src/aws_setup_sqs.py"])
time.sleep(2)

# Step 3: Launch EC2 workers
print("\n" + "=" * 60)
print("STEP 3: Launching EC2 Workers")
print("=" * 60)
print("\n  This will incur AWS charges (~$0.50 per hour)")
response = input("Continue? (y/n): ")
if response.lower() == 'y':
    subprocess.run(["python", "src/aws_launch_ec2.py"])
else:
    print("Skipped EC2 launch")

print("\n" + "=" * 60)
print(" AWS SETUP COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  1. Check AWS Console for running instances")
print("  2. Run cloud controller: python src/cloud/aws_cloud_controller_simple.py")
print("  3. Clean up: python src/cloud/aws_terminate_ec2.py")