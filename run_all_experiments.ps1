# Complete Experiment Runner - Local + AWS

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RUNNING ALL EXPERIMENTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$seeds = @(42, 123, 456, 789, 1010)
$datasets = @("mnist", "fashionmnist", "cifar10")

# ============================================
# PART 1: BASELINE EXPERIMENTS (LOCAL)
# ============================================
Write-Host "`n[PART 1] RUNNING BASELINE EXPERIMENTS (LOCAL)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

foreach ($dataset in $datasets) {
    Write-Host "`n--- $dataset".ToUpper() + " ---" -ForegroundColor Green
    
    foreach ($seed in $seeds) {
        Write-Host "  Seed: $seed" -ForegroundColor Gray
        
        # Random Search
        python src/core/random_search.py --dataset $dataset --trials 12 --seed $seed
        
        # Grid Search
        python src/core/grid_search.py --dataset $dataset --seed $seed
        
        # Sequential BO
        python src/core/sequential_bo.py --dataset $dataset --trials 12 --seed $seed
    }
}

# ============================================
# PART 2: AWS SETUP (ONCE)
# ============================================
Write-Host "`n[PART 2] AWS SETUP" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

Write-Host "Checking AWS credentials..." -ForegroundColor Gray
aws sts get-caller-identity

# Upload code to S3
Write-Host "Uploading code to S3..." -ForegroundColor Gray
aws s3 cp src/training/train_mnist.py s3://cloud-edge-bo-20260704175620/scripts/
aws s3 cp src/training/train_fashionmnist.py s3://cloud-edge-bo-20260704175620/scripts/
aws s3 cp src/training/train_cifar10.py s3://cloud-edge-bo-20260704175620/scripts/
aws s3 cp src/cloud/aws_worker.py s3://cloud-edge-bo-20260704175620/scripts/
aws s3 cp requirements.txt s3://cloud-edge-bo-20260704175620/scripts/
aws s3 cp aws_config.json s3://cloud-edge-bo-20260704175620/scripts/

# Launch EC2 workers
Write-Host "Launching EC2 workers..." -ForegroundColor Gray
python src/cloud/launch_ec2_paper.py

Write-Host "Waiting 3 minutes for workers to setup..." -ForegroundColor Gray
Start-Sleep -Seconds 180

# ============================================
# PART 3: CLOUD-EDGE BO (AWS)
# ============================================
Write-Host "`n[PART 3] RUNNING CLOUD-EDGE BO (AWS)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

foreach ($dataset in $datasets) {
    Write-Host "`n--- $dataset".ToUpper() + " ---" -ForegroundColor Green
    
    foreach ($seed in $seeds) {
        Write-Host "  Seed: $seed" -ForegroundColor Gray
        python src/core/cloud_edge_bo.py --dataset $dataset --batches 4 --beta 2.0 --gamma 0.2 --seed $seed
    }
}

# ============================================
# PART 4: ABLATION STUDIES (AWS)
# ============================================
Write-Host "`n[PART 4] RUNNING ABLATION STUDIES (AWS)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

# gamma=0 (no heterogeneity compensation)
foreach ($dataset in $datasets) {
    Write-Host "`n--- $dataset".ToUpper() + " (gamma=0)" -ForegroundColor Green
    foreach ($seed in $seeds) {
        python src/core/cloud_edge_bo.py --dataset $dataset --batches 4 --beta 2.0 --gamma 0.0 --seed $seed
    }
}

# ============================================
# PART 5: STATISTICS AND GRAPHS (LOCAL)
# ============================================
Write-Host "`n[PART 5] GENERATING STATISTICS AND GRAPHS (LOCAL)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

Write-Host "Computing statistics..." -ForegroundColor Gray
python src/analysis/compute_stats.py

Write-Host "Generating graphs..." -ForegroundColor Gray
python src/analysis/generate_graphs.py

# ============================================
# PART 6: CLEAN UP AWS
# ============================================
Write-Host "`n[PART 6] CLEANING UP AWS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

$response = Read-Host "Terminate EC2 workers? (y/n)"
if ($response -eq 'y') {
    python src/cloud/aws_terminate_ec2.py
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Show results
python -c "
import json
import glob

print('='*60)
print('RESULTS SUMMARY')
print('='*60)
print(f'{"Dataset":<15} {"Method":<20} {"Accuracy":<12} {"Std":<10}')
print('-'*60)

datasets = ['mnist', 'fashionmnist', 'cifar10']
methods = ['random_search', 'grid_search', 'sequential_bo', 'cloud_edge_bo']
method_labels = {'random_search':'Random Search', 'grid_search':'Grid Search', 'sequential_bo':'Sequential BO', 'cloud_edge_bo':'Cloud-Edge BO'}

for dataset in datasets:
    for method in methods:
        files = glob.glob(f'results/{method}_{dataset}_seed*.json')
        accs = []
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    accs.append(data['best_accuracy'])
            except:
                pass
        if accs:
            print(f'{dataset:<15} {method_labels[method]:<20} {np.mean(accs):<12.2f}% {np.std(accs):<10.3f}')
"
