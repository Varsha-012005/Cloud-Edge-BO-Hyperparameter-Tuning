# COMPLETE EXPERIMENT RUN

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "COMPLETE EXPERIMENT RUN" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Reset
Write-Host "`n[1/5] Resetting..." -ForegroundColor Yellow
python reset_run.py

# 2. Launch workers if needed
Write-Host "`n[2/5] Launching workers..." -ForegroundColor Yellow
python src/cloud/launch_ec2_paper.py

# 3. Wait for workers
Write-Host "`n[3/5] Waiting 2 minutes..." -ForegroundColor Yellow
Start-Sleep -Seconds 120

# 4. Run experiments
Write-Host "`n[4/5] Running experiments..." -ForegroundColor Yellow
$seeds = @(42, 123, 456)
$datasets = @("mnist", "fashionmnist", "cifar10")

foreach ($dataset in $datasets) {
    Write-Host "`n--- $dataset ---" -ForegroundColor Green
    foreach ($seed in $seeds) {
        Write-Host "  Seed: $seed" -ForegroundColor Gray
        python src/cloud/aws_cloud_controller.py --dataset $dataset --batches 4 --seed $seed
    }
}

# 5. Collect results
Write-Host "`n[5/5] Collecting results..." -ForegroundColor Yellow
aws s3 sync s3://cloud-edge-bo-ap-south/results/ results/ --region ap-south-1

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
