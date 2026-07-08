# Run All Experiments on AWS

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RUNNING EXPERIMENTS ON AWS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$datasets = @("mnist", "fashionmnist", "cifar10")

foreach ($dataset in $datasets) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "RUNNING EXPERIMENT ON $dataset".ToUpper() -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    python src/cloud/aws_cloud_controller.py --dataset $dataset --batches 4
    
    Write-Host "`nWaiting for results..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check S3 results
Write-Host "`nResults in S3:" -ForegroundColor Yellow
aws s3 ls s3://cloud-edge-bo-20260704175620/results/ --recursive

# Download results
Write-Host "`nDownloading results..." -ForegroundColor Yellow
aws s3 sync s3://cloud-edge-bo-20260704175620/results/ results/ --recursive

# Show results
Write-Host "`nResults Summary:" -ForegroundColor Yellow
python -c "
import json

datasets = ['mnist', 'fashionmnist', 'cifar10']
print('='*60)
print('RESULTS SUMMARY')
print('='*60)
print(f'{"Dataset":<15} {"Accuracy":<12} {"Expected":<12}')
print('-'*60)

expected = {'mnist': 99.58, 'fashionmnist': 92.79, 'cifar10': 86.73}

for dataset in datasets:
    try:
        with open(f'results/cloud_edge_bo_fixed_results.json', 'r') as f:
            data = json.load(f)
            acc = data.get('best_accuracy', 0)
            exp = expected.get(dataset, 0)
            print(f'{dataset:<15} {acc:<12.2f}% {exp:<12.2f}%')
    except:
        print(f'{dataset:<15} {"N/A":<12} {expected.get(dataset, 0):<12.2f}%')
"
