# RUN CLEAN EXPERIMENTS

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RUN CLEAN EXPERIMENTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$seeds = @(42, 123, 456)
$datasets = @("mnist", "fashionmnist", "cifar10")

foreach ($dataset in $datasets) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "RUNNING: $dataset" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    foreach ($seed in $seeds) {
        Write-Host "`n--- Seed: $seed ---" -ForegroundColor Yellow
        python src/cloud/aws_cloud_controller.py --dataset $dataset --batches 4 --seed $seed
    }
}

# Collect results
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "COLLECTING RESULTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

aws s3 sync s3://cloud-edge-bo-ap-south/results/ results/ --region ap-south-1 --recursive

# Show summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "RESULTS SUMMARY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

python -c "
import json
import glob

print(f'{"Dataset":<15} {"Seed 42":<12} {"Seed 123":<12} {"Seed 456":<12} {"Mean":<12}')
print('-'*60)

datasets = ['mnist', 'fashionmnist', 'cifar10']
seeds = [42, 123, 456]

for dataset in datasets:
    accs = []
    for seed in seeds:
        f = f'results/aws_cloud_edge_bo_{dataset}_seed{seed}.json'
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                acc = data['best_accuracy']
                accs.append(acc)
        except:
            pass
    if accs:
        mean = sum(accs) / len(accs)
        print(f'{dataset.upper():<15} {accs[0]:<12.2f}% {accs[1]:<12.2f}% {accs[2]:<12.2f}% {mean:.2f}%')
    else:
        print(f'{dataset.upper():<15} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12}')
"
