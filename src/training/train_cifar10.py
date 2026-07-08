import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mnist import SimpleCNN, count_parameters


def train_cifar10(lr=0.001, batch_size=64, num_epochs=2, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    # Light augmentation for CPU speed
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print(f"  Loading CIFAR-10 dataset...")
    
    # Check if data already exists locally
    data_path = './data'
    if os.path.exists('./data/cifar-10-batches-py/'):
        print("  Using pre-downloaded CIFAR-10 data from ./data/")
        # Force torchvision to use existing data without downloading
        train_data = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
        val_data = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
    else:
        print("  CIFAR-10 data not found locally, attempting download...")
        try:
            train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
            val_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Please ensure data is synced from S3")
            raise
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    model = SimpleCNN(input_channels=3, num_classes=10).to(device)
    param_count = count_parameters(model)
    print(f"  Model parameters: {param_count:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = output.max(1)
            train_total += target.size(0)
            train_correct += pred.eq(target).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, pred = output.max(1)
                val_total += target.size(0)
                val_correct += pred.eq(target).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    elapsed_time = time.time() - start_time
    
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Total Training Time: {elapsed_time:.1f}s")
    
    return best_val_acc, elapsed_time


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CIFAR-10 LIGHT TRAINING")
    print("=" * 60)
    acc, runtime = train_cifar10(lr=0.001, batch_size=64, num_epochs=1)
    print(f"\nTest complete: {acc:.2f}% in {runtime:.1f}s")