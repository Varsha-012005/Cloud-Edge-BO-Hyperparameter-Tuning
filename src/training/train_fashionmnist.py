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


def train_fashionmnist(lr=0.001, batch_size=64, num_epochs=2, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_train)
    val_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    param_count = count_parameters(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
    
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
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}, Params={param_count:,}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/best_fashionmnist_model.pth')
    
    elapsed_time = time.time() - start_time
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Total Training Time: {elapsed_time:.1f}s")
    
    return best_val_acc, elapsed_time


if __name__ == "__main__":
    acc, runtime = train_fashionmnist(lr=0.001, batch_size=64, num_epochs=2)
    print(f"\nTest complete: {acc:.2f}% in {runtime:.1f}s")
