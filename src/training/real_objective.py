"""
Real CNN training objective for Bayesian Optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleCNN

def train_and_evaluate(learning_rate, batch_size, num_epochs=2, dataset='mnist'):
    """
    Actually train a CNN and return validation accuracy
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Load dataset
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        input_channels = 1
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        input_channels = 3
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleCNN(input_channels=input_channels, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"    Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    return val_acc, elapsed_time

def objective_for_bo(learning_rate, batch_size):
    """
    Wrapper for Bayesian Optimization
    Returns negative accuracy because optimizers minimize
    """
    print(f"\n  Training with: lr={learning_rate:.6f}, batch_size={int(batch_size)}")
    
    # Actually train the model
    accuracy, runtime = train_and_evaluate(
        learning_rate=learning_rate,
        batch_size=int(batch_size),
        num_epochs=2,
        dataset='mnist'
    )
    
    print(f"  Validation Accuracy: {accuracy:.2f}%, Time: {runtime:.1f}s")
    
    # Return negative because BO minimizes
    return -accuracy

if __name__ == "__main__":
    # Test the function
    print("Testing real training function...")
    acc, runtime = train_and_evaluate(0.001, 64, num_epochs=1)
    print(f"\nTest complete! Accuracy: {acc:.2f}%, Time: {runtime:.1f}s")
