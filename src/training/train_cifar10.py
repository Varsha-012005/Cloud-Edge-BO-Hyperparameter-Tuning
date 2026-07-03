import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleCNN

class ImprovedCIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_cifar10(lr=0.001, batch_size=128, num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training CIFAR-10 for {num_epochs} epochs...")
    print("-" * 50)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR-10 dataset...")
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    val_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_val)
    
    train_loader = DataLoader(train_data, batch_size=int(batch_size), shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print("-" * 50)
    
    model = ImprovedCIFAR10Net(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_correct, train_total = 0, 0
        train_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'results/best_cifar10_improved.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("CIFAR-10 TRAINING COMPLETE!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"Time per epoch: {elapsed/num_epochs:.1f} seconds")
    print("="*50)
    
    return best_acc, elapsed

if __name__ == "__main__":
    acc, t = train_cifar10(lr=0.001, batch_size=128, num_epochs=30)
    print(f"\nFinal Result: {acc:.2f}%")
