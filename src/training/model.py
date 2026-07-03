# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        
        # Dynamic FC layers based on input type
        if input_channels == 1:
            # MNIST: after 2 poolings, 28x28 -> 7x7
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
        else:
            # CIFAR-10: after 2 poolings, 32x32 -> 8x8
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        if hasattr(self, 'fc3'):
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing MNIST Model")
    print("=" * 50)
    model_mnist = SimpleCNN(input_channels=1, num_classes=10)
    print(f"Parameters: {count_parameters(model_mnist):,}")
    dummy = torch.randn(1, 1, 28, 28)
    print(f"Output shape: {model_mnist(dummy).shape}")
    
    print("\n" + "=" * 50)
    print("Testing CIFAR-10 Model")
    print("=" * 50)
    model_cifar = SimpleCNN(input_channels=3, num_classes=10)
    print(f"Parameters: {count_parameters(model_cifar):,}")
    dummy = torch.randn(1, 3, 32, 32)
    print(f"Output shape: {model_cifar(dummy).shape}")
    print("\n Model verification complete!")