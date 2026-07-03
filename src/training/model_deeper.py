# src/model_deeper.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperCNN(nn.Module):
    """
    Deeper CNN for CIFAR-10 to compete with literature benchmarks
    Designed to achieve 75-80% accuracy on CIFAR-10
    """
    def __init__(self, num_classes=10):
        super(DeeperCNN, self).__init__()
        
        # ============================================================
        # Convolutional Layers with Batch Normalization
        # ============================================================
        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2: 64 -> 128
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3: 128 -> 256
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4: 256 -> 512
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # ============================================================
        # Fully Connected Layers
        # After 4 pooling layers: 32 -> 16 -> 8 -> 4 -> 2
        # Flattened size = 512 * 2 * 2 = 2048
        # ============================================================
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # ============================================================
        # Optional: Add dropout layers between FC layers
        # ============================================================
        self.fc_dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # ============================================================
        # Convolutional blocks with BatchNorm and ReLU
        # ============================================================
        # Block 1: Input 32x32x3 -> 32x32x64 -> Pool -> 16x16x64
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: 16x16x64 -> 16x16x128 -> Pool -> 8x8x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: 8x8x128 -> 8x8x256 -> Pool -> 4x4x256
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4: 4x4x256 -> 4x4x512 -> Pool -> 2x2x512
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # ============================================================
        # Flatten for fully connected layers
        # ============================================================
        x = x.view(-1, 512 * 2 * 2)  # 512 * 4 = 2048
        
        # ============================================================
        # Fully connected layers with dropout
        # ============================================================
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc_dropout(x)
        
        x = self.fc3(x)
        
        return x


# Optional: Create a smaller version for faster training
class DeeperCNNLight(nn.Module):
    """
    Lighter version of DeeperCNN for faster training
    """
    def __init__(self, num_classes=10):
        super(DeeperCNNLight, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # After 4 pools: 32 -> 2, size = 256 * 2 * 2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Deeper CNN Models for CIFAR-10")
    print("=" * 60)
    
    # Test DeeperCNN
    print("\n1. DeeperCNN (Full Version)")
    print("-" * 40)
    model_full = DeeperCNN(num_classes=10)
    params_full = count_parameters(model_full)
    print(f"Parameters: {params_full:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output_full = model_full(dummy_input)
    print(f"Output shape: {output_full.shape}")
    
    # Test DeeperCNNLight
    print("\n2. DeeperCNNLight (Light Version)")
    print("-" * 40)
    model_light = DeeperCNNLight(num_classes=10)
    params_light = count_parameters(model_light)
    print(f"Parameters: {params_light:,}")
    
    dummy_input = torch.randn(1, 3, 32, 32)
    output_light = model_light(dummy_input)
    print(f"Output shape: {output_light.shape}")
    
    print("\n" + "=" * 60)
    print(" Model verification complete!")
    print("=" * 60)
    
    print("\n Model Comparison:")
    print(f"   SimpleCNN:   ~1.2M params   67.72% accuracy")
    print(f"   DeeperCNNLight: {params_light:,} params   Target: 70-75%")
    print(f"   DeeperCNN:   {params_full:,} params   Target: 75-80%")