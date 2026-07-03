import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleCNN

def quick_test():
    print("Testing setup...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test model
    model = SimpleCNN()
    print("Model loaded successfully!")
    
    # Test data loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    print(f"Data loaded: {len(test_data)} samples")
    
    return True

if __name__ == "__main__":
    quick_test()
    print("Setup complete!")
