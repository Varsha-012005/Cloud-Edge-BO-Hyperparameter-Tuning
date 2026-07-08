from torchvision import datasets
print("Downloading CIFAR-10 training set...")
datasets.CIFAR10('./data', train=True, download=True)
print("Downloading CIFAR-10 test set...")
datasets.CIFAR10('./data', train=False, download=True)
print("Download complete!")