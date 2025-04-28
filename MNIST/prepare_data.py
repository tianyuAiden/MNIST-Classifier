from torchvision import datasets, transforms

# download dataset
train_set = datasets.MNIST("dataset", train=True, download=True, transform=transforms.ToTensor)
test_set = datasets.MNIST("dataset", train=False, download=True, transform=transforms.ToTensor)

