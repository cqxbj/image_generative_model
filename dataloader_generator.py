from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def generate_MNIST_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor()]
        )
    mnist_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform = transform
        )
    return DataLoader(
        mnist_dataset,
        batch_size = 128,
        shuffle=True
    )

def generate():
    pass

def generate_CIFAR_10_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor()]
        )
    cifar_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform = transform
        )
    return DataLoader(
        cifar_dataset,
        batch_size = 128,
        shuffle=True
    )

def generate_EMNIST_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),]
        )
    emnist_dataset = datasets.EMNIST(
            root="./data",
            split="letters",
            download=True,
            transform = transform
        )
    return DataLoader(
        emnist_dataset,
        batch_size = 128,
        shuffle=True
    )
