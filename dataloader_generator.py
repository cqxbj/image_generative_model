from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset
import os
import numpy as np
import torch

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
        transforms.ToTensor(),
        transforms.Lambda(lambda img:img.transpose(1,2))]
        )
    
    # transforms.Lambda(lambda img:img.transpose(1,2))
    emnist_dataset = datasets.EMNIST(
            root="./data",
            split="byclass",
            download=True,
            transform = transform
        )
    return DataLoader(
        emnist_dataset,
        batch_size = 64,
        shuffle=True
    )


def generate_Handwritten_dataloader():
    dataset = HandWrittenDataset() 
    return DataLoader(dataset, batch_size= 64, shuffle= True)

class HandWrittenDataset(Dataset):
    def __init__(self, path = "./data/handwritten_data" ):
        super().__init__()
        self.np_path = path
        all_char_list = []
        labels = []
        for index, char_np_files in enumerate(os.listdir(self.np_path)):
            char_np = np.load(os.path.join(self.np_path,char_np_files))
            all_char_list.append(char_np)
            labels.extend([index] * len(char_np))

        self.all_char_list = np.concatenate(all_char_list,axis=0)
        self.labels = np.array(labels)
        print(self.all_char_list.shape)
        print(self.labels.shape)

        self.transform =  transforms.Compose([
            transforms.ToTensor()])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        img = self.all_char_list[index]
        label = self.labels[index]
        
        img = self.transform(img)
        label = torch.tensor(label)
        return self.all_char_list[index], self.labels[index]
