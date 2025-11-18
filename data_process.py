from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import numpy as np
import torch
from PIL import Image

def generate_MNIST_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor()
        ]
        )
    mnist_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform = transform
        )

    return DataLoader(
        mnist_dataset,
        batch_size = 1024,
        shuffle=True
    )


def generate_CIFAR_10_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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


def generate_Handwritten_dataloader():
    dataset = HandWrittenDataset() 
    return DataLoader(dataset, batch_size= 128, shuffle= True)

class HandWrittenDataset(Dataset):
    def __init__(self, path = "./data/handwriting_data" ):
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
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        img = self.all_char_list[index]
        label = self.labels[index]
        img = self.transform(img)
        label = torch.tensor(label)
        return img, label

HandWrittenDataset()
# preprocess
# convert hw imgs to numpy.
def __imgs_to_numpy():
    folder_path = "data/all_char"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_folder = "data/handwriting_data"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # add some empty imgs
    save_path = f"data/handwriting_data/__space.npy"
    character_empty_np = np.zeros((1000,32,32,1),dtype=np.uint8)
    np.save(save_path, character_empty_np)
    
    for folder_name in os.listdir(folder_path):
        character_folder_path = os.path.join(folder_path, folder_name)
        print(character_folder_path)
        character_s = []
        for each_img in os.listdir(character_folder_path):
           image_path =  os.path.join(character_folder_path, each_img)
           img = Image.open(image_path).convert("L")
           img_np_array = np.array(img,dtype=np.uint8)[:,:,np.newaxis]
           character_s.append(img_np_array)
        character_s_np = np.array(character_s)
        print(character_s_np.shape)
        save_path = f"./data/handwritten_data/{folder_name}.npy"
        np.save(save_path,character_s_np)



class Tokenizer():
    def __init__(self):
        self.list = ['0','1','2','3','4','5','6','7','8','9',
                     "A","B","C","D","E","F","G","H","I","J",
                     "K","L","M","N","O","P","Q","R","S","T",
                     "U","V","W","X","Y","Z","a","b","c","d",
                     "e","f","g","h","i","j","k","l","m","n",
                     "o","p","q","r","s","t","u","v","w","x","y","z"," "]
                     
        self.char_dict = {char: index for index, char in enumerate(self.list)}

    def handwriting_tokenize(self, str):    
        tokens = []
        for s in str:
            tokens.append(self.char_dict.get(s, 62))
        return torch.tensor(tokens)


    def muti_lines_tokenize(self, str:str):
        lines = str.split("\n")
        max_len = max([len(e)for e in lines])
        new_str = ""
        for each_line in lines:
            each_line = each_line + " " * (max_len - len(each_line))
            new_str += each_line
        tokens = self.handwriting_tokenize(new_str)    
        return tokens, max_len  

