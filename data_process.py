from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import numpy as np
import torch
from PIL import Image


# 1. cifar_10

def generate_CIFAR_10_dataset(train = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    cifar_dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform = transform
        )
    return cifar_dataset

def generate_CIFAR_10_dataloader(train = True):
    return DataLoader(
        generate_CIFAR_10_dataset(train = train),
        batch_size = 128,
        shuffle=True
    )

# 2. hand_writting


# preprocess
# convert hw imgs to numpy.
def _preprocess_imgs_to_numpy():
    image_path = "all_char"
    if not os.path.exists(image_path):
        print("all_char does not exist")
        return

    save_path_train = "data/handwriting_data/train/"
    save_path_val = "data/handwriting_data/val/"
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
    if not os.path.exists(save_path_val):
        os.makedirs(save_path_val)

    #split imgs into train 0.9 and val 0.1
    # add some empty imgs
    # train 
    train_space_path = save_path_train + "__space.npy"
    train_space_np = np.zeros((1000,32,32,1),dtype=np.uint8)
    np.save(train_space_path, train_space_np)
    # val
    val_space_path = save_path_val + "__space.npy"
    val_space_np = np.zeros((100,32,32,1),dtype=np.uint8)
    np.save(val_space_path, val_space_np)

    for folder_name in os.listdir(image_path):
        character_folder_path = os.path.join(image_path, folder_name)
        print(character_folder_path)
        character_train = []
        character_val = []
  
        size = len(os.listdir(character_folder_path))
        train_index_max = int(size*0.9)

        train_paths = os.listdir(character_folder_path)[:train_index_max]
        val_paths = os.listdir(character_folder_path)[train_index_max:]

        for each_img in train_paths:
           each_image_train_path =  os.path.join(character_folder_path, each_img)
           img = Image.open(each_image_train_path).convert("L")
           img_np_array = np.array(img,dtype=np.uint8)[:,:,np.newaxis]
           character_train.append(img_np_array)
        
        for each_img in val_paths:
           each_image_val_path =  os.path.join(character_folder_path, each_img)
           img = Image.open(each_image_val_path).convert("L")
           img_np_array = np.array(img,dtype=np.uint8)[:,:,np.newaxis]
           character_val.append(img_np_array)

        character_train_np = np.array(character_train)
        character_val_np = np.array(character_val)

        # print(folder_name)
        # print(character_train_np.shape)
        # print(character_val_np.shape)
        # print("\n")

        save_char_train_path = save_path_train + f"{folder_name}.npy"
        save_char_val_path = save_path_val + f"{folder_name}.npy"

        np.save(save_char_train_path,character_train_np)
        np.save(save_char_val_path,character_val_np)

def generate_Handwriting_dataloader(train = True, less_space = False):
    dataset = HandWritingDataset(train = train, less_space = less_space) 
    return DataLoader(dataset, 
                      batch_size= 128, 
                      shuffle= True)


class HandWritingDataset(Dataset):
    def __init__(self,  train = True, less_space = False, path = "data/handwriting_data/" ):
        super().__init__()

        if not os.path.exists(path):
            print("not exist")
            _preprocess_imgs_to_numpy()

        if train : path = path + "train/"
        else: path = path + "val/"

        min = 100
        if train: min = 1000

        self.np_path = path
        all_char_list = []
        labels = []

        for index, char_np_files in enumerate(os.listdir(self.np_path)):
            char_np = np.load(os.path.join(self.np_path,char_np_files))
            if len(char_np) < min:
                repeat_data = []
                while len(repeat_data) < min:
                    repeat_data.extend(char_np)

                char_np = np.array(repeat_data[:min])
            
            if less_space and index == len(os.listdir(self.np_path)) - 1:
                n_space = int(0.25*len(char_np))
                char_np = char_np[:n_space].copy()

          

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

