import numpy as np
from PIL import Image
import os

def process_data():
    folder_path = "./data/handwritten-characters-complete/train"

    for folder_name in os.listdir(folder_path):
        character_folder_path = os.path.join(folder_path, folder_name)
        print(character_folder_path)
        
        character_s = []

        for each_img in os.listdir(character_folder_path):
           image_path =  os.path.join(character_folder_path, each_img)
           img = Image.open(image_path).convert("L")
           
           img_np_array = np.array(img,dtype=np.float32)[np.newaxis,:,:]

           character_s.append(img_np_array)
        
        character_s_np = np.array(character_s)
        
        print(character_s_np.shape)

        save_path = f"./data/handwritten_data/{folder_name}.npy"
        np.save(save_path,character_s_np)


process_data()