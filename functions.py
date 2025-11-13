
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
import torchvision.utils as utils
from PIL import Image
from model_DDPM.unet import UNet
from model_DDPM.diffuser import Diffuser
from pytorch_fid import fid_score
import os


def plot_values(losses = [], model_name = "" ,start_index = 0):
    if len(losses) == 0 : return
    Epoches = np.arange(len(losses)) + start_index
    plt.plot(Epoches,losses,label = "Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots_pdf/{model_name}_Loss.pdf")
    plt.close()

def calcualte_fid(path1, path2):
    if os.path.exists(path1) and os.path.exists(path2):
        score = fid_score.calculate_fid_given_paths(
            [path1,path2],
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_size=128,
            dims=2048
        )
        return score
         
    else:
        print("path does not exist")
        return -1


### functions for DDPM 
def load_ddpm_model(model_name, is_attention_on = False, is_residual_on = False, n_label = 0, T = 1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(attention=is_attention_on,residual_on=is_residual_on, n_label= n_label).to(device)
    diffuser = Diffuser(device = device, T = T)
    parameters_load_path ="trained_parameters/" 
    try:
        model.load_state_dict(torch.load(parameters_load_path + model_name + ".pth")) 
        print("load parameters successfully")
    except:
        print("load parameters errors")
    return model, diffuser

# normally use this function to generate imgs for calculate_fid
def generate_imgs(model: UNet, diffuser : Diffuser,save_folder = "_FID_model_gen_images", n_label = 0, n_100 = 100):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    n_samples = n_100
    i = 0
    labels = None
    if n_label > 0 :
        labels = torch.tensor([i for i in range(n_label) for _ in range(10)], dtype=torch.long)
    for index in range(n_samples):
        imgs = diffuser.denoised_sampling(model, imgs_shape=(100,3,32,32), labels=labels)
        for img in imgs:
            print(index)
            save_image(img,fp = f"{save_folder}/gen{i}.png",normalize=True, value_range=(-1, 1))
            i += 1

#normally use this function for visualization and demonstration              
def generate_gird_imgs(model: UNet, diffuser : Diffuser, n_label = 0, modelname = "modelname", epoch_index = 0, save_folder="_grid_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    labels = None
    if n_label > 0:
        labels = torch.tensor([i for i in range(n_label) for _ in range(10)], dtype=torch.long)
    imgs_in_tensor = diffuser.denoised_sampling(model, imgs_shape=(100, 3, 32, 32), labels=labels)
    imgs_in_tensor = imgs_in_tensor.to("cpu")
    grid_image = utils.make_grid(imgs_in_tensor, nrow=10, normalize=True, value_range=(-1,1))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{modelname}{epoch_index}", bbox_inches='tight')
    plt.close()


# data_preprocess of handwritten_characters 
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

## generate real images . For fid_calculation
def save_cifar10_real_images(num_images=60000):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        print(len(dataset))
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(os.path.join("cifar_real_images", f'{i:05d}.png'))

def save_test_cifar10_real_images(num_images=60000):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        print(len(dataset))
        if not os.path.exists("cifar_test_real_images"):
            os.makedirs("cifar_test_real_images")
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(os.path.join("cifar_test_real_images", f'{i:05d}.png'))



def demo_denoise():
    pass

def demo_addnoise():
    pass

