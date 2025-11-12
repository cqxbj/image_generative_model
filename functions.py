
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as utils
from model_DDPM.unet import UNet
from model_DDPM.diffuser import Diffuser


def plot_values(losses = [], model_name = "" ,start_index = 0):
    if len(losses) == 0 : return
    Epoches = np.arange(len(losses)) + start_index
    plt.plot(Epoches,losses,label = "Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots_pdf/{model_name}_Loss.pdf")
    plt.close()

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

def generate_imgs(model: UNet, diffuser : Diffuser,path = "_FID_model_gen_images", n_label = 0, n_100 = 100):
    n_samples = n_100
    i = 0
    labels = None
    if n_label > 0 :
        labels = torch.tensor([i for i in range(n_label) for _ in range(10)], dtype=torch.long)
    for index in range(n_samples):
        imgs = diffuser.sampling(model, imgs_shape=(100,3,32,32), labels=labels)
        for img in imgs:
            print(index)
            save_image(img,fp = f"{path}/gen{i}.png",normalize=True, value_range=(-1, 1))
            i += 1
                
def generate_gird_imgs(model: UNet, diffuser : Diffuser, n_label = 0, modelname = "modelname", epoch_index = 0, parent_folder="_grid_images"):
    labels = None
    if n_label > 0:
        labels = torch.tensor([i for i in range(n_label) for _ in range(10)], dtype=torch.long)
    imgs_in_tensor = diffuser.denoised_sampling(model, imgs_shape=(100, 3, 32, 32), labels=labels)
    imgs_in_tensor = imgs_in_tensor.to("cpu")
    grid_image = utils.make_grid(imgs_in_tensor, nrow=10, normalize=True, value_range=(-1,1))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{parent_folder}/{modelname}{epoch_index}")
    plt.close()

# def generate_gird_imgs(imgs, nrow = 4, modelname = "modelname", epoch_index = 0, path="_grid_images"):
#     imgs = imgs.to("cpu")
#     grid_image = utils.make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1,1))
#     plt.imshow(grid_image.permute(1,2,0))
#     plt.axis("off")
#     plt.savefig(f"{path}/{modelname}{epoch_index}")
#     plt.close()

def demo_denoise():
     pass

def demo_addnoise():
     pass

