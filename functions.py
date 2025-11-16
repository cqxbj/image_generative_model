
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
import torchvision.utils as utils
from model_DDPM.unet import UNet
from model_DDPM.diffuser import Diffuser
from pytorch_fid import fid_score
from model_VAE.con_vae import Conv_VAE
from dataloader_generator import HandWrittenDataset
from model_GANs.gans import Discriminator, Generator  
import shutil
import os


''' 

    this .py is implemented for additional functions.
    1. DDPM functions
    2. GANs functions
    3. VAE  functions
    4. Other functions

'''

''' 
    1. DDPM functions.
'''

def demo_denoise():
    pass

def demo_addnoise():
    pass

def load_ddpm_model(
        model_name, 
        is_attention_on = False, is_residual_on = False, 
        n_class = 10, T = 1000, 
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    model = UNet(attention=is_attention_on,residual_on=is_residual_on, n_class= n_class).to(device)
    diffuser = Diffuser(device = device, T = T)
    parameters_load_path ="trained_parameters/" 
    try:
        model.load_state_dict(torch.load(parameters_load_path + model_name + ".pth")) 
        print("DDPM load pretrained_parameters successfully")
    except:
        print("DDPM load pretrained_parameters error")
    finally:
        num_param =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"DDPM model size:{num_param} parameters")
    return model, diffuser

#normally we call this function to generate imgs for FID testing
def ddpm_generate_imgs(
        model: UNet, 
        diffuser : Diffuser,
        save_folder = "_FID_imgs/ddpm",
        n_100 = 100):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    i = 0
    labels = None
    if model.n_class > 0 :
        labels = torch.tensor([i for i in range(model.n_class) for _ in range(10)], dtype=torch.long)
    for _ in range(n_100):
        imgs = diffuser.denoised_sampling(model, imgs_shape=(100,3,32,32), labels=labels)
        for img in imgs:
            save_image(img,fp = f"{save_folder}/gen{i}.png",normalize=True, value_range=(-1, 1))
            i += 1

#normally use this function for visualization and demonstration              
def ddpm_save_samples(
        model: UNet, diffuser : Diffuser,  modelname = "modelname",
        epoch_index = 0, save_folder="_samples/ddpm_grid_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    labels = None
    if model.n_class > 0:
        labels = torch.tensor([i for i in range(model.n_class) for _ in range(10)], dtype=torch.long)
    imgs_in_tensor = diffuser.denoised_sampling(model, imgs_shape=(100, 3, 32, 32), labels=labels)
    imgs_in_tensor = imgs_in_tensor.to("cpu")
    grid_image = utils.make_grid(imgs_in_tensor, nrow=10, normalize=True, value_range=(-1,1))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{modelname}{epoch_index}", bbox_inches='tight', pad_inches=1)
    plt.close()


''' 
    2. GANs functions.
'''

def load_gans_model(model_name, 
                    z_dim = 128,
                    n_class = 37,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    parameters_load_path ="trained_parameters/" 

    generator = Generator(z_dim=z_dim, n_class=n_class).to(device)
    discriminator = Discriminator(label_dim=z_dim, n_class=n_class).to(device)

    try:
        generator.load_state_dict(torch.load(parameters_load_path + model_name+"_G"+ ".pth")) 
        discriminator.load_state_dict(torch.load(parameters_load_path + model_name+"_D" + ".pth")) 
        print("GANs load pretrained_parameters successfully")
    except:
        print("GANs load pretrained_parameters error")
    finally:
        num_param_G =  sum(p.numel() for p in generator.parameters() if p.requires_grad)
        num_param_D =  sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"GANs discriminator size :{num_param_D} parameters")
        print(f"GANs generator size :{num_param_G} parameters")
    
    return generator, discriminator





def gans_generate_imgs(
        generator:Generator,
        save_folder = "_FID_imgs/gans",
        device = "cuda" if torch.cuda.is_available() else "cpu",
        n_100 = 100):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if generator.n_class > 0:
        labels = torch.tensor([i for i in range(generator.n_class - 1) for _ in range(100)],dtype=torch.long).to(device)
    
    generator.eval()
    with torch.no_grad():
        i = 0
        for _ in range(10):
            z = torch.randn(len(labels), generator.z_dim).to(device)
            imgs = generator(z, labels=labels)
            for img in imgs:
                img = transforms.ToPILImage()(img)
                img.save(os.path.join(save_folder, f'{i:05d}.png'))
                i += 1
    generator.train()




def gans_save_samples(generator, 
                     black_in_white = True,
                     input_str = "WE ALL LIKE COMP7015", 
                     epoch = 0 ,
                     model_name = "gans",
                     device = "cuda" if torch.cuda.is_available() else "cpu", 
                     save_folder = "_samples/gans_grid_images"):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    labels, max_len = muti_lines_tokenize(input_str)
    labels = labels.to(device)
    # labels = handwriting_tokenize(input_str).to(device)
        
    generator.eval()
    with torch.no_grad():
        noises = torch.randn(len(labels), generator.z_dim).to(device)
        images = generator(noises, labels)
        images = images.to("cpu")
    generator.train()

    if black_in_white: images = 1 - images
    grid_image = utils.make_grid(images, nrow=max_len, normalize=True, padding=0)
    plt.figure(figsize=(80, 80))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{model_name}{epoch}",
                bbox_inches='tight', 
                pad_inches=1)
    plt.close()    

''' 
    3. VAE functions.
'''

list = ['0','1','2','3','4','5','6','7','8','9',
                    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",' ']
char_dict = {char: index for index, char in enumerate(list)}
def handwriting_tokenize(str):    
    tokens = []
    for s in str:
         tokens.append(char_dict.get(s, 36))
    return torch.tensor(tokens)



def muti_lines_tokenize(str:str):
    lines = str.split("\n")
    max_len = max([len(e)for e in lines])
    new_str = ""
    for each_line in lines:
        each_line = each_line + " " * (max_len - len(each_line))
        new_str += each_line
    tokens = handwriting_tokenize(new_str)    
    return tokens, max_len  



def load_vae_model(
        model_name,
        z_dim = 128, 
        n_class = 40, 
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    model = Conv_VAE(z_dim = z_dim, name=model_name, n_class= n_class)
    parameters_load_path ="trained_parameters/" 
    try:
        model.load_state_dict(torch.load(parameters_load_path + model_name + ".pth")) 
        print("VAE load pretrained_parameters successfully")
    except:
        print("VAE load pretrained_parameters error")
    finally:
        num_param =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"vae model size:{num_param} parameters")
    model.to(device)
    return model


#normally we call this function to generate imgs for FID testing.
def vae_generate_imgs(
        model:Conv_VAE, 
        save_folder = "_FID_imgs/vae", 
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    labels = torch.tensor([i for i in range(model.n_class - 1) for _ in range(100)],dtype=torch.long).to(device)
 
    i = 0
    model.eval()
    with torch.no_grad():
        for _ in range(10): 
            z = torch.randn(len(labels), model.z_dim).to(device)
            imgs = model.generate_images(z, labels=labels)
            for img in imgs:
                img = transforms.ToPILImage()(img)
                img.save(os.path.join(save_folder, f'{i:05d}.png'))
                i += 1
    model.train()


def vae_save_samples(model:Conv_VAE, 
                     black_in_white = True,
                     input_str = "WE ALL LIKE COMP7015", 
                     epoch = 0 ,
                     device = "cuda" if torch.cuda.is_available() else "cpu", 
                     save_folder = "_samples/vae_grid_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model.eval()
    len_str = len(input_str)
    with torch.no_grad():
        z = torch.randn(len_str, model.z_dim).to(device)
        labels = handwriting_tokenize(input_str).to(device)
        images = model.generate_images(z, labels)
        if black_in_white : images = 1 - images
        images = images.to("cpu")
    grid_image = utils.make_grid(images,normalize=True, nrow=min(50, len_str), padding=0)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{model.name}{epoch}", 
                bbox_inches='tight', 
                pad_inches=1)
    plt.close()
    model.train()


''' 
    4. Others.
'''

def plot_values(
        losses1 = [], losses2 = [], losses3 = [], labels = None,       
        model_name = "" ,start_index = 0, save_folder = "plots_pdf"):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(losses1) == 0 and len(losses2) == 0 and len(losses3) == 0:
        return
    
    if labels is None:
        labels = ["Loss 1", "Loss 2", "Loss 3"]
    
    plt.figure(figsize=(10, 6))
    
    if len(losses1) > 0:
        epochs1 = np.arange(len(losses1)) + start_index
        plt.plot(epochs1, losses1, label=labels[0], linewidth=2)
    
    if len(losses2) > 0:
        epochs2 = np.arange(len(losses2)) + start_index
        plt.plot(epochs2, losses2, label=labels[1], linewidth=2)
    
    if len(losses3) > 0:
        epochs3 = np.arange(len(losses3)) + start_index
        plt.plot(epochs3, losses3, label=labels[2], linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Losses - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{save_folder}/{model_name}_Loss.pdf", bbox_inches='tight')
    plt.close()

def calcualte_fid(path1, path2):

    path1 = "_FID_imgs/" + path1
    path2 = "_FID_imgs/" + path2

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

# normally we call those two go generate imgs for FID test
def save_hand_writing_images(save_folder = "hw_train"):
    save_path = "_FID_imgs/" + save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    dataset = HandWrittenDataset()
    for i in range(len(dataset)):
        image, _ = dataset[i]
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(os.path.join(save_path, f'{i:05d}.png'))

def save_cifar10_images(save_folder = "cifar_train", num_images=60000, train_data = True):
        save_path = "_FID_imgs/" + save_folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=train_data, 
            download=True, 
            transform=transform
        )
        print(len(dataset))
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(os.path.join(save_path, f'{i:05d}.png'))




