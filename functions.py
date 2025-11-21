
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
import data_process
from torchvision.utils import save_image
import torchvision.utils as utils
from model_DDPM.unet import UNet
from model_DDPM.diffuser import Diffuser
from pytorch_fid import fid_score
from model_VAE.vae import VAE
from data_process import HandWritingDataset
from model_GANs.gans import Discriminator, Generator  
from data_process import Tokenizer
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


def load_ddpm_model(
        model_name, 
        is_attention_on = True, 
        is_residual_on = True, 
        n_class = 10, 
        T = 1000, 
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
        imgs = (torch.clamp(imgs, -1, 1) + 1 ) / 2
        for img in imgs:
            save_image(img,fp = f"{save_folder}/gen{i}.png",normalize=False)
            i += 1

#normally use this function for visualization and demonstration              
def ddpm_save_samples(
        model: UNet, diffuser : Diffuser,  modelname = "modelname",
        epoch_index = 0, save_folder="_samples/ddpm_grid_images", select_class = -1):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    labels = None
    if model.n_class > 0:
        if select_class >=0 : labels = torch.tensor([select_class for i in range(model.n_class) for _ in range(10)], dtype=torch.long)
        else : labels = torch.tensor([i for i in range(model.n_class) for _ in range(10)], dtype=torch.long)
    imgs_in_tensor = diffuser.denoised_sampling(model, imgs_shape=(100, 3, 32, 32), labels=labels)
    imgs_in_tensor = imgs_in_tensor.to("cpu")
    imgs_in_tensor = (torch.clamp(imgs_in_tensor, -1, 1) + 1 ) / 2
    grid_image = utils.make_grid(imgs_in_tensor, nrow=10)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{modelname}{epoch_index}", bbox_inches='tight', pad_inches=0.1)
    plt.close()

# add noises on random_10 imgs from cifar_10 dataset.
def demo_addnoise(diffuser ,save_folder = "demo_save"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    data_set = data_process.generate_CIFAR_10_dataset()
    random_10_index = torch.randint(0, len(data_set) - 1, size=(10,))
    random_10_images = [data_set[i][0] for i in random_10_index]
    imgs_t0_to_t1000 = []
    random_10_images = torch.stack(random_10_images).to(diffuser.device)
    # add noise fro images from 100 - 1000
    for i in range(5,1000,100):
        t = torch.tensor([i]*10).to(diffuser.device)
        img_noised, _ = diffuser.add_noise(random_10_images, t=t)
        imgs_t0_to_t1000.extend(img_noised)
    imgs_t0_to_t1000 = torch.stack(imgs_t0_to_t1000).to("cpu")

    grid_image = utils.make_grid(imgs_t0_to_t1000, nrow=10, normalize=True, value_range=(-1,1))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/ddpm_addnoise", bbox_inches='tight', pad_inches=0.1)
    plt.close()
            
# demo denoise process with our pretrained model
def demo_denoise(model, diffuser, save_folder = "demo_save"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    T = diffuser.T
    labels = torch.tensor([i for i in range(10)]).to(diffuser.device)
    imgs = torch.randn((10,3,32,32)).to(diffuser.device)

    imgs_T1000_to_T0 = [] 
    model.eval()
    for i in range(T, 0, -1):
        t = torch.tensor([i]*10,dtype =torch.long).to(diffuser.device)
        imgs = diffuser.de_noise(imgs, t, model, labels)
        
        if (i % 100 == 0 and i != 1000) or (i == 1):
            imgs_T1000_to_T0.extend(imgs)
    model.train()
    
    imgs_T1000_to_T0 = torch.stack(imgs_T1000_to_T0)
    imgs_T1000_to_T0 = imgs_T1000_to_T0.to("cpu")

    imgs_T1000_to_T0 = (torch.clamp(imgs_T1000_to_T0, -1, 1) + 1 ) / 2
    grid_image = utils.make_grid(imgs_T1000_to_T0, nrow=10)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/ddpm_denoise", bbox_inches='tight', pad_inches=0.1)
    plt.close()

''' 
    2. GANs functions.
'''

def load_gans_model(model_name, 
                    z_dim = 256,
                    n_class = 63,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    if generator.n_class <= 1: return
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    generator.eval()
    with torch.no_grad():
        labels = torch.tensor([i for i in range(generator.n_class) for _ in range(100)],dtype=torch.long).to(device)
        i = 0
        for _ in range(2):
            noise = torch.randn(len(labels), generator.z_dim).to(device)
            imgs = generator(noise, labels=labels)
            for img in imgs:
                img = transforms.ToPILImage()(img)
                img.save(os.path.join(save_folder, f'{i:05d}.png'))
                i += 1
    generator.train()


def gans_save_samples(generator, 
                     show_now=False,
                     black_in_white = True,
                     input_str = "WE ALL LIKE COMP7015", 
                     epoch = 0 ,
                     model_name = "gans",
                     device = "cuda" if torch.cuda.is_available() else "cpu", 
                     save_folder = "_samples/gans_grid_images"):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    generator.eval()
    with torch.no_grad():
        tk = Tokenizer()
        labels, max_len = tk.muti_lines_tokenize(input_str)
        labels = labels.to(device)
        noises = torch.randn(len(labels), generator.z_dim).to(device)
        images = generator(noises, labels)
        images = images.to("cpu")
    generator.train()

    if black_in_white: images = 1 -images
    grid_image = utils.make_grid(images, nrow=max_len, normalize=False, padding=0)
    plt.figure(figsize=(200, 160))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{model_name}{epoch}",
                bbox_inches='tight', 
                pad_inches=1)
    if show_now : plt.show()
    else: plt.close()    

''' 
    3. VAE functions.
'''

def load_vae_model(
        model_name,
        z_dim = 256, 
        n_class = 63, 
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    model = VAE(z_dim = z_dim, n_class= n_class)
    parameters_load_path ="trained_parameters/" 
    try:
        model.load_state_dict(torch.load(parameters_load_path + model_name + ".pth")) 
        print("VAE load pretrained_parameters successfully")
    except:
        print("VAE load pretrained_parameters error")
    finally:
        num_param =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"VAE model size:{num_param} parameters")
    model.to(device)
    return model


#normally we call this function to generate imgs for FID testing.
def vae_generate_imgs(
        model:VAE, 
        save_folder = "_FID_imgs/vae", 
        device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    if model.n_class <= 1: return
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.eval()
    with torch.no_grad():
        
        labels = torch.tensor([i for i in range(model.n_class) for _ in range(100)],dtype=torch.long).to(device)
        i = 0
        for _ in range(2): 
            z = torch.randn(len(labels), model.z_dim).to(device)
            imgs = model.generate_images(z, labels=labels)
            for img in imgs:
                img = transforms.ToPILImage()(img)
                img.save(os.path.join(save_folder, f'{i:05d}.png'))
                i += 1
    model.train()


def vae_save_samples(model:VAE, 
                     show_now = False,
                     black_in_white = True,
                     model_name = "vae",
                     input_str = "WE ALL LIKE COMP7015", 
                     epoch = 0 ,
                     device = "cuda" if torch.cuda.is_available() else "cpu", 
                     save_folder = "_samples/vae_grid_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
 
    model.eval()
    tk = Tokenizer()
    with torch.no_grad():

        labels, max_len = tk.muti_lines_tokenize(input_str)
        labels = labels.to(device)
        noises = torch.randn(len(labels), model.z_dim).to(device)
        images = model.generate_images(noises, labels)
        
    if black_in_white : images = 1-images
    images = images.to("cpu")
    model.train()
    grid_image = utils.make_grid(images,normalize=False, nrow=max_len, padding=0)
    plt.figure(figsize=(200, 160))
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"{save_folder}/{model_name}{epoch}", 
                bbox_inches='tight', 
                pad_inches=1)
    if show_now : plt.show()
    else: plt.close()

''' 
    4. Others.
'''

def plot_list(
        losses1 = [], losses2 = [], losses3 = [], eval_losses = [], eval_interval = 10, eval_start_index = 10, labels = None,       
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
    
    if len(eval_losses) > 0:
        eval_epochs = np.arange(len(eval_losses)) * eval_interval + eval_start_index
        plt.plot(eval_epochs, eval_losses, label=labels[3] if len(labels) > 3 else "Eval Loss", 
                marker='o', markersize=4, linewidth=2, linestyle='--', alpha=0.8) 
    
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

        fid_score.save_fid_stats
        return score
         
    else:
        print("path does not exist")
        return -1

# normally we call those two go generate imgs for FID test
def save_hand_writing_images(train = True):

    if train: save_folder = "hw_train"
    else : save_folder = "hw_val"
    save_path = "_FID_imgs/" + save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
        dataset = HandWritingDataset(train=train)
        for i in range(len(dataset)):
            image, _ = dataset[i]
            save_image(image,os.path.join(save_path, f'{i:05d}.png'))

def save_cifar10_images(num_images=60000, train_data = True):
        if train_data: save_path = "_FID_imgs/" + "cifar_train"
        else: save_path = "_FID_imgs/" + "cifar_test"
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
            for i in range(min(num_images, len(dataset))):
                image, _ = dataset[i]
                save_image(image,os.path.join(save_path, f'{i:05d}.png'))


def save_all_FID_test_imgs():
    print("generate FID test imgs")
    save_hand_writing_images(train=True)
    save_hand_writing_images(train=False)
    save_cifar10_images(train_data=True)
    save_cifar10_images(train_data=False)
    print("generate FID test imgs done")
