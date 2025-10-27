import dataloader_generator 
from model_VAE.vae import VAE
from model_VAE.con_vae import Conv_VAE
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

'''

this .py is unfinished.



go to train_gans first, then come back a few days later.



'''

def vae_loss_function(u, log_var, out, x, kl_weight = 0.05):
        batch_size = len(x)
        reconstrut_loss = F.l1_loss(out, x, reduction='sum') 
        kl_loss = -torch.sum(1 + log_var - u.pow(2) - log_var.exp()) 
        loss = reconstrut_loss + kl_weight * kl_loss
        return loss / batch_size , reconstrut_loss.item() / batch_size, kl_weight * kl_loss.item() / batch_size

def start_training(model, dataloader, optimizer, epoches ,device):   
    model =  model.to(device)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    model.train()
    for each in range(epoches):
        start_time = time.time()
        sum_loss = 0
        sum_rct_loss = 0
        sum_kl_loss = 0
        count = 0
        for x, label in dataloader:
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u, log_var,  out = model(x)
            loss, reconstrut_loss, kl_loss= vae_loss_function(u, log_var, out, torch.flatten(x, start_dim=1))
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()
            sum_rct_loss += reconstrut_loss
            sum_kl_loss += kl_loss
            count += 1

        #scheduler.step()
        end_time = time.time()    
        print(f"epoch {each} time: {end_time - start_time:.0f}")
        print(f"\t loss:", (sum_loss/count))
        print(f"\t rct_loss:", (sum_rct_loss/count))
        print(f"\t kl_loss:", (sum_kl_loss/count))
        if each % 20 == 0 : generate_data(model,name=f"images_at_ep_{each}")
        model.train()
    generate_data(model,name=f"images_at_ep_{each}")
    torch.save(model.state_dict(),"CIFAR_10_CON_VAE.pth")
    return model

def generate_data(model, sample_size = 9, name = "image.png"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(sample_size, z_dim).to(device)
        images = model.generate_images(z)
        images = images.to("cpu")
    grid_image = utils.make_grid(images,normalize=True,nrow=3)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"ai_images/CIFAR_10_CON_VAE_{name}")
    plt.close()
    
# model config here
# you can change the hidden_dim and z_dim here.
z_dim = 256
con_vae_model = Conv_VAE(z_dim)
parameters_load_path ="trained_parameters/" 

# training config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer =  optim.AdamW(con_vae_model.parameters(), lr=0.0001)
dataloader = dataloader_generator.generate_CIFAR_10_dataloader()

# starts training
con_vae_model.load_state_dict(torch.load(f"{parameters_load_path}CIFAR_VAE.pth", map_location=device))
with torch.autograd.set_detect_anomaly(True):
    start_training(con_vae_model, dataloader, optimizer, 200, device)
# then generate some images, you can just load our saved pre-trained model
# model = m.VAE(image_dim, hidden_dim, z_dim)
# con_vae_model.load_state_dict(torch.load("CIFAR_10_CON_VAE.pth", map_location=device))
# con_vae_model.to(device)
# generate_data(con_vae_model,sample_size=16, name= "test.png")
