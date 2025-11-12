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

def __vae_loss_function(u, log_var, out, x, kl_weight = 0.1):
        batch_size = len(x)
        reconstrut_loss = F.mse_loss(out, x, reduction='sum') 
        kl_loss = -torch.sum(1 + log_var - u.pow(2) - log_var.exp()) 
        loss = reconstrut_loss + kl_weight * kl_loss
        return loss / batch_size , reconstrut_loss.item() / batch_size, kl_weight * kl_loss.item() / batch_size

def _save_samples(model, sample_size = 12, epoch = 0 , z_dim = 128, device = "cpu"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(sample_size, z_dim).to(device)
        images = model.generate_images(z)
        images = images.to("cpu")
    grid_image = utils.make_grid(images,normalize=True,nrow=4)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"ai_images/{model.name}{epoch}")
    plt.close()
    model.train()


z_dim = 128
model_name = "hw_VAE"
model = Conv_VAE(z_dim, name=model_name)
parameters_load_path ="trained_parameters/" 

# training config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer =  optim.AdamW(model.parameters(), lr=0.0001)
dataloader = dataloader_generator.generate_CIFAR_10_dataloader()
model.load_state_dict(torch.load(parameters_load_path + model_name+ ".pth"))
epoches = 200
start_training = True

if start_training:
    model =  model.to(device)
    model.train()
    for epoch in range(epoches):
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
            loss, reconstrut_loss, kl_loss= __vae_loss_function(u, log_var, torch.flatten(out, start_dim=1), torch.flatten(x, start_dim=1))
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()
            sum_rct_loss += reconstrut_loss
            sum_kl_loss += kl_loss
            count += 1

        end_time = time.time()    
        print(f"epoch {epoch} time: {end_time - start_time:.0f}")
        print(f"\t loss:", (sum_loss/count))
        print(f"\t rct_loss:", (sum_rct_loss/count))
        print(f"\t kl_loss:", (sum_kl_loss/count))        
        if epoch % 2 == 0 : _save_samples(model,epoch= epoch, device= device)
    _save_samples(model,name=f"images_at_ep_{epoch}")
    torch.save(model.state_dict(),parameters_load_path+model_name+".pth")

