import dataloader_generator 
from model_VAE.con_vae import Conv_VAE
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import functions as my_F

''' 

    this .py is implemented for training VAE_based models.

'''


def __vae_loss_function(u, log_var, out, x, kl_weight = 0.3):
        batch_size = len(x)
        reconstrut_loss = F.mse_loss(out, x, reduction='sum') 
        kl_loss = -torch.sum(1 + log_var - u.pow(2) - log_var.exp()) 
        loss = reconstrut_loss + kl_weight * kl_loss
        return loss / batch_size , reconstrut_loss.item() / batch_size, kl_weight * kl_loss.item() / batch_size


parameters_load_path ="trained_parameters/" 
device = "cuda" if torch.cuda.is_available() else "cpu"

#model parameter
model_name = "vae"
z_dim = 128
n_class = 37
model = my_F.load_vae_model(model_name, z_dim=z_dim, n_class=n_class, device=device)


#hyper parameter
lr = 0.0003
optimizer =  optim.AdamW(model.parameters(), lr=lr)
dataloader = dataloader_generator.generate_Handwritten_dataloader()
epoches = 200

start_training = True
if start_training:
    model.train()
    loss_list = []
    rct_loss_list = []
    kl_loss_list = []
    for epoch in range(epoches):
        start_time = time.time()
        sum_loss = 0
        sum_rct_loss = 0
        sum_kl_loss = 0
        count = 0
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            u, log_var,  out = model(x, labels)
            loss, reconstrut_loss, kl_loss= __vae_loss_function(u, log_var, out, x)
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()
            sum_rct_loss += reconstrut_loss
            sum_kl_loss += kl_loss
            count += 1

        if epoch > 0: 
            end_time = time.time()    
            print(f"epoch {epoch} time: {end_time - start_time:.0f}")

            loss_list.append(sum_loss/count)
            rct_loss_list.append(sum_rct_loss/count)
            kl_loss_list.append(sum_kl_loss/count)
                
            print(f"\t loss:", (sum_loss/count))
            print(f"\t rct_loss:", (sum_rct_loss/count))
            print(f"\t kl_loss:", (sum_kl_loss/count))  
            my_F.vae_save_samples(model, epoch= epoch,input_str="DAAI COMP7015 XIANGBOJIE")
            my_F.plot_values(loss_list, rct_loss_list, kl_loss_list, start_index=1, labels=["loss", "rct_loss", "kl_loss"],model_name=model_name)

    my_F.plot_values(loss_list, rct_loss_list, kl_loss_list, start_index=1,labels=["loss", "rct_loss", "kl_loss"], model_name=model_name)
    my_F.vae_save_samples(model,epoch= epoch, device= device)
    torch.save(model.state_dict(),parameters_load_path+model_name+".pth")

