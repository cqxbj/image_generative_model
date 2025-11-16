import torch
import dataloader_generator
import torch.optim 
import torch.nn.functional as F
import functions as my_F
import numpy as np
import time


''' 

    this .py is implemented for training DDPM_based models.

'''

## train DDPM for cifar_10
device = "cuda" if torch.cuda.is_available() else "cpu"
dataloader = dataloader_generator.generate_CIFAR_10_dataloader()


# models parameters
parameters_load_path ="trained_parameters/" 
modelname = "_conditional_residual_attention_DDPM"
T = 1000
    # set n_label = 0 to make model unconditional.
n_label = 10
is_attention_on = True
is_residual_on =True

model, diffuser = my_F.load_ddpm_model(modelname, 
                                       is_attention_on=is_attention_on, 
                                       is_residual_on=is_residual_on, 
                                       n_class=n_label, 
                                       T = T)

# hyper_parameters 
epoches = 500
lr= 0.0001

# other training parameters
epoch_startindex = 0
fre_generate_samples = 10
fre_save_model = 20


# trainning 
start_training = True

if start_training:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_list = []
    for epoch in range(epoches):
        loss_sum = 0
        cnt = 0
        start_time = time.time()
        for img, labels in dataloader:
            optimizer.zero_grad()
            x = img.to(device)
            batch_size = len(x)
            t = torch.randint(1,T+1,size=(batch_size,)).to(device)
            x_noised, real_noise = diffuser.add_noise(x, t)
            if n_label > 0 :
                labels = labels.to(device)
                noise_pre = model(x_noised, t, labels)
            else:
                noise_pre = model(x_noised, t)
            loss = F.mse_loss(noise_pre, real_noise)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1 
            # print(f"{loss}")
        if epoch > 0:
            loss = loss_sum/cnt
            loss_list.append(loss)
            end_time = time.time()
            print(f"epoch {epoch} with loss: {loss},  time: {end_time - start_time:.2f}")
            if epoch % fre_generate_samples == 0 :
                my_F.plot_values(loss_list, model_name=modelname, start_index=epoch_startindex)
                my_F.ddpm_save_samples(model, diffuser, n_class= n_label, modelname = modelname, epoch_index=epoch)
            if epoch % fre_save_model == 0 :
                torch.save(model.state_dict(),parameters_load_path + modelname + ".pth")
    my_F.plot_values(loss_list, model_name=modelname, start_index=epoch_startindex)
    torch.save(model.state_dict(),parameters_load_path + modelname + ".pth")
    np.save(f"{modelname}{epoch_startindex}_losses.py", np.array(loss_list))
