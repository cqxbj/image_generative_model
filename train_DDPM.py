import torch
import data_process
import torch.optim 
import torch.nn.functional as F
import functions as my_F
import numpy as np
import time


''' 

    this .py is implemented for training DDPM_based models.

'''
model_name = "ddpm"

## train DDPM for cifar_10
device = "cuda" if torch.cuda.is_available() else "cpu"
dataloader = data_process.generate_CIFAR_10_dataloader()
val_dataloader = data_process.generate_CIFAR_10_dataloader(train = False)


# models parameters
parameters_load_path ="trained_parameters/" 
T = 1000
    # set n_label = 0 to make model unconditional.
n_label = 10
is_attention_on = True
is_residual_on =True

model, diffuser = my_F.load_ddpm_model(model_name, 
                                       is_attention_on=is_attention_on, 
                                       is_residual_on=is_residual_on, 
                                       n_class=n_label, 
                                       T = T)

# hyper_parameters 
epoches = 500
lr= 0.0001

# other training parameters
loss_plot_startindex = 1
fre_generate_samples = 10
fre_save_model = 100
fre_eval_model = 5

# training 
start_training = True

if start_training:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_list = []
    eval_loss_list = []
    step = 0

    for epoch in range(epoches):
        
        
        # training 
        loss_sum = 0
        cnt = 0
        start_time = time.time()

        for img, labels in dataloader:
            optimizer.zero_grad()
            batch_size = len(img)
            x = img.to(device)
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
            print(f"epoch {epoch} step {step} with loss: {loss}")
            step += 1


        # evaluation
        if epoch >0 and epoch % fre_eval_model == 0:
            model.eval()
            with torch.no_grad():
                eval_loss_sum = 0
                val_cnt = 0
                for val_img, val_labels in val_dataloader:
                    val_batch_size = len(val_img)
                    val_img = val_img.to(device)
                    t = torch.randint(1,T+1,size=(val_batch_size,)).to(device)
                    val_x_noised, val_real_noise = diffuser.add_noise(val_img, t)
                    if n_label > 0 :
                        val_labels = val_labels.to(device)
                        val_noise_pre = model(val_x_noised, t, val_labels)
                    else:
                        val_noise_pre = model(val_x_noised, t)
                    _loss = F.mse_loss(val_noise_pre, val_real_noise)  
                    eval_loss_sum += _loss.item()
                    val_cnt += 1
                

                eval_loss_sum = eval_loss_sum / val_cnt
                eval_loss_list.append(eval_loss_sum)
                print(f"epoch {epoch} with eval_loss: {eval_loss_sum}")
            model.train()

        
        if epoch > 0:
            loss_epoch = loss_sum/cnt
            loss_list.append(loss_epoch)
            end_time = time.time()
            print(f"epoch {epoch} with loss: {loss_epoch},  time: {end_time - start_time:.2f}")
            my_F.plot_list(loss_list,
                           eval_losses=eval_loss_list,
                           eval_interval=fre_eval_model, 
                           eval_start_index=fre_eval_model, 
                           model_name=model_name, 
                           start_index=loss_plot_startindex)
            if epoch % fre_generate_samples == 0 :
                my_F.ddpm_save_samples(model, diffuser, modelname = model_name, epoch_index=epoch)
            if epoch % fre_save_model == 0 :
                torch.save(model.state_dict(),parameters_load_path + model_name +str(epoch)+ ".pth")



    my_F.plot_list(loss_list, 
                   eval_losses=eval_loss_list,
                   eval_interval=fre_eval_model, 
                   eval_start_index=fre_eval_model, 
                   model_name=model_name, 
                   start_index=loss_plot_startindex)
    torch.save(model.state_dict(),parameters_load_path + model_name + ".pth")

