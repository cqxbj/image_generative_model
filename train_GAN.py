import torch
import torch.nn.functional as F
import data_process as dg
import torch
import torch.optim as optim

from model_GANs.gans import Discriminator, Generator  
import torchvision.utils as utils
import matplotlib.pyplot as plt 
import functions as my_F
import os

''' 

    this .py is implemented for training GANs_based models.

'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------------------------------
### step - 1  load model and data

model_name = "gans" 
z_dim = 256
n_class = 63
parameters_load_path ="trained_parameters/" 
generator, discriminator = my_F.load_gans_model(model_name, z_dim=z_dim, n_class=n_class, device=device)


data_loader = dg.generate_Handwritten_dataloader()

# ----------------------------------------------------------------------------------------------------------------------------------------------

### step - 2  hyper_parameters
epoches = 50
lr_d = 0.0002
lr_g = 0.0002
n_repeat_generator = 2
loss_threshold = 2


discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=lr_d)
generator_optimizer = optim.AdamW(generator.parameters(), lr=lr_g)

freq_gen_sample = 1
# ----------------------------------------------------------------------------------------------------------------------------------------------
### step -3 GANs training
start_training = True

if start_training:
    len_data = len(data_loader)
    d_losses = []
    g_losses = []

    generator.train()
    discriminator.train()
    for epoch in range(epoches):
        sum_g_loss = 0
        sum_d_loss = 0
        for real_images, labels in data_loader:
            batch_size = len(real_images)
            real_images = real_images.to(device)
            labels = labels.to(device)

            '''
            To understand this training process - > check GANs loss function:
            Minimax Objective: min_G max_D V(D,G) = E_{x~p_data(x)}[log(D(x))] + E_{z~p_data(z)}[log(1 - D(G(z)))]
            loss_d = - E_{x~p_data(x)}[log(D(x))] - E_{z~p_z(z)}[log(1 - D(G(z)))]
            loss_g = - E_{z~p_z(z)}[log(D(G(z)))]
            '''     
            ### ------ step -3.1 - Train Discriminator 
            discriminator.train()
            discriminator_optimizer.zero_grad()
            # we need to train this Discriminator with a real batch and a noise batch
            # on a real batch
            ground_truth = torch.ones(batch_size,1).to(device)
            discriminator_prediction = discriminator(real_images, labels)
            d_loss_1 = F.binary_cross_entropy(discriminator_prediction, ground_truth)
            
            # on a fake batch
            noises = torch.randn(batch_size, z_dim).to(device)
            # gen_labels = torch.randint(0, n_class, (batch_size,)).to(device)
            with torch.no_grad():
                fake_images = generator(noises, labels)
            false_truth = torch.zeros(batch_size,1).to(device)
            discriminator_prediction_on_fake = discriminator(fake_images,labels)
            d_loss_2 = F.binary_cross_entropy(discriminator_prediction_on_fake, false_truth)
            
            # train
            d_loss = (d_loss_1 + d_loss_2)/2
            d_loss.backward()
            discriminator_optimizer.step()
            sum_d_loss += d_loss

            ### ------ Step -3.2 - Train Generator
            for _ in range(n_repeat_generator):
                generator.train()
                generator_optimizer.zero_grad()

                new_noises = torch.randn(batch_size, z_dim).to(device)
                generative_images = generator(new_noises, labels)  
                
                discriminartor_predictions = discriminator(generative_images, labels)
                false_truth = torch.ones(batch_size,1).to(device)
               
                g_loss = F.binary_cross_entropy(discriminartor_predictions, false_truth)
                g_loss.backward()
                generator_optimizer.step()
                sum_g_loss += g_loss


        epoch_d_loss = sum_d_loss.item()/len_data
        epoch_g_loss = sum_g_loss.item()/(n_repeat_generator*len_data)
        print(f"at epoch {epoch}")
        print(f"\td_loss:{epoch_d_loss}")
        print(f"\tg_loss: {epoch_g_loss}")   
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

        
        if epoch_g_loss > loss_threshold: 
            n_repeat_generator = n_repeat_generator+1
            n_repeat_generator = min(n_repeat_generator, 5)
            print(f"t - {n_repeat_generator}")
        ### generate some samples 
        if epoch % freq_gen_sample == 0:
            my_F.gans_save_samples(generator, epoch=epoch)
            my_F.plot_list(d_losses, g_losses, labels=["Discriminator losses", "Generator losses"], model_name="gans")

    ### save parameters
    torch.save(generator.state_dict(),f"{parameters_load_path+model_name}_G.pth")
    torch.save(discriminator.state_dict(),f"{parameters_load_path+model_name}_D.pth")

