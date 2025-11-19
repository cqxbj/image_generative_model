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
    It is very difficult to find balanced training parameters for Gernerators and Discriminators.
'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------------------------------
### step - 1  load model and data

model_name = "gans" 
z_dim = 256
n_class = 63
parameters_load_path ="trained_parameters/" 
generator, discriminator = my_F.load_gans_model(model_name, z_dim=z_dim, n_class=n_class, device=device)


data_loader = dg.generate_Handwriting_dataloader(less_space=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------

### step - 2  hyper_parameters
epoches = 50
lr_d = 0.0002
lr_g = 0.0002

generator_strategy = True
n_repeat_generator = 3
loss_threshold = 1.5


discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=lr_d)
generator_optimizer = optim.AdamW(generator.parameters(), lr=lr_g)

freq_gen_sample = 1
freq_save_model = 10
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

            original_labels = labels
            original_batch_size = batch_size
            

            # make the number of image_space double. make D to be great on discriminating space/blank img.
            blank_mask = (labels == 62)
        
            if blank_mask.sum() > 0:
                blank_images = real_images[blank_mask]
                blank_labels = labels[blank_mask]

                real_images = torch.cat([real_images, blank_images], dim=0)
                labels = torch.cat([labels, blank_labels], dim=0)

                batch_size = len(real_images)

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
            true_positive = torch.ones(batch_size,1).to(device)
            discriminator_prediction = discriminator(real_images, labels)
            d_loss_1 = F.binary_cross_entropy(discriminator_prediction, true_positive)
            
            # on a fake batch
            noises = torch.randn(batch_size, z_dim).to(device)
            # gen_labels = torch.randint(0, n_class, (batch_size,)).to(device)
            with torch.no_grad():
                fake_images = generator(noises, labels)
            false_positive = torch.zeros(batch_size,1).to(device)
            discriminator_prediction_on_fake = discriminator(fake_images,labels)
            d_loss_2 = F.binary_cross_entropy(discriminator_prediction_on_fake, false_positive)
            
            # train
            d_loss = (d_loss_1 + d_loss_2)/2
            d_loss.backward()
            discriminator_optimizer.step()
            sum_d_loss += d_loss





            ### ------ Step -3.2 - Train Generator
            for _ in range(n_repeat_generator):
                generator.train()
                generator_optimizer.zero_grad()

                new_noises = torch.randn(original_batch_size, z_dim).to(device)
                generative_images = generator(new_noises, original_labels)  
                
                discriminartor_predictions = discriminator(generative_images, original_labels)
                false_positive = torch.ones(original_batch_size,1).to(device)
               
                g_loss = F.binary_cross_entropy(discriminartor_predictions, false_positive)
                g_loss.backward()
                generator_optimizer.step()
                sum_g_loss += g_loss


        epoch_d_loss = sum_d_loss.item()/(len_data + 250)
        epoch_g_loss = sum_g_loss.item()/(n_repeat_generator*len_data)
        print(f"at epoch {epoch}")
        print(f"\td_loss:{epoch_d_loss}")
        print(f"\tg_loss: {epoch_g_loss}")   
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)


        #generator strategy
        if epoch >0 and generator_strategy:
            if epoch_g_loss > loss_threshold and g_losses[-1] > g_losses[-2]: 
                n_repeat_generator = n_repeat_generator+1
                n_repeat_generator = min(n_repeat_generator, 6)
                print(f"t - {n_repeat_generator}")
            
        
        
        ### generate some samples 
        if epoch % freq_gen_sample == 0:
            my_F.gans_save_samples(generator, epoch=epoch, input_str="AAAA BBBB We all like COMP 7015")
            my_F.plot_list(d_losses, g_losses, labels=["Discriminator losses", "Generator losses"], model_name="gans")
        
        if epoch % freq_save_model == 0 and epoch > 0:
            torch.save(generator.state_dict(),f"{parameters_load_path+model_name+str(epoch)}_G.pth")
            torch.save(discriminator.state_dict(),f"{parameters_load_path+model_name+str(epoch)}_D.pth")


    ### save parameters
    torch.save(generator.state_dict(),f"{parameters_load_path+model_name}_G.pth")
    torch.save(discriminator.state_dict(),f"{parameters_load_path+model_name}_D.pth")

