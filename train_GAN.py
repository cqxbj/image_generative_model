import torch
import torch.nn.functional as F
import dataloader_generator as dg
import torch
import torch.optim as optim

from model_GANs.hand_written import Discriminator as HandW_Discriminator, Generator as HandW_Generator 
from model_GANs.cifar import Conv_GANs_Discriminator as Cifar_Discriminator, Conv_GANS_Generator as Cifar_Generator 
from model_GANs.mnist import Conv_GANs_Discriminator as Mnist_Discriminator, Conv_GANS_Generator as Mnist_Generator
import torchvision.utils as utils
import matplotlib.pyplot as plt 

import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' 

    this .py is implemented for training GANs_based models.

'''
# input here.  
#  1. CIFAR  2.MNIST  3.EMNIST 4LT10
m = "CIFAR" 
parameters_load_path ="trained_parameters/" 
# ----------------------------------------------------------------------------------------------------------------------------------------------
### step - 1 create model and load parameters if exists
if m == "CIFAR":
    z_dim = 100
    data_loader = dg.generate_CIFAR_10_dataloader()
    generator = Cifar_Generator(z_dim).to(device)
    discriminator = Cifar_Discriminator().to(device) 
elif m == "HandW":
    z_dim = 128
    data_loader = dg.generate_Handwritten_dataloader()
    generator = HandW_Generator(z_dim).to(device)
    discriminator = HandW_Discriminator().to(device)
else:
    print("让你随便写,随便写就没东西给你了")
    print("byebye")
    sys.exit()

    
# ----------------------------------------------------------------------------------------------------------------------------------------------

### step - 2 training — config hyper_parameters
epoches = 200
lr_d = 0.0002
lr_g = 0.0003
t = 1
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

freq_gen_sample = 1
freq_save_model = 20
n_sample = 16
# ----------------------------------------------------------------------------------------------------------------------------------------------
### step -3 GANs training_function implementation.
start_training = False

if start_training:
    len_data = len(data_loader)
    print(f"len_data: {len_data}")
    num_param_D =  sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    num_param_G =  sum(p.numel() for p in generator.parameters() if p.requires_grad)

    print(f"num param D :{num_param_D}")
    print(f"num param G :{num_param_G}")

    for epoch in range(epoches + 1):
        sum_g_loss = 0
        sum_d_loss = 0
        for x, _ in data_loader:
            x = x.to(device)
            batch_size = len(x)
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
            real_labels = torch.ones(batch_size,1).to(device)
            real_images = x
            real_images_prediction = discriminator(real_images)
            d_loss_1 = F.binary_cross_entropy(real_images_prediction, real_labels)
            # on a fake batch
            noises = torch.randn(batch_size, z_dim).to(device)
            with torch.no_grad():
                fake_images = generator(noises).to(device)
            fake_labels = torch.zeros(batch_size,1).to(device)
            fake_images_prediction = discriminator(fake_images)
            d_loss_2 = F.binary_cross_entropy(fake_images_prediction, fake_labels)
            # train
            d_loss = (d_loss_1 + d_loss_2)/2
            d_loss.backward()
            discriminator_optimizer.step()
            sum_d_loss += d_loss

            ### ------ Step -3.2 - Train Generator
          
            generator.train()
            generator_optimizer.zero_grad()
            new_noises = torch.randn(batch_size, z_dim).to(device)
            generative_images = generator(new_noises)  
            discriminartor_predictions = discriminator(generative_images)
            generative_labels = torch.ones(batch_size,1).to(device)
            g_loss = F.binary_cross_entropy(discriminartor_predictions, generative_labels)
            g_loss.backward()
            generator_optimizer.step()
            sum_g_loss += g_loss


        epoch_d_loss = sum_d_loss/len_data
        epoch_g_loss = sum_g_loss/(t*len_data)
        print(f"at epoch {epoch}")
        print(f"\td_loss:{epoch_d_loss}")
        print(f"\tg_loss: {epoch_g_loss}")   

        # if epoch_g_loss > 1.25: 
        #     t = t*2
        #     print(f"t - {t}")
        ### generate some samples 
        if epoch % freq_gen_sample == 0:
            with torch.no_grad():
                noises = torch.randn(n_sample, z_dim).to(device)
                images = generator(noises)
                images = images.to("cpu")

            grid_image = utils.make_grid(images, nrow=4, normalize=True)
            plt.imshow(grid_image.permute(1,2,0))
            plt.axis("off")
            plt.savefig(f"ai_images/{m}_GANs_{epoch}")
            plt.close()

        ### save parameters
        if epoch % freq_save_model == 0:
            torch.save(generator.state_dict(),f"{parameters_load_path+m}_GANs_G.pth")
            torch.save(discriminator.state_dict(),f"{parameters_load_path+m}_GANs_D.pth")

