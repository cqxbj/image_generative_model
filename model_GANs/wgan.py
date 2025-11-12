import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, z_dim = 128):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 512 * 4 * 4),
            nn.LayerNorm(512 * 4 * 4),
            nn.LeakyReLU(True)
        )

        self.generator =  nn.Sequential(
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            
        )
  
    def forward(self, z_noise):
        hidden_input = self.linear(z_noise)
        t_conv_input = hidden_input.view(-1,512,4,4)
        out = self.generator(t_conv_input)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Flatten(),
            nn.Linear(512, 1),
            # nn.Sigmoid()   
        )

    def forward(self, image):
        return self.nn(image)
