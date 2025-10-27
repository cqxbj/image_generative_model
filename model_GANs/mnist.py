import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.optim as optim

# For MNIST , image size 1*28*28
class Conv_GANS_Generator(nn.Module):
    def __init__(self, z_dim = 128):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        self.generator =  nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
           
        )
  
    def forward(self, z_noise):
        hidden_input = self.linear(z_noise)
        t_conv_input = hidden_input.view(-1,128,7,7)
        out = self.generator(t_conv_input)
        return out

class Conv_GANs_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            # 28*28
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 14*14
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 7 * 7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),

            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.nn(image)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return x + self.block(x)