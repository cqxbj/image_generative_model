import torch.nn as nn
import torch
import torch.nn.functional as F

class RESIDUAL_CON_VAE(nn.Module):
    def __init__(self,z_dim):
        super().__init__()
        # Encoder implementation
        self.encoder = nn.Sequential(
            RESIDUAL_BLOCK(3,128),
            RESIDUAL_BLOCK(128,512),
            # output 1024*1*1
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),# 1*1
            nn.Flatten()
        )
        self.encoder_mu = nn.Linear(1024*1*1, z_dim)
        self.encoder_log_var = nn.Linear(1024*1*1, z_dim)
        # Decoder implementation
        self.hidden_layer_decoder = nn.Linear(z_dim, 1024*1*1)
        self.decoder =  nn.Sequential(
            TRANSPOSE_RESIDUAL_BLOCK(1024,256),
            TRANSPOSE_RESIDUAL_BLOCK(256,64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, x):
        # encoder outputs
        last_hidden_out = self.encoder(x)
        u = self.encoder_mu(last_hidden_out)
        log_var = self.encoder_log_var(last_hidden_out)
        
        # decoder outputs
        random_noise = self.__reparameterizationT(u,log_var)
        out = self.hidden_layer_decoder(random_noise)
        out = out.view(-1,1024, 1, 1)
        out = self.decoder(out)
        # results
        return u, log_var, out

    def generate_images(self, x):
        out = self.hidden_layer_decoder(x)
        out = out.view(-1,1024, 1, 1)
        out = self.decoder(out)
        return out.view(-1,3,32,32)

    def __reparameterizationT(self,u, log_var):
        s = torch.exp(0.5 * log_var)
        return u + s * torch.randn_like(s)

# kernel_size=3, stride=2, padding=1           
class RESIDUAL_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(out_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
        )
  
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 4, 0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(0.2,True)

    def forward(self, x):
        out = self.conv_layers(x)
        short_cut = self.short_cut(x)
        out += short_cut
        out = self.relu(out)
        return out
    
class TRANSPOSE_RESIDUAL_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):

        # for the transpose conv ： 
        # out_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,True),

            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.short_cut = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False), 
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.LeakyReLU(0.2,True)

    def forward(self, x):
        out = self.conv_layers(x)
        short_cut = self.short_cut(x)
        out += short_cut
        out = self.relu(out)
        return out