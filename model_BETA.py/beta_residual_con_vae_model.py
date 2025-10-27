import torch.nn as nn
import torch
import torch.nn.functional as F

class BETA_RESIDUAL_CON_VAE(nn.Module):
    def __init__(self,z_dim):
        super().__init__()
        # Encoder implementation
        self.encoder_residual_layer = nn.Sequential(
            nn.Linear(3*32*32,1024*1*1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # output 1024*1*1
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )      
        self.encoder_combine_layer = nn.Sequential(
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.encoder_mu = nn.Linear(1024*1*1, z_dim)
        self.encoder_log_var = nn.Linear(1024*1*1, z_dim)
        
        # Decoder implementation
        self.hidden_layer_decoder = nn.Linear(z_dim, 1024*1*1)

        self.decoder_residual_layer = nn.Sequential(
            nn.Linear(1024*1*1, 3*32*32),
            nn.BatchNorm1d(3*32*32),
            nn.LeakyReLU(0.2),

        )
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.decoder_combine_layer = nn.Sequential(
            nn.Linear(3*32*32,3*32*32),
            nn.Sigmoid()
        )
    def forward(self, x):
        # encoder outputs
        short_cut = self.encoder_residual_layer(x.view(-1,3*32*32))
        encoder_out = self.encoder(x)
        encoder_out = encoder_out + short_cut
        encoder_out = self.encoder_combine_layer(encoder_out)
        
        u = self.encoder_mu(encoder_out)
        log_var = self.encoder_log_var(encoder_out)
        
        # decoder outputs
        random_noise = self.__reparameterizationT(u,log_var)
        out = self.hidden_layer_decoder(random_noise)
        short_cut = self.decoder_residual_layer(out)
        out = out.view(-1,1024, 1, 1)
        out = self.decoder(out)
        out = out + short_cut
        out = self.decoder_combine_layer(out)
        # results
        return u, log_var, out

    def generate_images(self, x):
        out = self.hidden_layer_decoder(x)
        short_cut = self.decoder_residual_layer(out)
        out = out.view(-1,1024, 1, 1)
        out = self.decoder(out)
        out = out + short_cut
        out = self.decoder_combine_layer(out)
        return out.view(-1,3,32,32)

    def __reparameterizationT(self,u, log_var):
        s = torch.exp(0.5 * log_var)
        return u + s * torch.randn_like(s)
            
