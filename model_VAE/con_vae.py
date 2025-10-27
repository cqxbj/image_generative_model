import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv_VAE(nn.Module):
    def __init__(self,z_dim):
        super().__init__()
        # Encoder implementation
        # I am testing if we need the BatchNorm here, cause I see sometimes model has better performance if it is trained without Batchnorm.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # output 1024*1*1
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),

        )
        self.encoder_mu = nn.Linear(1024*1*1, z_dim)
        self.encoder_log_var = nn.Linear(1024*1*1, z_dim)
        # Decoder implementation
        self.hidden_layer_decoder = nn.Linear(z_dim, 1024*1*1)
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
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
        # decoder_output 
        out = self.hidden_layer_decoder(x)
        out = out.view(-1,1024, 1, 1)
        out = self.decoder(out)
        return out.view(-1,3,32,32)

    def __reparameterizationT(self,u, log_var):
        s = torch.exp(0.5 * log_var)
        return u + s * torch.randn_like(s)
            
