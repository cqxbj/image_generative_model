import torch.nn as nn
import torch
import torch.nn.functional as F

# image size 1*32*32
class Conv_VAE(nn.Module):
    def __init__(self,z_dim = 128, n_class = 40, name = "hw_VAE"):
        super().__init__()
        self.z_dim = 128
        self.n_class = n_class
        self.label_embedding = nn.Embedding(self.n_class, 16)
        # Encoder implementation
        self.encoder =  nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),   # -> 16*16

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),   # -> 8*8

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),     #  -> 8*8

            nn.Flatten(),     # 8*8
        )
        self.encoder_mu = nn.Linear(64 * 8 * 8, z_dim)
        self.encoder_log_var = nn.Linear(64 * 8 * 8, z_dim)
        self.name = name
        # Decoder implementation
        self.hidden_layer_decoder = nn.Sequential(
            nn.Linear(z_dim + 16, 64 * 8 * 8), 
            nn.BatchNorm1d(64 * 8 * 8),
            nn.ReLU()
        )
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x, labels):
        # encoder outputs
        last_hidden_out = self.encoder(x)
        u = self.encoder_mu(last_hidden_out)
        log_var = self.encoder_log_var(last_hidden_out)
        
        # decoder outputs
        random_noise = self.__reparameterizationT(u,log_var)
        label_embedding = self.label_embedding(labels)
        out = self.hidden_layer_decoder(torch.cat([random_noise,label_embedding],dim = 1))
        out = out.view(-1,64, 8, 8)
        out = self.decoder(out)
    
        # results
        return u, log_var, out

    def generate_images(self, z, labels):
        # decoder_output 
        label_embedding = self.label_embedding(labels)
        out = self.hidden_layer_decoder(torch.cat([z,label_embedding],dim = 1))
        out = out.view(-1,64, 8, 8)
        return self.decoder(out)

    def __reparameterizationT(self,u, log_var):
        s = torch.exp(0.5 * log_var)
        return u + s * torch.randn_like(s)
            
