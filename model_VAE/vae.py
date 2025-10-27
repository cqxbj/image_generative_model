import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dim = 100, z_dim = 10):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.outlayer_mu = nn.Linear(hidden_dim, z_dim)
        self.outlayer_log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        x = self.hidden_layer(x)
        x = F.relu(x)
        
        u = self.outlayer_mu(x)
        log_var = self.outlayer_log_var(x)
        sigma = torch.exp(0.5 * log_var)
        return u, sigma

class Decoder(nn.Module):
    def __init__(self, input_z_dim = 10, hidden_dim = 100, output_dim = 28*28):
        super().__init__()
        self.hidden_layer = nn.Linear(input_z_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)
        x =  self.out_layer(x)
        return torch.sigmoid(x)
     
class VAE(nn.Module):
    def __init__(self, image_flatten_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(image_flatten_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, image_flatten_dim)

    def forward(self, x):
        u, sigma = self.encoder(x)
        out = self.decoder(self.reparameterizationT(u, sigma))
        return u, sigma, out

    def reparameterizationT(self,u, s):
        return u + s * torch.randn_like(s)
    
        
