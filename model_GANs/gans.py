import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.optim as optim

# For MNIST , image size 1*32*32
class Generator(nn.Module):
    def __init__(self, z_dim = 128, n_class = 40):
        super().__init__()
        self.z_dim = z_dim
        self.n_class = n_class

        self.label_embedding = nn.Embedding(self.n_class, 16)

        self.linear = nn.Sequential(
            nn.Linear(z_dim + 16, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.ReLU()
        )

        self.generator =  nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid() 

        )

    def forward(self, z_noise, labels):
        label_embedding = self.label_embedding(labels)
        hidden_input = self.linear(torch.concat([z_noise, label_embedding],dim=1))
        t_conv_input = hidden_input.view(-1,64,8,8)
        out = self.generator(t_conv_input)
        return out

# class Discriminator(nn.Module):
#     def __init__(self, n_class = 40, label_dim = 128):
#         super().__init__()
#         self.n_class = n_class
#         self.label_embedding = nn.Embedding(n_class, label_dim)

#         self.nn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),      

#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Flatten(),  
#             nn.Linear(64*8*8, label_dim),
#             nn.ReLU()

#         )

#         self.classifier = nn.Sequential(
#                 nn.Linear(label_dim+ label_dim, 1),
#                 nn.Sigmoid()
#         )
        
#     def forward(self, image, labels):
#         labels = self.label_embedding(labels) 
#         # -> [batch, 128]
        
#         features = self.nn(image)
#         # -> [batch, 64*8*8]
        
#         concat = torch.concat([features,labels], dim=1)

#         return self.classifier(concat)
class Discriminator(nn.Module):
    def __init__(self, n_class=37, label_dim=16):
        super().__init__()
        self.n_class = n_class
        self.label_embedding = nn.Embedding(n_class, label_dim)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),      
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),    
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 128),  
            nn.ReLU(),    
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + label_dim, 128),
            nn.ReLU(),    
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, labels):
        image_features = self.image_encoder(image)  # [batch, 256]
        label_emb = self.label_embedding(labels)    
        combined = torch.cat([image_features, label_emb], dim=1)
        return self.classifier(combined)