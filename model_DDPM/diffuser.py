import torch
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# it can performs add_nosie and de_noise.
class Diffuser:
    def __init__(self, T = 1000, device = "cpu"):
        self.b_start = 0.0001
        self.b_end = 0.02
        self.T = T
        
        self.betas = torch.linspace(self.b_start, self.b_end, self.T,device = device)
        self.alps = 1 - self.betas
        self.alps_cumprod = torch.cumprod(self.alps, dim=0)

        self.device = device


    # according to the ddpm algorithm, actually we do not need to add_noise step by step
    # we can calculate a final T result in one step.
    # x_t_0 is the original img in tensor.
    def add_noise(self, x_t_0, t):
        if (t> self.T).all() or (t<1).all(): 
            print("here not add noise")
            return x_t_0
        
        t = t - 1
        alp_cum_t = self.alps_cumprod[t]

        # for batch calculation
        n = alp_cum_t.size(0)
        alp_cum_t = alp_cum_t.view(n,1,1,1)

        noise = torch.randn_like(x_t_0).to(self.device)
        return torch.sqrt(alp_cum_t)*x_t_0 + torch.sqrt(1-alp_cum_t)*noise, noise
    


    # when we do de_noise, we need to perform de_noise step by step. Unet is a noise predictor.
    def de_noise(self, x, t, model, labels = None):
        if (t>self.T).all() or (t<1).all(): 
            print("here not de noise")
            return 
        t_index = t - 1
        
        alp_t = self.alps[t_index]
        alp_cum_t = self.alps_cumprod[t_index]
        alp_cum_t_1 = self.alps_cumprod[t_index - 1]

        # for batch calculation
        n = alp_t.size(0)
        alp_t = alp_t.view(n,1,1,1)
        alp_cum_t = alp_cum_t.view(n,1,1,1)
        alp_cum_t_1 = alp_cum_t_1.view(n,1,1,1)

        # model result
        with torch.no_grad():
            if labels is not None:
                eps = model(x,t,labels)
            else:
                eps = model(x,t)
        
        # cal results
        noise = torch.randn_like(x).to(self.device)
        noise[t == 1] = 0
        # result_mu = (x - ((1-alp_t)/torch.sqrt(1-alp_cum_t))*eps)
        # result_mu = result_mu/(torch.sqrt(alp_t))

        result_mu = (x - ((1 - alp_t) / torch.sqrt(1 - alp_cum_t)) * eps) / torch.sqrt(alp_t)
        std = torch.sqrt((1-alp_t)*(1-alp_cum_t_1)/(1-alp_cum_t))
        return result_mu + std * noise


    # cifar_shape imgs in tensor
    # does not return image file,  returns image in tensor instead.
    def denoised_sampling(self, model, imgs_shape = (4,3,32,32), labels = None):
        n = imgs_shape[0]
        model.eval()
        if labels is not None : 
            labels = labels.to(self.device)
        x = torch.randn(imgs_shape).to(self.device)        
        for i in range(self.T, 0 , -1):
            t = torch.tensor([i]*n,dtype =torch.long).to(self.device)
            if labels is not None:
                x = self.de_noise(x,t,model,labels=labels)
            else:
                x = self.de_noise(x,t, model)
        model.train()
        return x





