import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, channels, k = 8):
        super().__init__()
        # self.channels = channels
        self.query = nn.Conv2d(channels, channels // k, 1)
        self.key = nn.Conv2d(channels, channels // k, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))  
        
    def forward(self, x):
        batch_size, C, H, W = x.shape

        q = self.query(x).view(batch_size, -1, H * W).transpose(1, 2)  # [B, HW, C//8]
        k = self.key(x).view(batch_size, -1, H * W)  # [B, C//8, HW]
        v = self.value(x).view(batch_size, -1, H * W)  # [B, C, HW]
        
        attention = torch.bmm(q, k)  # [B, HW, HW]
        attention = attention / (q.size(-1) ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, C, H, W)
        
        out = self.proj(out)
        return x + self.gamma * out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, residual_on = False):
        super().__init__()
        self.residual_on = residual_on

        #gn_groups = 8
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size= 3, padding = 1),
            #nn.GroupNorm(gn_groups,out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, kernel_size= 3 , padding = 1),
            #nn.GroupNorm(gn_groups,out_channel),
            nn.BatchNorm2d(out_channel),
        )

        if self.residual_on and in_channel != out_channel:
            self.residual_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        self.time_encoding_mlp = nn.Sequential(
            nn.Linear(time_dim, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel)   
        )

        self.relu = nn.ReLU()

    def forward(self, x, t):
        n, c, _, _ = x.shape
        #print(x.shape)
        v = self.time_encoding_mlp(t) 
        #print(v.shape)
        v = v.view(n,c,1,1)
        out = self.blocks(x+v)

        if self.residual_on:
            out = out + self.residual_conv(x)
        
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channel = 3, time_dim = 256, n_class = 0, attention = False, residual_on = False):
        super().__init__()
        
        self.time_dim = time_dim
        self.attention = attention
        self.residual_on = residual_on
        self.n_class = n_class

        if self.n_class > 0:
            self.label_embeding = nn.Embedding(self.n_class,  self.time_dim)

        if self.attention:
            # self.attention_down_1 = SelfAttention(64)
            # self.attention_down_2 = SelfAttention(128)
            self.attention_down_3 = SelfAttention(256)
            self.attention_down_4 = SelfAttention(512)
            self.attention_bottom = SelfAttention(512)
            self.attention_up_4 = SelfAttention(512)
            self.attention_up_3 = SelfAttention(256)
            # self.attention_up_2 = SelfAttention(128)
            # self.attention_up_1 = SelfAttention(64)

        # max_pool for downsampling 
        self.max_pool = nn.MaxPool2d(2)
        # upsamping
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # those conv_blocks need time_v 
        self.cov_down_1 = ConvBlock(in_channel, 64, time_dim,residual_on)
        self.con_down_2 = ConvBlock(64,128, time_dim,residual_on)
        self.con_down_3 = ConvBlock(128,256, time_dim,residual_on)
        self.con_down_4 = ConvBlock(256,512, time_dim,residual_on)

        self.bottom = ConvBlock(512, 512, time_dim,residual_on)

        self.cov_up_4 = ConvBlock(512 + 512, 512, time_dim,residual_on)
        self.cov_up_3 = ConvBlock(512 + 256, 256, time_dim,residual_on)
        self.cov_up_2 = ConvBlock(256 + 128, 128, time_dim,residual_on)
        self.cov_up_1 = ConvBlock(128 + 64, 64, time_dim,residual_on)

        # out
        self.output = nn.Conv2d(64,in_channel, kernel_size=1)

    def forward(self, x, t_steps, labels = None):
        v_time = pos_encoding(t_steps, self.time_dim, device=x.device).to(x.device)

        if labels is not None:
            v_time += self.label_embeding(labels).to()

        # left side
        x1 = self.cov_down_1(x, v_time)
        # if self.attention : x1 = self.attention_down_1(x1)
        x_max_down = self.max_pool(x1)

        x2 = self.con_down_2(x_max_down,v_time)
        # if self.attention : x2 = self.attention_down_2(x2)
        x2_max_down = self.max_pool(x2)

        x3 = self.con_down_3(x2_max_down,v_time)
        if self.attention : x3 = self.attention_down_3(x3)
        x3_max_down = self.max_pool(x3)

        x4 = self.con_down_4(x3_max_down,v_time)
        if self.attention : x4 = self.attention_down_4(x4) 
        # x4_max_down = self.max_pool(x4)
        x4_max_down = x4

        # bottom
        x_bottom_left = self.bottom(x4_max_down,v_time)
        if self.attention : x_bottom_left = self.attention_bottom(x_bottom_left)
        # x_bottom_right = self.upsample(x_bottom_left)
        x_bottom_right = x_bottom_left

        # right side 
        x_4_right = torch.cat([x_bottom_right, x4], dim = 1)
        x_4_right = self.cov_up_4(x_4_right, v_time)
        if self.attention : x_4_right = self.attention_up_4(x_4_right)
        x_4_upsample = self.upsample(x_4_right)

        x_3_right = torch.cat([x_4_upsample, x3], dim = 1)
        x_3_right = self.cov_up_3(x_3_right, v_time)
        if self.attention : x_3_right = self.attention_up_3(x_3_right)
        x_3_upsample = self.upsample(x_3_right)
        
        x_2_right = torch.cat([x_3_upsample, x2], dim = 1)
        x_2_right = self.cov_up_2(x_2_right,v_time)
        # if self.attention : x_2_right = self.attention_up_2(x_2_right)
        x_2_upsample = self.upsample(x_2_right)
        
        x_1_right = torch.cat([x_2_upsample, x1], dim = 1)
        x_1_right = self.cov_up_1(x_1_right, v_time)
        # if self.attention: x_1_right = self.attention_up_1(x_1_right)

        # out
        out = self.output(x_1_right)

        return out

def pos_encoding(ts, out_dim, device = "cpu"):

    def _encoding(t, out_dim, device = device):
        t, D = t, out_dim
        v = torch.zeros(D, device=device)
        i = torch.arange(0, D, device=device)
        div_term = torch.exp(i / D * math.log(10000))
        v[0::2] = torch.sin(t / div_term[0::2])
        v[1::2] = torch.cos(t / div_term[1::2])
        return v

    batch_size = len(ts)
    v = torch.zeros(batch_size, out_dim).to(device)
    for  i in range(batch_size):
        v[i] = _encoding(ts[i],out_dim, device)
    return v
