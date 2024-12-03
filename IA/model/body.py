import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """
    SE Layer
    inputs= blabla
    outputs= blabla
    """
    def __init__(self, filters, se_channels):
        super(SELayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(filters, se_channels)  
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(se_channels, 2 * filters)

    def forward(self, x):
        batch_size, filters, _, _ = x.size()

        out = self.global_avg_pool(x)

        out = out.view(batch_size, filters) 

        out = self.fc1(out)
        out = self.relu(out) 

        out = self.fc2(out) 

        w, b = torch.split(out, filters, dim=1) 

        out = torch.sigmoid(w) 

        return x * out.view(batch_size, filters, 1, 1) + b.view(batch_size, filters, 1, 1)


class ResidualBlock(nn.Module):
    """
    Residual tower
    inputs
    outputs
    """
    def __init__(self, filters, se_channels=None, bias=True, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.se_channels = se_channels
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias) 
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias) 
        self.se_layer = SELayer(filters, se_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out) 

        out = self.conv2(out)
        out = self.relu(out) 

        if self.se_channels:  
            out = self.se_layer(out)

        out += x 
        out = self.relu(out)
        
        return out


class Body(nn.Module):
    """
    inputs
    outputs
    """
    def __init__(self, input_channels, filters, blocks, se_channels=None, bias=True, kernel_size=3, stride=1, padding=1):
        super(Body, self).__init__()
        self.conv = nn.Conv2d(input_channels, filters, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(filters, se_channels, bias, kernel_size, stride, padding) for _ in range(blocks)]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x) 
        x = self.residual_blocks(x) 
        return x
    
# input_channels = 112 
# filters = 128  
# blocks = 10
# se_channels = 32
# batch_size=16

# body = Body(input_channels, filters, blocks, se_channels)
# x = torch.randn(batch_size, input_channels, 8, 8) 
# out = body(x)
# print("OUT final",out.shape)