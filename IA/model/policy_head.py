import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, filters, output_dim=80, mapping_dim=1858, bias=True, kernel_size=3, stride=1, padding=1):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(filters, output_dim, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mapping_dim = mapping_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        batch_size, _, _, _ = x.shape 
        x = x.view(batch_size, -1)   # flatten x pour obtenir un tenseur de 80*8*8

        x = x[:, :self.mapping_dim]    # on ne prend que les 1858e premières valeurs --> moves pertinents/légaux
 
        return x
              
# output.shape = [batch_size, 1858]
    
# filters = 128
# pol_head = PolicyHead(filters)

# batch_size = 16
# input_channels = 128
# x = torch.randn(batch_size, input_channels, 8, 8) 
# output = pol_head(x)

# print(output.shape)
