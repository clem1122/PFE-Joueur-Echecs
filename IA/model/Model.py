import torch
import torch.nn as nn
from model.body import Body
from model.policy_head import PolicyHead
from model.value_head import ValueHead

class Model(nn.Module):
    def __init__(self, input_channels, filters, blocks, se_channels=32, bias=True, kernel_size=3, stride=1, padding=1, output_dim=80, mapping_dim=1976, valout1=32, valout2=128):
        super(Model, self).__init__()
        self.body = Body(input_channels, filters, blocks, se_channels=se_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.policy_head = PolicyHead(filters, output_dim=output_dim, mapping_dim=mapping_dim, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.value_head = ValueHead(filters, out1=valout1, out2=valout2, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.body(x)

        out_policy = self.policy_head(x)
        out_value = self.value_head(x)

        return out_policy, out_value
