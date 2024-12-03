import torch
import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, filters, out1=32, out2=128, bias=True, kernel_size=3, stride=1, padding=1):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, out1, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out1, out2, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU() 
        self.fc = nn.Linear(out2 * 8 * 8, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)

        batch_size, _, _, _ = x.shape 
        x = x.view(batch_size, -1)

        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x
#ouput.shape = [batch_size,3], proba de win, draw et loss
    
# filters = 128
# val_head = ValueHead(filters)

# batch_size = 16
# input_channels = 128
# x = torch.randn(batch_size, input_channels, 8, 8) 
# output = val_head(x)

# print(output.shape)
