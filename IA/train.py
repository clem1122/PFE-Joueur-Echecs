import torch
import os
from torch import optim
from torch.utils.tensorboard import SummaryWriter 
import yaml
from datetime import datetime

from model.Model import Model
from model.Loss import Loss

CONFIGROOT = "cfg"

learning_configs = 'config.yaml'
cfgs = os.path.join(CONFIGROOT, learning_configs)
with open(cfgs, "r") as f:
    config = yaml.safe_load(f)

class Training():
    def __init__(self, config, device):
        self.device = device
        self.model = Model(
            input_channels=config.input_channels, 
            filters=config.filters, 
            blocks=config.blocks, 
            se_channels=config.se_channels, 
            bias=config.bias, 
            kernel_size=config.kernel_size, 
            stride=config.stride, 
            padding=config.padding, 
            output_dim=config.output_dim, 
            mapping_dim=config.mapping_dim, 
            valout1=config.valout1, 
            valout2=config.valout2
        )
        self.num_epochs = config.num_epochs

        self.criterion = Loss(config.alpha, config.beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.model_path = 'model_checkpoint.pth'
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

        self.writer = SummaryWriter()  #Added to start tensorboard data capture

    def accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0
            for i, x, targets in enumerate(dataset):      #DATASET ???
                self.optimizer.zero_grad()
                
                outputs = self.model(x)                   #x ???
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                accuracy = self.accuracy(outputs, targets)
                running_accuracy += accuracy
                
                if i % 10 == 9:
                    self.writer.add_scalar('Training Loss', running_loss / 10)
                    self.writer.add_scalar('Training Accuracy', running_accuracy / 10)
                    running_loss = 0.0
                    running_accuracy = 0.0

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss}")

        print("Training Finished")
        self.writer.close()

        if os.path.exists(self.model_path):
            os.rename(self.model_path, self.model_path + datetime.now().strftime("%Y%m%d%H%M%S") )

        torch.save(self.model.state_dict(), self.model_path)

# trainer = Training(config=config, device=device)
# trainer.train()

