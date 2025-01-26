import os
import yaml
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Model import Model
from model.Loss import Loss
from preprocessing.dataloader import ChessDataset, collate_fn
from datasets import load_dataset
# import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6"

import json
with open("mapping.json", "r") as json_file:
    new_mapping = json.load(json_file)

from torch.utils.data import Subset
import random

# Load configuration
CONFIGROOT = "cfg"
learning_configs = 'config.yaml'
cfgs = os.path.join(CONFIGROOT, learning_configs)

with open(cfgs, "r") as f:
    config = yaml.safe_load(f)

# Training class
class Training:
    def __init__(self, config, device):
        config = config["train"]["tune"]
        self.device = device
        self.model = Model(
            input_channels=config["input_channels"], 
            filters=config["filters"], 
            blocks=config["blocks"], 
            se_channels=config["se_channels"], 
            bias=config["bias"], 
            kernel_size=config["kernel_size"], 
            stride=config["stride"], 
            padding=config["padding"], 
            output_dim=config["output_dim"], 
            mapping_dim=config["mapping_dim"], 
            valout1=config["valout1"], 
            valout2=config["valout2"]
        ).to(device)
        
        self.num_epochs = config["num_epochs"]
        self.current_epoch = 0  # Start from 0
        self.criterion = Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.batch_size = config["batch_size"]

        # Variable to track the best accuracy
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_path = 'best_model_checkpoint.pth'

        # Load model checkpoint if exists
        self.model_path = 'model_checkpoint.pth'
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint["epoch"]
            print(f"Checkpoint loaded. Resuming training from epoch {self.current_epoch + 1}.")

    def accuracy(self, outputs, targets, value_preds, value_targets):
        _, predicted_indices = torch.max(outputs, 1)  # Shape: [batch_size]
        _, predicted_values = torch.max(value_preds, 1)
    
        true_indices = torch.argmax(targets, 1)  # Shape: [batch_size]
        true_values = torch.argmax(value_targets, 1) 
        
        correct = (predicted_indices == true_indices).sum().item()
        correct_values = (predicted_values == true_values).sum().item()
        
        total = targets.size(0)
        total_values = value_targets.size(0)
       
        return correct / total, correct_values / total_values
    
    def train(self, dataset, mapping, fraction=0.05):
        chess_dataset = ChessDataset(dataset, mapping, fraction=fraction)
        train_loader = DataLoader(chess_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)

        for epoch in range(self.current_epoch + 1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            for i, (board_tensors, target_vectors, value_targets, move_indices, tot_moves, board_fens) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")
            ):
                board_tensors = board_tensors.float().to(self.device)
                target_vectors = target_vectors.float().to(self.device)
                value_targets = value_targets.float().to(self.device)

                self.optimizer.zero_grad()
                
                outputs, value_preds = self.model(board_tensors)
                outputs = outputs.float()
                value_preds = value_preds.float()
                
                loss = self.criterion(board_fens, outputs, target_vectors, value_preds, value_targets, move_indices, tot_moves)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_accuracy += self.accuracy(outputs, target_vectors, value_preds, value_targets)[0]

            avg_accuracy = running_accuracy / len(train_loader)
            print(f"[Epoch {epoch}] Loss: {running_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

            # If the current model's accuracy is better than the best accuracy, save the model
            if avg_accuracy > self.best_accuracy:
                self.best_accuracy = avg_accuracy
                self.best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "accuracy": avg_accuracy
                }, self.best_model_path)
                print(f"New best model found at epoch {epoch} with accuracy {avg_accuracy:.4f}. Model saved.")

            # Save checkpoint after every epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, self.model_path)

        print(f"Training finished. Best model saved at epoch {self.best_epoch} with accuracy {self.best_accuracy:.4f}")


def main():
    dataset = load_dataset('angeluriot/chess_games')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    print(device)
    trainer = Training(config, device)
    
    trainer.train(dataset, new_mapping, fraction=0.1)

if __name__ == "__main__":
    main()