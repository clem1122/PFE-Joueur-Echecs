import torch
from model.Model import Model
import json
with open("mapping.json", "r") as json_file:
    new_mapping = json.load(json_file)
from preprocessing.dataloader import ChessDataset, collate_fn
from torch.utils.data import DataLoader
import os
import yaml

# Load configuration
CONFIGROOT = "cfg"
learning_configs = 'config.yaml'
cfgs = os.path.join(CONFIGROOT, learning_configs)

with open(cfgs, "r") as f:
    config = yaml.safe_load(f)


def accuracy(outputs, targets, value_preds, value_targets):

        _, predicted_indices = torch.max(outputs, 1)  # Shape: [batch_size]
        _, predicted_values = torch.max(value_preds, 1)
    
        true_indices = torch.argmax(targets, 1)  # Shape: [batch_size]
        true_values = torch.argmax(value_targets, 1) 
        
        correct = (predicted_indices == true_indices).sum().item()
        correct_values = (predicted_values == true_values).sum().item()
        
        total = targets.size(0)
        total_values = value_targets.size(0)
       
        return correct / total, correct_values / total_values

def test_model(model_path, test_dataset, mapping, config):

    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return
    
    config = config["train"]["tune"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
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

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare the test data loader
    test_loader = DataLoader(
        ChessDataset(test_dataset, mapping, fraction=0.05),
        batch_size=32, 
        collate_fn=collate_fn,
        shuffle=False
    )

    print("Testing the model...")
    with torch.no_grad():
        for i, (board_tensors, target_vectors, value_targets, move_indices, tot_moves, fens) in enumerate(test_loader):
            board_tensors = board_tensors.float().to(device)

            outputs, value_preds = model(board_tensors)

            print(f"Batch {i + 1}:")
            print(f"Predicted Outputs Shape: {outputs.shape}")
            print(f"Predicted Value Outputs Shape: {value_preds.shape}")
            vec_acc, val_acc = accuracy(outputs, target_vectors, value_preds, value_targets)

            print("accuracy", vec_acc, val_acc)

            

if __name__ == "__main__":
    model_path = "model_checkpoint.pth" 

    from datasets import load_dataset
    test_dataset = load_dataset('angeluriot/chess_games')

    test_model(model_path, test_dataset, new_mapping, config)
