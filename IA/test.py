import torch
from model.Model import Model
from preprocessing.mapping import new_mapping
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

def test_model(model_path, test_dataset, mapping):
    """Test the loaded model on a sample dataset."""

    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return

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
        batch_size=8,  # Example value; adjust as needed
        collate_fn=collate_fn,
        shuffle=False
    )

    print("Testing the model...")
    with torch.no_grad():
        for i, (board_tensors, target_vectors, value_targets, move_indices, tot_moves) in enumerate(test_loader):
            board_tensors = board_tensors.float().to(device)

            # Forward pass
            outputs, value_preds = model(board_tensors)

            print(f"Batch {i + 1}:")
            print(f"Predicted Outputs Shape: {outputs.shape}")
            print(f"Predicted Value Outputs Shape: {value_preds.shape}")

            # Only process the first batch for testing
            break

if __name__ == "__main__":
    model_path = "model.pth"  # Path to the saved model

    # Load the test dataset (replace with actual dataset path or loader)
    from datasets import load_dataset
    test_dataset = load_dataset('angeluriot/chess_games')

    # Perform the test
    test_model(model_path, test_dataset, new_mapping)
