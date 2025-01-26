import chess
from datasets import load_dataset
import torch

import json
with open("mapping.json", "r") as json_file:
    new_mapping = json.load(json_file)

def create_target_vector(move, mapping):
    move = str(move)
    target = torch.zeros(len(mapping), dtype=torch.float32)
    move_idx = mapping.get(move, -1)
    if move_idx != -1:
        target[move_idx] = 1.0  # Set the correct move index to 1
    return target
 
########################################################

def get_value_target(winner):
    if winner == "white":
        return [1, 0, 0]
    if winner == "black":
        return [0, 0, 1]
    else :
        return [0, 1, 0]
    
