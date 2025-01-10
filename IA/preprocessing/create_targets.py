import chess
from datasets import load_dataset
import torch

def create_target_vector(move, mapping):
    target = torch.zeros(len(mapping), dtype=torch.float32)
    move_idx = mapping.get(move, -1)
    if move_idx != -1:
        target[move_idx] = 1.0  # Set the correct move index to 1
    return target
 
########################################################

# CA C'EST BON, en partant du principe qu'on dit au robot que la partie est gagnée quand c'est white qui gagne et perdue quand c'est black qui gagne

def get_value_target(winner):
    if winner == "white":
        return [1, 0, 0]
    if winner == "black":
        return [0, 0, 1]
    else :
        return [0, 1, 0]
    


# def generate_policy_mapping():
#     policy_map = {}
#     index = 0
    
#     # Iterate over all source squares
#     for from_square in range(64):
#         for to_square in range(64):
#             # Skip illegal moves (like moving a piece off the board)
#             if from_square == to_square:
#                 continue
            
#             # Standard moves
#             move_uci = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
#             policy_map[move_uci] = index
#             index += 1
            
#             # Add promotion moves
#             if chess.square_rank(from_square) in [1, 6]:  # If it's a pawn's promotion rank
#                 for promo in "qrbn":
#                     promo_move_uci = f"{chess.square_name(from_square)}{chess.square_name(to_square)}{promo}"
#                     policy_map[promo_move_uci] = index
#                     index += 1

#     return policy_map
# print("POLICY MAP")
# policy_map = generate_policy_mapping()
# #print(policy_map)