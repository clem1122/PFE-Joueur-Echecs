import chess
import chess.pgn
import torch
import numpy as np

from datasets import load_dataset
from create_targets import create_target_vector
from mapping import mapping
compact_mapping = mapping
# Load dataset
dataset = load_dataset('angeluriot/chess_games')

# Initialize the model (example, adjust to your actual model setup)
input_channels = 112  # Example input_channels, adjust based on your model
filters = 128
blocks = 10

# Process the dataset and convert UCI moves to board
for game in dataset['train']:
    moves_uci = game['moves_uci']
    
    # Initialize a chess board
    board = chess.Board()

    board_tensor = torch.zeros((input_channels, 8, 8), dtype=torch.float32)
    
    # Apply each move to the board
    for move_uci in moves_uci:
        move = chess.Move.from_uci(move_uci)  # Convert UCI move to chess move
        if move in board.legal_moves:  # Ensure move is legal
            board.push(move)  # Apply the move to the board
            target_vector = create_target_vector(move_uci, compact_mapping)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Encode the piece type and color (this depends on your model's encoding scheme)
                piece_type = piece.piece_type - 1  # 0-indexed
                piece_color = piece.color  # 0 for white, 1 for black
                board_tensor[piece_type + 6 * piece_color, square // 8, square % 8] = 1  # Set the corresponding tensor element to 1
    
    # Now `board` contains the final position after all moves in the game
    # You need to convert the board into a format your model can accept

    # Example conversion (you may need to adjust this based on your model's input):
    # Convert the board to a tensor (here we assume 8x8x12 encoding for the pieces)
    
    
    
    
    # # Now `board_tensor` is a tensor representation of the board state
    # # Pass this to your model for predictions (policy and value heads)
    # policy_output, value_output = model(board_tensor.unsqueeze(0))  # Assuming batch size of 1

    # print("Policy Output:", policy_output)
    # print("Value Output:", value_output)

    break  # Just process one game for testing

print(board_tensor)
print(board)