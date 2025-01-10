import chess
import chess.pgn
import torch
import numpy as np

from datasets import load_dataset
from create_targets import create_target_vector
from mapping import new_mapping
compact_mapping = new_mapping

dataset = load_dataset('angeluriot/chess_games')

input_channels = 112 
filters = 128
blocks = 10

for game in dataset['train']:
    moves_uci = game['moves_uci']
    total_moves = len(moves_uci)
    print(total_moves)
    board = chess.Board()

    board_tensor = torch.zeros((input_channels, 8, 8), dtype=torch.float32)
    
    for i, move_uci in enumerate(moves_uci):
        move = chess.Move.from_uci(move_uci)  # Convert UCI move to chess move 
        if move in board.legal_moves:  # Ensure move is legal
            board.push(move)  # Apply the move to the board
            target_vector = create_target_vector(move_uci, compact_mapping)
            print(target_vector)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    # Encode the piece type and color (this depends on your model's encoding scheme)
                    piece_type = piece.piece_type - 1  # 0-indexed
                    piece_color = piece.color  # 0 for white, 1 for black
                    board_tensor[piece_type + 6 * piece_color, square // 8, square % 8] = 1  # Set the corresponding tensor element to 1
            break
    
    break 

print(board_tensor)
print(board)