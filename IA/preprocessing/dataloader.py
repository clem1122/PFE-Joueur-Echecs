import chess
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.utils.data import Subset
import random

from preprocessing.create_targets import create_target_vector, get_value_target

import json
with open("mapping.json", "r") as json_file:
    new_mapping = json.load(json_file)

class ChessDataset(Dataset):
    def __init__(self, dataset, compact_mapping, fraction=0.25):
        # Longueur totale de dataset['train']
        train_length = len(dataset['train'])

        # Nombre d'échantillons dans le sous-ensemble
        subset_length = int(train_length * fraction)

        # Indices aléatoires pour le sous-ensemble
        subset_indices = random.sample(range(train_length), subset_length)

        # Création du sous-ensemble
        self.dataset = Subset(dataset['train'], subset_indices)
        self.compact_mapping = compact_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Accéder directement au jeu correspondant dans le sous-ensemble
        game = self.dataset[idx]  # Subset gère l'indexation vers dataset['train']
        moves_uci = game['moves_uci']  # On extrait les moves
        winner = game['winner']  # On extrait le gagnant

        value_target = get_value_target(winner)  # On transforme le gagnant pour la loss

        # Initialisations
        board = chess.Board() 
        board_tensors = []
        target_vectors = []
        value_targets = []
        move_indices = []  
        board_fens = []

        for i, move_uci in enumerate(moves_uci):
            move = chess.Move.from_uci(move_uci)  # On transforme le move 

            if move in board.legal_moves:
                board_tensor = self._generate_board_tensor(board)  # On génère notre vecteur
                board_fens.append(board.fen())
                board.push(move)  # On joue le prochain coup
                
                target_vector = create_target_vector(move, self.compact_mapping)  # On génère nos targets grâce au prochain coup

                board_tensors.append(board_tensor)
                target_vectors.append(target_vector)
                value_targets.append(value_target)
                move_indices.append(i) 

        total_moves = torch.full((len(move_indices),), len(move_indices), dtype=torch.long)
        return (
            torch.stack(board_tensors),
            torch.stack(target_vectors),
            torch.tensor(value_targets),
            torch.tensor(move_indices),
            total_moves,
            board_fens
        )

    def _generate_board_tensor(self, board):
        input_channels = 112
        board_tensor = torch.zeros((input_channels, 8, 8), dtype=torch.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type - 1
                piece_color = 0 if piece.color == chess.WHITE else 1
                channel = piece_type + 6 * piece_color
                row, col = divmod(square, 8)
                board_tensor[channel, row, col] = 1

        return board_tensor



def collate_fn(batch):
    board_tensors = []
    target_vectors = []
    value_targets = []
    move_indices = []
    total_moves = []
    fens = []

    for boards, targets, values, indices, totals, fen in batch:
        board_tensors.append(boards)
        target_vectors.append(targets)
        value_targets.append(values)
        move_indices.append(indices)
        total_moves.append(totals)
        fens.extend(fen)

    return (
        torch.cat(board_tensors),
        torch.cat(target_vectors),
        torch.cat(value_targets),
        torch.cat(move_indices),
        torch.cat(total_moves),
        fens
    )


# # Dataset and DataLoader
# dataset = load_dataset('angeluriot/chess_games')
# compact_mapping = new_mapping
# chess_dataset = ChessDataset(dataset, compact_mapping)
# chess_dataloader = DataLoader(chess_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

# for board_tensors, target_vectors, value_targets, move_indices in chess_dataloader:
#     print("Board Tensors:", board_tensors.shape)
#     print("Target Vectors:", target_vectors.shape)
#     print("Value Targets:", value_targets.shape)
#     print("Move Indices:", move_indices.shape)
#     break
