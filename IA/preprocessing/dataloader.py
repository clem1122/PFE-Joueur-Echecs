import chess
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from create_targets import create_target_vector, get_value_target
from mapping import new_mapping

class ChessDataset(Dataset):
    def __init__(self, dataset, compact_mapping):
        self.dataset = dataset['train']
        self.compact_mapping = compact_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        game = self.dataset[idx] #un jeu dans le dataset train
        moves_uci = game['moves_uci'] #on extrait les moves
        winner = game['winner']  #on extrait le gagnant

        value_target = get_value_target(winner) #on tansforme le gagnant pour la loss

        #Initialisations
        board = chess.Board() 
        board_tensors = []
        target_vectors = []
        value_targets = []
        move_indices = []  

        for i, move_uci in enumerate(moves_uci):
            move = chess.Move.from_uci(move_uci) #On transforme le move 

            if move in board.legal_moves:
                board_tensor = self._generate_board_tensor(board)  #on génère notre vecteur

                board.push(move)  # On joue le prochain coup
                
                target_vector = create_target_vector(move, self.compact_mapping) #on génère nos targets grâce au prochain coup

                board_tensors.append(board_tensor)
                target_vectors.append(target_vector)
                value_targets.append(value_target)
                move_indices.append(i) 

        return (
            torch.stack(board_tensors),
            torch.stack(target_vectors),
            torch.tensor(value_targets),
            torch.tensor(move_indices),
            torch.tensor(len(move_indices))
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

    for boards, targets, values, indices in batch:
        board_tensors.append(boards)
        target_vectors.append(targets)
        value_targets.append(values)
        move_indices.append(indices)

    return (
        torch.cat(board_tensors),
        torch.cat(target_vectors),
        torch.cat(value_targets),
        torch.cat(move_indices),
    )


# Dataset and DataLoader
dataset = load_dataset('angeluriot/chess_games')
compact_mapping = new_mapping
chess_dataset = ChessDataset(dataset, compact_mapping)
chess_dataloader = DataLoader(chess_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

for board_tensors, target_vectors, value_targets, move_indices in chess_dataloader:
    print("Board Tensors:", board_tensors.shape)
    print("Target Vectors:", target_vectors.shape)
    print("Value Targets:", value_targets.shape)
    print("Move Indices:", move_indices.shape)
    break
