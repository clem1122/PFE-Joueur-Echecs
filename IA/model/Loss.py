import torch
import torch.nn as nn
import chess
import json

with open("mapping.json", "r") as json_file:
    new_mapping = json.load(json_file)

class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        return self.CELoss(preds, targets)

class ValueLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(ValueLoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.gamma = gamma

    def forward(self, preds, targets, num_moves, total_moves):
        device = preds.device
        num_moves, total_moves = num_moves.to(device), total_moves.to(device)

        weight = 1 / (1 + torch.exp(-self.gamma * (num_moves - total_moves / 2)))

        loss = self.MSE(preds, targets)
        return (weight * loss).mean()

class Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, penalty=500.0):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty
        self.PolicyLoss = PolicyLoss()
        self.ValueLoss = ValueLoss(gamma)

    def forward(self, board_fens, preds_pol, targets_pol, preds_val, targets_val, num_moves, tot_moves):
        device = preds_pol.device 

        pol_loss = self.PolicyLoss(preds_pol, targets_pol)
        val_loss = self.ValueLoss(preds_val, targets_val, num_moves, tot_moves)

        predicted_moves_idx = torch.argmax(preds_pol, dim=1).cpu().tolist()
        predicted_moves = [move for idx in predicted_moves_idx if (move := new_mapping.get(str(idx), None))]

        penalties = torch.tensor(0.0, device=device)
        for fen, move_uci in zip(board_fens, predicted_moves):
            if move_uci: 
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                if not board.is_legal(move):
                    penalties += self.penalty 

        return self.alpha * pol_loss + self.beta * val_loss + penalties