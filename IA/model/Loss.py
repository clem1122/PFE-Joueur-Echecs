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
        loss = self.CELoss(preds, targets)

        return loss
    
class ValueLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(ValueLoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.gamma = gamma

    def forward(self, preds, targets, num_moves, total_moves):
        device = preds.device 

        weight = 1 / (1 + torch.exp(-self.gamma * (num_moves - total_moves/ 2))).to(device)

        loss = self.MSE(preds, targets)

        weighted_loss = weight * loss
        
        return weighted_loss.mean()
    
class Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.PolicyLoss = PolicyLoss()
        self.ValueLoss = ValueLoss(gamma)

    def forward(self, board_fens, preds_pol, targets_pol, preds_val, targets_val, num_moves, tot_moves):
        pol_loss = self.PolicyLoss(preds_pol, targets_pol)
        val_loss = self.ValueLoss(preds_val, targets_val, num_moves, tot_moves)

        for fen, predicted_move_idx in zip(board_fens, torch.argmax(preds_pol, dim=1)):
            board = chess.Board(fen)  # Reconstruire le board à partir du FEN
            move = [key for key, val in new_mapping.items() if val == predicted_move_idx.item()]
        
        if move:  # Si un mouvement est trouvé
            move = chess.Move.from_uci(move[0])
            if not board.is_legal(move):
                print(move, "IS ILLEGAL")
                return (self.alpha * pol_loss + self.beta * val_loss) + 500  # Pénalisation

        return self.alpha * pol_loss + self.beta * val_loss
