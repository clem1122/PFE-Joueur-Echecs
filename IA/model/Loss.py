import torch
import torch.nn as nn

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
        weight = 1 / (1 + torch.exp(-self.gamma * (num_moves - total_moves/ 2)))

        loss = self.MSE(preds, targets)

        weighted_loss = weight * loss
        
        return weighted_loss.mean()
    
class Loss():
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        self.alpha = alpha
        self.beta = beta
        self.PolicyLoss = PolicyLoss()
        self.ValueLoss = ValueLoss(gamma)

    def forward(self, preds_pol, targets_pol, preds_val, targets_val, num_moves, tot_moves):
        pol_loss = self.PolicyLoss(preds_pol, targets_pol)
        val_loss = self.ValueLoss(preds_val, targets_val, num_moves, tot_moves)

        return self.alpha * pol_loss + self.beta * val_loss