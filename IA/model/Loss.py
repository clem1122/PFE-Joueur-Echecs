import torch
import torch as nn

class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        loss = self.CELoss(preds, targets)

        return loss
    
class ValueLoss(nn.Module):
    def __init__(self):
        super(ValueLoss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, preds, targets):
        loss = self.MSE(preds, targets)

        return loss
    
class Loss():
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.PolicyLoss = PolicyLoss()
        self.ValueLoss = ValueLoss()

    def forward(self, preds_pol, targets_pol, preds_val, targets_val):
        pol_loss = self.PolicyLoss(preds_pol, targets_pol)
        val_loss = self.ValueLoss(preds_val, targets_val)

        return self.alpha * pol_loss + self.beta * val_loss