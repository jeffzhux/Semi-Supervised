import torch
import torch.nn as nn
import torch.nn.functional as F

class FixMatchLoss(nn.Module):
    def __init__(self, threshold, lambda_u = 1.0, T = 1.0) -> None:
        super(FixMatchLoss, self).__init__()
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.T = T
        self.x_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.u_criterion = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits_x, logits_wu, logits_su, targets_x):

        pseudo_label = torch.softmax(logits_wu.detach()/self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lx = self.x_criterion(logits_x, targets_x)
        Lu = (self.u_criterion(logits_su, targets_u) * mask).mean()

        return Lx + self.lambda_u * Lu, Lx, Lu, max_probs