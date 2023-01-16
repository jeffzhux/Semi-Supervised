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

class CReSTLoss(nn.Module):
    def __init__(self, threshold, lambda_u = 1.0, T = 1.0):
        super(CReSTLoss, self).__init__()
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.T = T
        self.x_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.u_criterion = nn.CrossEntropyLoss(reduction='none')

    # @torch.no_grad
    # def _get_pseudo_target(self, logit, gt_p_data, p_data, t):

    #     target_dist = torch.pow(gt_p_data, t)
    #     target_dist /= target_dist.sum()

    #     q = F.softmax(logit, dim=-1)
    #     q = q * (1e-6 + target_dist) / (1e-6 + p_data)
    #     q /= torch.sum(q, dim=-1, keepdim=True)
        
    #     max_probs, pseudo_target= torch.max(q, dim=-1)

    #     return pseudo_target.detach(), max_probs.detach()

    torch.no_grad
    def _get_pseudo_target(self, logist_x, logits_wu, t):

        q = F.softmax(logits_wu, dim=-1)
        q = q * (F.softmax(logist_x, dim=-1)).mean() / q.mean()
        
        pt = q ** 2
        targets_u = pt / pt.sum(dim=-1, keepdim=True)
        max_probs, pseudo_target= torch.max(targets_u, dim=-1)

        return pseudo_target.detach().long(), max_probs.detach()

    def forward(
            self, logits_x, logits_wu, logits_su, # pred
            targets_x, # label data target
            gt_p_data, # label data class distribution
            p_data, # unlabeled data class distribution
            t
        ):

        # Compute supervised loss.
        Lx = self.x_criterion(logits_x, targets_x)

        # Compute unsupervised loss.
        # pseudo_target, pseudo_probs = self._get_pseudo_target(logits_wu, gt_p_data, p_data, t)
        pseudo_target, pseudo_probs = self._get_pseudo_target(logits_x, logits_wu, t)

        mask = pseudo_probs.ge(self.threshold).float()
        Lu = (self.u_criterion(logits_su, pseudo_target) * mask).mean()
        
        return Lx + self.lambda_u * Lu, Lx, Lu, pseudo_probs