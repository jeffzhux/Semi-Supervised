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
        # print('記得要還原 CrestLoss')
        self.threshold = threshold
        # self.threshold = 0.5
        self.lambda_u = lambda_u
        self.T = T
        self.x_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.u_criterion = nn.CrossEntropyLoss(reduction='none')

    def _get_pseudo_target(self, logit):

        with torch.no_grad():
            pseudo_label = torch.softmax(logit.detach()/self.T, dim=-1)
            max_probs, pseudo_target = torch.max(pseudo_label, dim=-1)

        return pseudo_target.detach(), max_probs.detach()

    def _class_rebalancing(self, pseudo_target, pseudo_probs, gt_p_data):

        mask = pseudo_probs.ge(self.threshold)
        pseudo_target = pseudo_target * mask #可考慮信心度跟不考慮信心度
        unique, count = torch.unique(pseudo_target, return_counts=True)
        if unique[0] == 0:
            # ignore class zero
            unique = unique[1:]
            count = count[1:]
        balance_num = torch.round(count * gt_p_data[unique]).int()
        class_rebalancing_mask =torch.zeros_like(pseudo_target)
        # print(unique)
        # print(balance_num)
        for class_idx, bn in zip(unique, balance_num):
            class_mask = torch.eq(pseudo_target, class_idx).float() # get mask by class index

            class_rebalancing_indices = torch.where(class_mask==1)[0][:bn.item()] # class_rebalancing sampling
            
            sub_mask = torch.zeros_like(pseudo_target)
            sub_mask[class_rebalancing_indices] = 1
            class_rebalancing_mask = torch.logical_or(class_rebalancing_mask, sub_mask)

            # print(f'class : {class_idx}, mask num : {sum(class_mask)}, rebalance class : {torch.sum(class_rebalancing_mask.float())}')
        return class_rebalancing_mask.float()
    def forward(
            self, logits_x, logits_wu, logits_su, # pred
            targets_x, # label data target
            gt_p_data, # label data class distribution
            t
        ):

        # Compute supervised loss.
        Lx = self.x_criterion(logits_x, targets_x)

        # Compute unsupervised loss.
        pseudo_target, pseudo_probs = self._get_pseudo_target(logits_wu)
        # pseudo_target, pseudo_probs = self._get_pseudo_target(logits_x, logits_wu, t)
        mask = pseudo_probs.ge(self.threshold).float()
        class_rebalancing_mask = self._class_rebalancing(pseudo_target, pseudo_probs, gt_p_data)

        Lu = (self.u_criterion(logits_su, pseudo_target) * mask * class_rebalancing_mask).mean()

        return Lx + self.lambda_u * Lu, Lx, Lu, pseudo_probs