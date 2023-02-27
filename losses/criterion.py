import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list, gt_p_data, threshold, lambda_u = 1.0, T = 1.0):
        super().__init__()
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.T = T
        self.x_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.u_criterion = nn.CrossEntropyLoss(reduction='none')

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.gt_p_data = gt_p_data ** (1/3)
        self.C_number = len(cls_num_list)  # class number

    def _class_rebalancing(self, pseudo_target):
        p_pt = self.gt_p_data[pseudo_target]
        random_p = torch.rand(p_pt.size(), device = p_pt.device)
        mask = p_pt.ge(random_p).float().detach()
        return mask

    def base_loss(self, logits_x, logits_wu, logits_su, targets_x,
        do_resampling=False):

        with torch.no_grad():
            pseudo_label = torch.softmax(logits_wu.detach()/self.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float().detach()

            if do_resampling:
                mask = mask * self._class_rebalancing(targets_u)
                
        Lx = self.x_criterion(logits_x, targets_x)
        Lu = (self.u_criterion(logits_su, targets_u) * mask).mean()

        return Lx + self.lambda_u * Lu

    def forward(self, logits_x, logits_wu, logits_su, # pred
            targets_x, # label data target
        ):
        logits_x = logits_x.transpose(0, 1) # (B, Expert, logit) -> (Expert, B, logit)
        logits_wu = logits_wu.transpose(0, 1) # (B, Expert, logit) -> (Expert, B, logit)
        logits_su = logits_su.transpose(0, 1) # (B, Expert, logit) -> (Expert, B, logit)

        loss = 0
        # Obtain label logits from each expert  
        expert1_x_logits = logits_x[0]
        expert2_x_logits = logits_x[1]

        # Obtain weak augmentation unlabel logits from each expert  
        expert1_wu_logits = logits_wu[0]
        expert2_wu_logits = logits_wu[1]

        # Obtain weak augmentation unlabel logits from each expert  
        expert1_su_logits = logits_su[0]
        expert2_su_logits = logits_su[1]

         
        # Softmax loss for expert 1 -> long tail
        expert1_loss = self.base_loss(expert1_x_logits, expert2_wu_logits, expert1_su_logits, targets_x)
        loss += expert1_loss
        
        # Balanced Softmax loss for expert 2
        expert2_x_logits = expert2_x_logits + torch.log(self.prior + 1e-9)
        expert2_su_logits = expert2_su_logits + torch.log(self.prior + 1e-9)
        expert2_wu_logits = expert2_wu_logits + torch.log(self.prior + 1e-9)
        expert2_loss = self.base_loss(expert2_x_logits, expert1_wu_logits, expert2_su_logits, targets_x, do_resampling=True)
        loss += expert2_loss

        return loss, expert1_loss, expert2_loss