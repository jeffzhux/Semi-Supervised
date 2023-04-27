import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from datasets.build import get_cifar100
from utils.config import Config

cfg = Config.fromfile('./configs/SADE/cifar100_config.py')
# print(cfg)
labeled_dataset, unlabeled_dataset, valid_dataset = get_cifar100(cfg.data)

sample_rate = torch.as_tensor(labeled_dataset.p_data, device='cuda')
sample_rate = torch.flip(sample_rate, dims=(0,))/sample_rate[0]
print(sample_rate)
sample_rate = torch.as_tensor(unlabeled_dataset.p_data, device='cuda')
sample_rate = torch.flip(sample_rate, dims=(0,))/sample_rate[0]
print(sample_rate)
for i in sample_rate:
    print(i.item())
# unique, indices, count = torch.unique(b, return_inverse =True, return_counts=True)

# if unique[0] == 0:
#     unique = unique[1:]
#     count = count[1:]
# print(count)
# print(a[unique])
# print(torch.round(count * a[unique]))
# get_num = torch.round(count * a[unique]).int()
# mask2 =torch.zeros_like(b)
# for i, num in zip(unique, get_num):
#     class_mask = torch.eq(b, i).float() # get mask by class index
#     class_rebalancing_indices = torch.topk(class_mask, num.item())[1] # class_rebalancing sampling
#     mask2[class_rebalancing_indices] = 1
    
#     print(mask2)
#     # mask2 = torch.logical_and(mask2, sub_mask).float()
#     break
#     # print((b==i).nonzero(as_tuple=True))
#     # mask2 += (b==i).nonzero(as_tuple=True)
