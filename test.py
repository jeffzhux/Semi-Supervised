import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10



a = torch.tensor([0.4026, 0.2415, 0.1449, 0.0870, 0.0523, 0.0314, 0.0185, 0.0113, 0.0064, 0.0040])


b = torch.tensor([0,1,2,3,4,5,1,2,1,2,3,1,2,4,3,5,3,2,1,3,2,3])
print(torch.where(b==1)[0])
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
