import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10



a = torch.tensor([0.2154, 0.2520, 0.3037, 0.3583, 0.4273, 0.5066, 0.6000, 0.7114, 0.8434, 1.0000])
a

b = torch.tensor([0,1,2,3,4,5,1,2,1,2,3,1,2,4,3,5,3,2,1,3,2,3, 9])
print(a[b])
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
