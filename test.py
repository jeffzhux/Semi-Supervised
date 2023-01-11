import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class MovingAverage:
    """Class which accumilates moving average of distribution of labels."""
    def __init__(self, num_classes, buffer_size = 128):
        # Mean
        self.ma = torch.ones(size=(buffer_size, num_classes)) / num_classes

    def __call__(self):
        v = self.ma.mean(dim=0)
        print(self.ma.sum())
        return v / self.ma.sum()
    
    def update(self, entry):
        entry = torch.mean(entry, dim=0, keepdim=True)
        self.ma = torch.cat([self.ma[1:], entry])
dataset = CIFAR10(root='./data/cifar10')
gt_p_data = (np.unique(dataset.targets, return_counts=True)[1]/len(dataset.targets)).tolist()
print(gt_p_data)
    

