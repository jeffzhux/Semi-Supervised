from typing import List
from torchvision import transforms as T

from .randaugment import RandAugmentMC

def valid_transform(normal:List=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = normal[0], std = normal[1])
    ])
    return transform

def weak_transform(size: int=32, normal:List=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
    transform = T.Compose([
        T.RandomCrop(size = size, padding = int(size*0.125), padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean = normal[0], std = normal[1])
    ])
    return transform

def strong_transform(size: int=32, normal:List=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
    transform = T.Compose([
        T.RandomCrop(size = size, padding = int(size*0.125), padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10, size=size),
        T.ToTensor(),
        T.Normalize(mean = normal[0], std = normal[1])
    ])
    return transform