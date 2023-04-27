import torchvision
from torchvision import transforms as T
from datasets.transforms.randaugment import RandAugmentMC


weak_transform = T.Compose([
        T.RandomCrop(size = 32, padding = int(32*0.125), padding_mode='reflect'),
        T.RandomHorizontalFlip(),
    ])
strong_transform = T.Compose([
        T.RandomCrop(size = 32, padding = int(32*0.125), padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        RandAugmentMC(n=6, m=10)
    ])
dataset = torchvision.datasets.CIFAR100(
    root='./data/cifar100',
    train=True,
    #transform=weak_transform,
    download=False
)
flag = True
temp_label = []
import torch
from torch.utils.data import DataLoader

labeled_trainloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=2,
    drop_last=True
)
for img, label in labeled_trainloader:
    print(img.shape)
    print(label)

    break

print(dataset.classes)
exit()
for img, label in dataset:
    # if flag and label == 6:
    #     flag = False
    #     continue
    if len(temp_label) == 100:
        exit()
    if label in temp_label:
        continue
    
    img.save(f'img_{label}.png')
    temp_label.append(label)
    # weak_img = weak_transform(img)
    # strong_img = strong_transform(img)
    # weak_img.save('weak_img.png')
    # strong_img.save('strong_img.png')
