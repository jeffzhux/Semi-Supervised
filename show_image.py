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
dataset = torchvision.datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    #transform=weak_transform,
    download=False
)
for img, label in dataset:
    print(img)
    img.save('img.png')
    weak_img = weak_transform(img)
    strong_img = strong_transform(img)
    weak_img.save('weak_img.png')
    strong_img.save('strong_img.png')
    exit()