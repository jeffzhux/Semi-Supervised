

# log & save
port = 10001
work_dir = './experiment/cifar100/SADE'
log_interval = 100
save_interval = 20

# train
num_classes = 100
epochs = 100 # 1024
iters = 1024 # 1024
batch_size = 128 #128
mu = 7 # unlabeled data batch_size = batch_size * mu
dalign_t = 0.5 # temperature-scaled distribution


# model
model = dict(
    type='WideResNet2expert',
    num_classes = num_classes,
    depth = 28,
    widen_factor = 2,
    drop_rate = 0.0
)

# optimizer
lr = 0.03
weight_decay = 5e-4
optimizer = dict(
    type = 'SGD',
    lr = lr,
    momentum = 0.9,
    weight_decay = weight_decay,
)

lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    #start_step=0,
    warmup_steps=0, # 100
    # warmup_from=lr * 0.1
)

# loss
loss = dict(
    type = 'DiverseExpertLoss',
    threshold = 0.95,
    lambda_u = 1,
    T = 1
)

# data
num_workers = 4
dataset = 'cifar100'
data = dict(
    split = dict(
        beta = 0.1,
        gamma = 50,
        num_classes = 100
    ),

    base = dict(
        type = 'CIFAR100',
        root = './data/cifar100',
        train = True,
        download = False
    ),
    train_labeled = dict(
        type='CIFAR100SSL',
        root='./data/cifar100',
        train=True,
        transform = dict(
            type='weak_transform',
            size=32,
            normal=[(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
        )
    ),
    train_unlabeled = dict(
        type='CIFAR100SSL',
        root='./data/cifar100',
        train=True,
        transform = dict(
            type='FixMatchTransform',
            weak = dict(
                type='weak_transform',
                size=32,
                normal=[(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
            ),
            strong = dict(
                type='strong_transform',
                size=32,
                normal=[(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
            )
        )
    ),
    valid = dict(
        type='CIFAR100',
        root='./data/cifar100',
        train=False,
        download = False,
        transform = dict(
            type= 'valid_transform',
            normal = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
        )
    )
)

