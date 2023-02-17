

# log & save
port = 10001
work_dir = './experiment/cifar10/CReST'
log_interval = 100
save_interval = 20

# train
num_classes = 10
epochs = 200 # 1024
iters = 1024 # 1024
batch_size = 128 #128
mu = 7 # unlabeled data batch_size = batch_size * mu
dalign_t = 0.5 # temperature-scaled distribution


# model
model = dict(
    type='WideResNet',
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
    type = 'CReSTLoss',
    threshold = 0.95,
    lambda_u = 1,
    T = 1
)

# data
num_workers = 4
dataset = 'cifar10'
data = dict(
    split = dict(
        beta = 0.1,
        gamma = 100,
        num_classes = 10
    ),

    base = dict(
        type = 'CIFAR10',
        root = './data/cifar10',
        train = True,
        download = False
    ),
    train_labeled = dict(
        type='CIFAR10SSL',
        root='./data/cifar10',
        train=True,
        transform = dict(
            type='weak_transform',
            size=32,
            normal=[(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
        )
    ),
    train_unlabeled = dict(
        type='CIFAR10SSL',
        root='./data/cifar10',
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
        type='CIFAR10',
        root='./data/cifar10',
        train=False,
        download = False,
        transform = dict(
            type= 'valid_transform',
            normal = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)] # mean, std
        )
    )
)

