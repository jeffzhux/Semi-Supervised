import numpy as np
import math
import torchvision

import datasets
from datasets.transforms.build import build_transform
from utils.config import ConfigDict

# def x_u_split(labels, args):
#     labels = np.array(labels)
#     label_per_class = args.num_labeled // args.num_classes
#     unlabeled_idx = []
#     labeled_idx = []
#     for i in range(args.num_classes):
#         idx = np.where(labels == i)[0]
#         l_idx = np.random.choice(idx, label_per_class, False)
#         unl_idx = np.setdiff1d(idx, l_idx)

#         unlabeled_idx.extend(unl_idx)
#         labeled_idx.extend(l_idx)
#     labeled_idx = np.array(labeled_idx)
#     assert len(labeled_idx) == args.num_labeled

#     # if args.expand_labels or args.num_labeled < args.batch_size:
#     #     num_expand_x = math.ceil(
#     #         args.batch_size * args.eval_step / args.num_labeled)
#     #     labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
#     # np.random.shuffle(labeled_idx)
#     return labeled_idx, unlabeled_idx

def x_u_split(labels, args):
    labels = np.array(labels)
    max_class = len(labels) // args.num_classes
    unlabeled_idx = []
    labeled_idx = []
    for i in range(args.num_classes):
        num_class = max_class * ((1/args.gamma)**(i/(args.num_classes-1)))
        num_label_class = round(num_class * args.beta)

        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, int(num_class), False)
        l_idx = np.random.choice(idx, num_label_class, False)
        unl_idx = np.setdiff1d(idx, l_idx)

        unlabeled_idx.extend(unl_idx)
        labeled_idx.extend(l_idx)
    labeled_idx = np.array(labeled_idx)

    return labeled_idx, unlabeled_idx

def get_cifar10(cfg:ConfigDict):
    args = cfg.copy()
    base_args = args.pop('base')
    split_args = args.pop('split')
    label_args = args.pop('train_labeled')
    unlabel_args =args.pop('train_unlabeled')
    valid_args = args.pop('valid')

    base_dataset = build_dataset(base_args)
    split_args['num_classes'] = 10
    label_idx, unlabel_idx = x_u_split(base_dataset.targets, split_args)

    # labeled
    label_args['indexs'] = label_idx
    label_dataset = build_dataset(label_args)

    # unlabeled
    unlabel_args['indexs'] = unlabel_idx
    unlabel_dataset = build_dataset(unlabel_args)

    # valid
    valid_dataset = build_dataset(valid_args)

    return label_dataset, unlabel_dataset, valid_dataset

def get_cifar100(cfg:ConfigDict):
    args = cfg.copy()
    base_args = args.pop('base')
    split_args = args.pop('split')
    label_args = args.pop('train_labeled')
    unlabel_args =args.pop('train_unlabeled')
    valid_args = args.pop('valid')

    base_dataset = build_dataset(base_args)
    split_args['num_classes'] = 100
    label_idx, unlabel_idx = x_u_split(base_dataset.targets, split_args)

    # labeled
    label_args['indexs'] = label_idx
    label_dataset = build_dataset(label_args)

    # unlabeled
    unlabel_args['indexs'] = unlabel_idx
    unlabel_dataset = build_dataset(unlabel_args)

    # valid
    valid_dataset = build_dataset(valid_args)

    return label_dataset, unlabel_dataset, valid_dataset

def build_dataset(cfg: ConfigDict):
    args = cfg.copy()
    
    transform_args = args.pop('transform') if args.get('transform') else None
    
    if transform_args != None:
        transform  = build_transform(transform_args)
        args.transform = transform
    ds_name = args.pop('type')
    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(**args)
    else:
        ds = datasets.__dict__[ds_name](**args)

    return ds