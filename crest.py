import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.util import AverageMeter
from utils.util import accuracy, adjust_learning_rate, format_time, torch_distributed_zero_first, kl_divergence
from utils.config import ConfigDict
from utils.build import build_logger
from datasets.build import get_cifar10, get_cifar100
from datasets.sampler.distributed import WeightDistributedSampler

from models.build import build_model
from losses.build import build_loss
from optimizers.build import build_optimizer

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class Trainer(object):
    def __init__(self, cfg: ConfigDict, rank:int):
        self.cfg = cfg
        self.rank = rank

        # clear gpu memory
        torch.cuda.empty_cache()

        self.logger, self.writer = None, None
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
            self.logger = build_logger(self.cfg.work_dir, 'train')

        with torch_distributed_zero_first(self.rank):
            # build dataset
            if cfg.dataset == 'cifar10':
                self.labeled_dataset, self.unlabeled_dataset, self.valid_dataset = get_cifar10(cfg.data)
            
            if cfg.dataset == 'cifar100':
                self.labeled_dataset, self.unlabeled_dataset, self.valid_dataset = get_cifar100(cfg.data)

        self.sample_rate = torch.as_tensor(self.labeled_dataset.p_data, device='cuda')
        self.sample_rate = torch.flip(self.sample_rate, dims=(0,))/self.sample_rate[0]

        train_sampler = torch.utils.data.distributed.DistributedSampler

        self.labeled_trainloader = DataLoader(
            self.labeled_dataset,
            sampler=train_sampler(self.labeled_dataset),
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        

        self.unlabeled_trainloader = DataLoader(
            self.unlabeled_dataset,
            sampler = train_sampler(self.unlabeled_dataset),
            batch_size=self.cfg.batch_size * self.cfg.mu,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
        
        # build model
        with torch_distributed_zero_first(self.rank):
            self.model = build_model(cfg.model)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)
        self.model_without_ddp = self.model.module
        
        # build criterion & optimizer
        cfg.loss['cls_num_list'] = self.labeled_dataset.cls_num_list
        cfg.loss['gt_p_data'] = self.sample_rate
        self.criterion = build_loss(cfg.loss).cuda()

        self.optimizer = build_optimizer(cfg.optimizer, self.model.parameters())
        self.optimizer.zero_grad()

        self.start_epoch = 1
        self.epochs = cfg.epochs
        self.dalign_t = cfg.dalign_t
    def _get_dalign_t(self, current_epoch):
        cur = current_epoch / (self.epochs - 1)
        return (1.0 - cur) * 1.0 + cur * self.dalign_t

    def train(self, epoch):
        self.model.train()

        # log
        epoch_end  = time.time()
        iter_end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        # data
        labeled_epoch = 0
        unlabeled_epoch = 0
        self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
        self.unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
        
        labeled_iter = iter(self.labeled_trainloader)
        unlabeled_iter = iter(self.unlabeled_trainloader)

        # utils
        # control the temperature-scaled distribution
        #current_dalign_t = self._get_dalign_t(epoch)

        p_bar = tqdm(range(self.cfg.iters), disable=self.rank not in [-1, 0])
        for batch_idx in range(self.cfg.iters):
            # labeled data
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_epoch += 1
                self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(self.labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            # unlabeled data
            try:
                (inputs_wu, inputs_su), _ = next(unlabeled_iter)
            except:
                unlabeled_epoch += 1
                self.unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.unlabeled_trainloader)
                (inputs_wu, inputs_su), _ = next(unlabeled_iter)

            data_time.update(time.time() - iter_end)
            batch_size = inputs_x.size(0)
            inputs = interleave(torch.cat((inputs_x, inputs_wu, inputs_su)), 2*self.cfg.mu+1).cuda(non_blocking=True)
            targets_x = targets_x.cuda(non_blocking=True).long()
            
            logits = self.model(inputs)
            logits = de_interleave(logits, 2*self.cfg.mu+1)
            logits_x = logits[:batch_size] # label data
            logits_wu, logits_su = logits[batch_size:].chunk(2) # unlabeled data
            del logits

            loss, expert1, expert2 = self.criterion(logits_x, logits_wu, logits_su, targets_x)

            loss.backward()

            losses.update(loss.item())
            losses_x.update(expert1.item())
            losses_u.update(expert2.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            batch_time.update(time.time() - iter_end)
            iter_end = time.time()

            # print info
            lr = self.optimizer.param_groups[0]['lr']
            p_bar.set_description(
                f'[Epoch]/[Iter]: [{epoch:4}/{self.cfg.epochs:4}]/[{batch_idx:4}/{self.cfg.iters:4}] - '
                f'Data: {data_time.avg:.3f}, '
                f'Batch: {batch_time.avg:.3f}, '
                f'lr: {lr:.5f}, '
                f'loss: {losses.avg:.3f}, '
                f'expert1: {losses_x.avg:.3f}, '
                f'expert2: {losses_u.avg:.3f}'
            )
            p_bar.update()

        if self.logger is not None: 
            epoch_time = format_time(time.time() - epoch_end)
            self.logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                        f'Data: {data_time.avg:.3f}, '
                        f'Batch: {batch_time.avg:.3f}, '
                        f'lr: {lr:.5f}, '
                        f'train_loss: {losses.avg:.3f}, '
                        f'train_loss_expert1: {losses_x.avg:.3f}, '
                        f'train_loss_expert2: {losses_u.avg:.3f}, '
            )
        
        if self.writer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/lr', lr, epoch)
            self.writer.add_scalar('Train/loss', losses.avg, epoch)
            self.writer.add_scalar('Train/expert1', losses_x.avg, epoch)
            self.writer.add_scalar('Train/expert2', losses_u.avg, epoch)


    def valid(self, epoch):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        expert1_losses = AverageMeter()
        expert2_losses = AverageMeter()
        expert1_top1 = AverageMeter()
        expter1_top5 = AverageMeter()
        expert2_top1 = AverageMeter()
        expter2_top5 = AverageMeter()
        
        epoch_end  = time.time()
        iter_end = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                data_time.update(time.time() - iter_end)

                inputs = inputs.cuda()
                targets = targets.cuda()

                outputs = self.model(inputs)
                outputs = outputs.transpose(0, 1) # (B, Expert, logit) -> (Expert, B, logit)
                expert1_loss = F.cross_entropy(outputs[0], targets)
                expert2_loss = F.cross_entropy(outputs[1], targets)

                expert1_acc1, expert1_acc5 = accuracy(outputs[0], targets, topk=(1, 5))
                expert2_acc1, expert2_acc5 = accuracy(outputs[1], targets, topk=(1, 5))

                expert1_losses.update(expert1_loss.item(), inputs.shape[0])
                expert2_losses.update(expert2_loss.item(), inputs.shape[0])

                expert1_top1.update(expert1_acc1.item(), inputs.shape[0])
                expter1_top5.update(expert1_acc5.item(), inputs.shape[0])
                expert2_top1.update(expert2_acc1.item(), inputs.shape[0])
                expter2_top5.update(expert2_acc5.item(), inputs.shape[0])

                batch_time.update(time.time() - iter_end)
                iter_end = time.time()

        epoch_time = format_time(time.time() - epoch_end)
        if self.logger is not None: 
            epoch_time = format_time(time.time() - epoch_end)
            self.logger.info(f'Valid Epoch [{epoch}] - test_time: {epoch_time}, '
                        f'expert1 loss: {expert1_losses.avg:.3f}, '
                        f'expert2 loss: {expert2_losses.avg:.3f}, '
                        f'expert1_top1: {expert1_top1.avg:.3f}, '
                        f'expter1_top5: {expter1_top5.avg:.3f}, '
                        f'expert2_top1: {expert2_top1.avg:.3f}, '
                        f'expter2_top5: {expter2_top5.avg:.3f} '
            )
        
        if self.writer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Valid/expert1 loss', expert1_losses.avg, epoch)
            self.writer.add_scalar('Valid/expert1_top1', expert1_top1.avg, epoch)
            self.writer.add_scalar('Valid/expter1_top5', expter1_top5.avg, epoch)
            self.writer.add_scalar('Valid/expert2 loss', expert2_losses.avg, epoch)
            self.writer.add_scalar('Valid/expert2_top1', expert2_top1.avg, epoch)
            self.writer.add_scalar('Valid/expter2_top5', expter2_top5.avg, epoch)

    def save(self, epoch):
        if self.rank == 0 and epoch % self.cfg.save_interval == 0:
            model_path = os.path.join(self.cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': self.optimizer.state_dict(),
                'model_state': self.model_without_ddp.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, model_path)

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            adjust_learning_rate(self.cfg.lr_cfg, self.optimizer, epoch)

            self.train(epoch)

            self.valid(epoch)

            # save check point
            self.save(epoch)
