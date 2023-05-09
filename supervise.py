import os
import time
from tqdm import tqdm

from utils.util import AverageMeter
from utils.util import format_time
from utils.config import ConfigDict

from fixmatch import Trainer


class SL_Trainer(Trainer):
    def __init__(self, cfg: ConfigDict, rank:int):
        super().__init__(cfg, rank)

    def train(self, epoch):
        self.model.train()

        # log
        epoch_end  = time.time()
        iter_end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # data
        labeled_epoch = 0
        self.labeled_trainloader.sampler.set_epoch(labeled_epoch)
        
        labeled_iter = iter(self.labeled_trainloader)

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


            data_time.update(time.time() - iter_end)
            inputs = inputs_x.cuda(non_blocking=True)
            targets_x = targets_x.cuda(non_blocking=True).long()
            logits = self.model(inputs)

            loss = self.criterion(logits, targets_x)
            loss.backward()

            losses.update(loss.item())

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
                f'loss: {losses.avg:.3f} '
            )
            p_bar.update()
        p_bar.close()

        if self.logger is not None: 
            epoch_time = format_time(time.time() - epoch_end)
            self.logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                        f'Data: {data_time.avg:.3f}, '
                        f'Batch: {batch_time.avg:.3f}, '
                        f'lr: {lr:.5f}, '
                        f'train_loss: {losses.avg:.3f} '
            )
        
        if self.writer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/lr', lr, epoch)
            self.writer.add_scalar('Train/loss', losses.avg, epoch)

