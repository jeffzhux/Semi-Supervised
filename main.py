import time
import os
import platform
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.config import Config
from utils.util import set_seed
from crest import CReST_Trainer
from fixmatch import Trainer as FixMatch_Trainer
from supervise import SL_Trainer

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--task', type=str, choices=['FixMatch','Our', 'SL'])
    parser.add_argument('--mode', type=str, choices=['train','test', 'export'])
    parser.add_argument('--weight', type=str)
    args = parser.parse_args()

    return args

def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    cfg.task = args.task
    cfg.mode = args.mode
    cfg.weight = args.weight
    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # worker
    if cfg.mode == 'train':
        cfg.work_dir = os.path.join(cfg.work_dir, f'{cfg.timestamp}')
        cfg.num_workers = min(cfg.num_workers, mp.cpu_count()-2)
        os.makedirs(cfg.work_dir, exist_ok=True)

        # cfgname
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
        assert cfg.cfgname is not None, f'{cfg.cfgname} is not exist'

    # seed
    if not hasattr(cfg, 'seed'):
        cfg.seed = 25
    set_seed(cfg.seed)

    return cfg

def main_worker(rank, world_size, cfg):
    print(f'==> start rank: {rank}')

    cfg.local_rank = rank % 8
    torch.cuda.set_device(rank)
    if cfg.mode == 'train':
        set_seed(cfg.seed+rank, cuda_deterministic=False)
    else:
        set_seed(cfg.seed+rank)

    print(f'System : {platform.system()}')
    if platform.system() == 'Windows':
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    else: # Linux
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)
    
    if cfg.task == 'Our':
        trainer = CReST_Trainer(cfg, rank)
    elif cfg.task == 'FixMatch':
        trainer = FixMatch_Trainer(cfg, rank)
    else:
        trainer = SL_Trainer(cfg, rank)

    if cfg.mode == 'train':
        trainer.fit()
    elif cfg.mode == 'export':
        trainer.export()
    else:
        trainer.test()

def main():
    args = get_args()
    cfg = get_config(args)

    cfg.world_size = torch.cuda.device_count()
    print(f'GPUs on this node: {cfg.world_size}')
    cfg.bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', cfg.bsz_gpu)
    
    log_file = os.path.join(cfg.work_dir, f'{cfg.timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    if cfg.world_size > 0:
        mp.spawn(main_worker, nprocs = cfg.world_size, args=(cfg.world_size, cfg))

if __name__ == '__main__':
    main()