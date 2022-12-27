import logging

import torchvision
from utils.config import Config
import models


def build_model(cfg: Config, logger: logging=None):
    
    args = cfg.copy()
    model_name = args.pop('type')

    if logger is not None:
        logger.info(f'Model: {model_name}')

    if hasattr(torchvision.models, model_name):
        model = getattr(torchvision.models, model_name)(**args)
    else:
        model = models.__dict__[model_name](**args)
    return model