import torch
from SITS.utils.build import build_from_cfg
from SITS.utils import LR_SCHEDULER
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

LR_SCHEDULER.register_module(module=torch.optim.lr_scheduler.ExponentialLR)
LR_SCHEDULER.register_module(module=torch.optim.lr_scheduler.StepLR)
LR_SCHEDULER.register_module(module=CosineLRScheduler)
LR_SCHEDULER.register_module(module=LambdaLR)
LR_SCHEDULER.register_module(module=ReduceLROnPlateau)

def build_lr_scheduler(cfg):
    return build_from_cfg(cfg, LR_SCHEDULER)