import torch
from SITS.utils import OPTIMIZERS
from SITS.utils import build_from_cfg

OPTIMIZERS.register_module(module=torch.optim.SGD)
OPTIMIZERS.register_module(module=torch.optim.Adam)
OPTIMIZERS.register_module(module=torch.optim.AdamW)

def build_optimizer(cfg):
    return build_from_cfg(cfg,OPTIMIZERS)