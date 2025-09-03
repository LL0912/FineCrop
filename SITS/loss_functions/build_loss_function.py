import torch
from SITS.utils.build import build_from_cfg
from SITS.utils.registry import LOSS_FUNCTIONS

LOSS_FUNCTIONS.register_module(module=torch.nn.CrossEntropyLoss)
LOSS_FUNCTIONS.register_module(module=torch.nn.BCELoss)

def build_loss_function(cfg):
    return build_from_cfg(cfg, LOSS_FUNCTIONS)

def build_loss_functions(**cfg):
    loss_functions = []
    for i,v in cfg.items():
        loss = build_from_cfg(cfg[i], LOSS_FUNCTIONS)
        loss_functions.append(loss)
    return loss_functions

