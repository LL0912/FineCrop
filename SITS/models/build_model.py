from SITS.utils.build import build_from_cfg
from SITS.utils.registry import MODELS


def build_model(cfg):
    return build_from_cfg(cfg, MODELS)