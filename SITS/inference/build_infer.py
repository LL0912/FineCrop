from SITS.utils import INFER
from SITS.utils.build import build_from_cfg


def build_infer(cfg):
    return build_from_cfg(cfg['infer'], INFER)
