import torch
import sys


def get_device(device_ids, allow_cpu=False):
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % device_ids[0])
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        sys.exit("No allowed device is found")
    return device