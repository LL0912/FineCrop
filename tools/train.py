import argparse
import sys
import os
import torch
sys.path.append(os.path.abspath('.'))
from SITS.apis import build_trainer
from SITS.utils import read_config
from SITS.utils import set_random_seed
import torch.distributed as dist
 
def Argparse():
    parser = argparse.ArgumentParser(description='HOC Training')
    parser.add_argument('-c', '--cfg', type=str, default="/home/ll20/code/SITS_Cls/config/single_image_dataset/EuroCrops/SK_Mulevel.py",help='File path of config')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed')
    parser.add_argument('-t', '--train_mode', action='store_false',  default=True, help='test mode')
    parser.add_argument('-dp', '--data_parallel', action='store_false', default=True, help='test mode')
    parser.add_argument("--local-rank", type=int,default=-1, help="Automatically passed by distributed launcher (DO NOT SET MANUALLY)")
    return parser.parse_args()

# def set_up():
#     print("begin to construct the environment!")
#     if torch.cuda.is_available():
#         local_rank=int(os.environ["LOCAL_RANK"]) 
#         print("rank num",local_rank)
#         torch.cuda.set_device(local_rank)
#         dist.init_process_group(
#         backend="nccl", init_method="env://" )
#     else:
#         print("This is only for DDP training!")
#     print("finish constructing DDP environment!")

if __name__ == "__main__":
    args = Argparse()
    config = read_config(args.cfg,"config")
    config['meta'].update(config_path=args.cfg)

    random_seed = args.random_seed
    if random_seed is not None:
        set_random_seed(random_seed)
        config['meta'].update(random_seed=random_seed)

    if args.data_parallel:
        config['meta'].update(local_rank=int(os.environ["LOCAL_RANK"]))
        config['meta'].update(world_size=torch.cuda.device_count())

    print(config.keys())
    print(args)

    trainer = build_trainer(config)(**config)
    if args.train_mode:
        trainer.run()
    else:
        trainer.test()

        


