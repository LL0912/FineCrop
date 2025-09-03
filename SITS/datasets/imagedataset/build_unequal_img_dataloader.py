from torch.utils.data import DataLoader
from .unequal_sampler import UnequalTileSampler,TileBatchSampler,DistributedUnequalTileSampler
from SITS.utils import build_from_cfg,DATASETS

class unequal_image_dataloader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers):
        self.dataset=dataset
        sampler=UnequalTileSampler(self.dataset)
        batch_sampler=TileBatchSampler(sampler,batch_size=batch_size,drop_last=False)
        super(unequal_image_dataloader,self).__init__(dataset=dataset,
                                                      batch_sampler=batch_sampler,
                                                      num_workers=num_workers,pin_memory=True)


def build_unequal_dataloader(cfg,batch_size,num_workers):
    dataset = build_from_cfg(cfg, DATASETS)
    # sampler = UnequalTileSampler(dataset)
    # # batch_sampler =build_from_cfg(cfg, BATCH_SAMPLER)
    # batch_sampler = TileBatchSampler(sampler, batch_size=cfg["meta"]["batch_size"], drop_last=False)
    # dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
    dataloader=unequal_image_dataloader(dataset,batch_size,num_workers,)
    return dataloader


class unequal_distributed_image_dataloader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers,num_replicas,rank,shuffle):
        self.dataset=dataset
        sampler=DistributedUnequalTileSampler(self.dataset,num_replicas=num_replicas,rank=rank,shuffle=shuffle)
        batch_sampler=TileBatchSampler(sampler,batch_size=batch_size,drop_last=False)
        super(unequal_distributed_image_dataloader,self).__init__(dataset=dataset,
                                                      batch_sampler=batch_sampler,
                                                      num_workers=num_workers,pin_memory=True)

def build_unequal_distributed_dataloader(cfg,batch_size,num_workers,num_replicas,rank,shuffle):
    dataset = build_from_cfg(cfg, DATASETS)
    dataloader=unequal_distributed_image_dataloader(dataset,batch_size,num_workers,num_replicas,rank,shuffle)
    return dataloader


