from torch.utils.data import DataLoader
from SITS.utils import build_from_cfg,DATASETS
# class sits_image_loader(DataLoader):
#     def __init__(self,dataset,batch_size,num_workers):
#         self.dataset=dataset
#         #pytorch build loader
#         super(sits_image_loader,self).__init__(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

def build_sits_dataloader(cfg,batch_size,num_workers):
    dataset = build_from_cfg(cfg, DATASETS)
    dataloader=DataLoader(dataset,batch_size,num_workers)
    return dataloader