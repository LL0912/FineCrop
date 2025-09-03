import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from SITS.utils import DATASETS

@DATASETS.register_module()
class ImageBase_MultiLevel_Dataset(Dataset):
    def __init__(self,data_path,doy,num_level,select_type,select_class):
        self.data_path = data_path
        self.doy = doy
        self.num_level = num_level
        self.select_type = select_type
        if select_type is not None:
            self.class_id=select_class[select_type]
        self.get_tilename_list()
        self.get_tile_image_list()
        self.count_dict, self.num_sample = self.build_count_dict()
    def get_tilename_list(self):
        self.tile_list=list(self.doy.keys())
        self.tile_image_dict={tile:[] for tile in self.tile_list}
        for tile_name in self.tile_list:
            self.doy[tile_name]=np.asarray(self.doy[tile_name])
    def get_tile_image_list(self):
        image_file_list=os.listdir(self.data_path)
        for file_name in image_file_list:
            tile_name=file_name.split("-")[0]
            self.tile_image_dict[tile_name].append(file_name)

        del_key=[]
        for tile_name in self.tile_list:
            if len(self.tile_image_dict[tile_name])==0:
                del_key.append(tile_name)
        if len(del_key)!=0:
            for tile_name in del_key:
                if tile_name in self.tile_list:
                    self.tile_list.remove(tile_name)
                if tile_name in self.tile_image_dict.keys():
                    del self.tile_image_dict[tile_name]

    def build_count_dict(self):
        count_dict={}
        sum_start=0
        sum_start=0
        for i, key in  enumerate(self.tile_list):
            count_tile=[]
            if i==0:
                count_tile.append(0)
                count_tile.append(len(self.tile_image_dict[key]))
            else:
                count_tile.append(sum_start)
                count_tile.append(sum_start+len(self.tile_image_dict[key]))

            count_tile.append(len(self.tile_image_dict[key]))
            sum_start += len(self.tile_image_dict[key])
            count_dict[key]=count_tile

        return count_dict,sum_start

    def __getitem__(self, idx):
        tile_name=None
        idx_tile=-1
        for key in self.tile_list:
            idx_s=self.count_dict[key][0]
            idx_e = self.count_dict[key][1]
            if idx>=idx_s and idx<idx_e:
                idx_tile=idx-idx_s
                tile_name=key

        if tile_name!=None and idx_tile!=-1:
            data_name=self.tile_image_dict[tile_name][idx_tile]
            
            data_file_path=os.path.join(self.data_path,data_name)
            
            
            x=np.load(data_file_path)["data"]
            label=np.load(data_file_path)["label"]

            if self.select_type is not None:
                new_label_t = np.ones_like(label) * 255
                for l in range(self.num_level):
                    cls_use_list=self.class_id[l]
                    for i,cls in enumerate(cls_use_list):
                        new_label_t[label[:,:,:,l]==cls,l]=i
                label = new_label_t


            return (torch.from_numpy(x).float(),torch.from_numpy(label[:,:,0]).long(),torch.from_numpy(label[:,:,1]).long(),
                    torch.from_numpy(label[:,:,2]).long(),torch.from_numpy(label[:,:,3]).long(),
                    torch.from_numpy(label[:,:,4]).long(),torch.from_numpy(self.doy[tile_name]),tile_name)

        else:
            print(idx)
            print(tile_name)
            print(idx_tile)
            print("index out limit!")





