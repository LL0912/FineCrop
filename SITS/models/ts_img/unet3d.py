"""
Taken from https://github.com/roserustowicz/crop-type-mapping/
Implementation by the authors of the paper :
"Semantic Segmentation of crop type in Africa: A novel Dataset and analysis of deep learning methods"
R.M. Rustowicz et al.

Slightly modified to support image sequences of varying length in the same batch.
"""

import torch
import torch.nn as nn
from SITS.utils.registry import MODELS


def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model

@MODELS.register_module(name='unet3d')
class UNet3D(nn.Module):
    def __init__(self, in_channel, feats=8, pad_value=None, zero_pad=True,num_labelevel=3,num_classlevel=[6,20,52],use_channel_fg=False,channel_group=None):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.pad_value = pad_value
        self.zero_pad = zero_pad
        self.num_labelevel=num_labelevel
        
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        self.use_channel_fg = use_channel_fg
        
        self.out_conv_dict = nn.ModuleList()
        self.classifier = nn.ModuleList()
        for i in range(num_labelevel):
            if self.use_channel_fg:
                self.out_conv_dict.append(nn.Conv3d(feats * 2, num_classlevel[i]*channel_group, kernel_size=3, stride=1, padding=1))
            else:
                self.out_conv_dict.append(nn.Conv3d(feats * 2, num_classlevel[i], kernel_size=3, stride=1, padding=1))
        if self.use_channel_fg:
            for i in range(num_labelevel):
                self.classifier.append(nn.Conv3d(num_classlevel[i]*channel_group, num_classlevel[i], kernel_size=3, stride=1, padding=1))


    def forward(self, x,return_fea=False):
        out = x.permute(0, 2, 1, 3, 4)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask
            if self.zero_pad:
                out[out == self.pad_value] = 0
        en3 = self.en3(out)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)
        concat4 = torch.cat([center_out, en4[:, :, :center_out.shape[2], :, :]], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3[:, :, :trans3.shape[2], :, :]], dim=1)
        dc3 = self.dc3(concat3)
        
        final_out={}
        final_fea={}
        for i in range(self.num_labelevel):
            if self.use_channel_fg:
                fea=self.out_conv_dict[i](dc3)
                out_g=self.classifier[i](fea)
                final_out[i] = out_g.permute(0, 1, 3, 4, 2)
                final_fea[i] = fea.permute(0, 1, 3, 4, 2)
            else:
                out_g = self.out_conv_dict[i](dc3)
                final_out[i]=out_g.permute(0, 1, 3, 4, 2)

        if self.pad_value is not None:
            if pad_mask.any():
                # masked mean
                pad_mask = pad_mask[:, :final_out[-1].shape[-1]]  # match new temporal length (due to pooling)
                pad_mask = ~pad_mask  # 0 on padded values
                for i in range(self.num_labelevel):
                    final_out[i]=(final_out[i].permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(dim=-1)[None, None, None, :]
                    if self.use_channel_fg:
                        final_fea[i]=(final_fea[i].permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(dim=-1)[None, None, None, :]
            else:
                for i in range(self.num_labelevel):
                    final_out[i]=final_out[i].mean(dim=-1)
                    if self.use_channel_fg:
                        final_fea[i]=final_fea[i].mean(dim=-1)

        else:
            for i in range(self.num_labelevel):
                final_out[i] = final_out[i].mean(dim=-1)
                if self.use_channel_fg:
                    final_fea[i] = final_fea[i].mean(dim=-1)


        if self.num_labelevel==3:
            if return_fea:
                return final_out[0],final_out[1],final_out[2],final_fea[0],final_fea[1],final_fea[2]
            else:
                return final_out[2],final_out[0],final_out[1]
        if self.num_labelevel==4:
            if return_fea:
                return final_out[0],final_out[1],final_out[2],final_out[3],final_fea[0],final_fea[1],final_fea[2],final_fea[3]
            else:
                return final_out[0],final_out[1],final_out[2],final_out[3]
                
        if self.num_labelevel==1:
            if return_fea:
                if self.use_channel_fg:
                    return final_out[0],final_fea[0]
                else:
                    return final_out[0],dc3
            else:
                return final_out[0]

# if __name__=="__main__":
#
#     res = 24
#     # gpu_ids = ["2","3"]
#     # device_ids = [int(i) for i in gpu_ids if i.isnumeric()]
#     # device = get_device(device_ids, allow_cpu=False)
#     x = torch.rand((2, 16, 4, res, res))#.to(device)
#     doy = torch.linspace(1, 356, 16).to(torch.int64).reshape(1, -1)  # .to(device)
#     print(doy.shape)
#     model=UNet3D(in_channel=4,num_labelevel=1,num_class=[10])
#     out_g=model(x)
#     print(out_g.shape)
