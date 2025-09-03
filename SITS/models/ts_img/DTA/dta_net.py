import sys
sys.path.append("..")
from .dta import Deformable_spatemp_atten
import torch.nn as nn
import torch
from einops import rearrange,repeat
from SITS.utils.registry import MODELS
@MODELS.register_module(name="dstan")
class Deformable_spatemp_atten_network(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_widths=[64,64,64,128],
                 decoder_widths=[64,64,64,128],
                 n_head=4,
                 str_conv_k=4,
                 str_conv_s=2,
                 str_conv_p=1,
                 reduce_ratio=2,
                 encoder_norm="group",
                 pad_value=0,
                 padding_mode="reflect",
                 positional_encoding=True,
                 offset_range_factor=-1,
                 use_pe=True,
                 no_off=False,
                 dwc_pe=False,
                 log_cpb=True,
                 stride=[2, 8],
                 ksize=[3, 9],
                 num_labelevel=3,
                 num_classlevel=[6, 20, 52],
                 ):

        super(Deformable_spatemp_atten_network,self).__init__()
        self.num_labelevel=num_labelevel
        self.num_classlevel = num_classlevel
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=0,
            norm="group",
            padding_mode="reflect",
        )

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                encoder_widths[0] // n_head, T=10000, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.n_stages=len(encoder_widths)

        self.downconv=nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )

        self.dta_module=nn.ModuleList(
            Deformable_spatemp_atten(input_channel=encoder_widths[i],
                                     n_heads=n_head,
                                     offset_range_factor=offset_range_factor,
                                     use_pe=use_pe,
                                     no_off=no_off,
                                     dwc_pe=dwc_pe,
                                     log_cpb=log_cpb,
                                     stride=stride,
                                     ksize=ksize,
            )
            for i in range(self.n_stages)
        )

        self.time_aggregator=nn.ModuleList(
            temporal_aggregator(input_channels=encoder_widths[i],
                                reduce_ratio=reduce_ratio,
                                )
            for i in range(self.n_stages)
        )

        self.upconv=nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )

        self.out_conv_dict = nn.ModuleList()
        for i in range(num_labelevel):
            out_conv=[decoder_widths[0]]+[num_classlevel[i]]
            self.out_conv_dict.append(ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode))


    def forward(self,x, batch_positions):
        sz_b, seq_len, d, h, w = x.shape
        out=self.in_conv.smart_forward(x)
        if self.positional_encoder is not None:
            batch_positions = batch_positions[0, :].reshape(1, -1)
            batch_positions = batch_positions.repeat(sz_b, 1)
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            bp = self.positional_encoder(bp)
            bp=rearrange(bp, "(b h w) t c -> b t c h w", h=h, w=w)
            out = out + bp# [sz_b * h * w, seq_len, d]

        out_t=self.dta_module[0](out)
        out_t=self.time_aggregator[0](out_t)

        feature_maps_aggregator=[out_t]

        for i in range(self.n_stages-1):
            out=self.downconv[i].smart_forward(out)
            out_t = self.time_aggregator[i+1](self.dta_module[i+1](out))
            feature_maps_aggregator.append(out_t)
        up=feature_maps_aggregator[-1]

        for i in range(self.n_stages - 1):
            up=self.upconv[i](up,feature_maps_aggregator[-(i+2)])

        if self.num_labelevel==3:
            out_g = self.out_conv_dict[2](up)
            out_1= self.out_conv_dict[0](up)
            out_2 = self.out_conv_dict[1](up)
            return out_g, out_1, out_2

        if self.num_labelevel == 4:
            out_l={}
            for i in range(self.num_labelevel):
                out_l[i]=self.out_conv_dict[i](up)
            return out_l[0], out_l[1], out_l[2], out_l[3]

        if self.num_labelevel==1:
            out_g = self.out_conv_dict[0](up)
            return out_g




class temporal_aggregator(nn.Module):
    def  __init__(self,input_channels,reduce_ratio):
        super(temporal_aggregator,self).__init__()
        self.t_atten=nn.Sequential(
            nn.Linear(input_channels, input_channels//reduce_ratio, bias=True),
            nn.ReLU(),
            nn.Linear( input_channels// reduce_ratio,1, bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        sz_b,seq_len,dims, h, w = x.shape
        out_avg=rearrange(x,"b t c h w -> b (h w) t c").mean(dim=1)

        t_weight=self.t_atten(out_avg)

        t_weight=repeat(t_weight,"b t c -> b t c h w", h=h, w=w)

        out=torch.einsum('b t c h w, b t x h w -> b c x h w', x, t_weight).squeeze(2)
        return out


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
                batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out

class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)

class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)

class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


if __name__=="__main__":
    res=24
    x = torch.rand((2, 14, 4, res, res))
    doy = torch.linspace(1, 356, 14).to(torch.int64).reshape(1, -1)
    model=Deformable_spatemp_atten_network(4,num_labelevel=4,num_classlevel=[4,5,7,8])
    out_g,out_1,out_2,out_3=model(x,doy)
    print(out_g.shape)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)