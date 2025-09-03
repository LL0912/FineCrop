"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import sys
sys.path.append("..")
import torch
import torch.nn as nn

from .convlstm import ConvLSTM, BConvLSTM
from .ltae import LTAE2d
from SITS.utils.registry import MODELS
@MODELS.register_module(name="utae_hml")
class UTAE_HML(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],  #,[32, 64, 128, 256]
        decoder_widths=[32, 32, 64, 128], #,[32, 64, 128, 256]

        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
        num_labelevel=3,
        num_classlevel=[6,20,52],
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UTAE_HML, self).__init__()
        self.num_labelevel=num_labelevel
        assert len(num_classlevel)==self.num_labelevel
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
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
        self.up_blocks = nn.ModuleList(
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
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.hidden_number=decoder_widths[0]
        self.out_conv_dict=nn.ModuleList()
        # for i in range(num_labelevel):
        #     # out_conv=[decoder_widths[0]]+[num_classlevel[i]]
        #     out_conv = [decoder_widths[0]] + [num_classlevel[i]]
        #     self.out_conv_dict.append(ConvBlock(nkernels=[decoder_widths[0]//self.num_labelevel*(i+1)] + out_conv, padding_mode=padding_mode))

        for i in range(num_labelevel):
            out_conv=[decoder_widths[0],decoder_widths[0]]+[num_classlevel[i]]
            self.out_conv_dict.append(ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode))
        self.sigmoid=nn.Sigmoid()
        # self.out_conv_g = ConvBlock(nkernels=[decoder_widths[0]] + out_conv_g, padding_mode=padding_mode)
        # self.out_conv_1 = ConvBlock(nkernels=[decoder_widths[0]] + out_conv_1, padding_mode=padding_mode)
        # self.out_conv_2 = ConvBlock(nkernels=[decoder_widths[0]] + out_conv_2, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None, return_att=False):
        B,T,C,H,W=input.shape
        batch_positions = batch_positions[0, :].reshape(1, -1)
        batch_positions = batch_positions.repeat(B, 1)
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)


        if self.num_labelevel==3:
            out_g = self.out_conv_dict[2](out)
            out_1= self.out_conv_dict[0](out)
            out_2 = self.out_conv_dict[1](out)

            if self.encoder:
                return out_g, out_1, out_2, maps
            else:
                if return_att:
                    return out_g, out_1, out_2, att
                if self.return_maps:
                    return out_g, out_1, out_2, maps
                else:
                    return out_g, out_1, out_2

        if self.num_labelevel == 4:
            # x_1=out[:,0:self.hidden_number//4,:,:]
            # x_2=out[:,self.hidden_number//4:self.hidden_number//4*2,:,:]
            # x_3 = out[:, self.hidden_number // 4 * 2:self.hidden_number // 4 * 3, :, :]
            # x_4 = out[:, self.hidden_number // 4 * 3:self.hidden_number, :, :]
            #
            # out_l_1=self.out_conv_dict[0](x_1)
            # out_l_2= self.out_conv_dict[1](torch.cat([x_2,x_1.detach()],1))
            # out_l_3 = self.out_conv_dict[2](torch.cat([x_3, x_2.detach(),x_1.detach()], 1))
            # out_l_4 = self.out_conv_dict[3](torch.cat([x_4, x_3.detach(), x_2.detach(), x_1.detach()], 1))

            out_l={}
            for i in range(self.num_labelevel):
                out_l[i]=self.out_conv_dict[i](out)

            out_all = torch.cat([out_l[0], out_l[1], out_l[2], out_l[3]],1)
            sigmoid_cat = self.sigmoid(out_all)
            if self.encoder:
                return out_l[0], out_l[1], out_l[2], out_l[3], maps
            else:
                if return_att:
                    return out_l[0], out_l[1], out_l[2], out_l[3], att
                if self.return_maps:
                    return out_l[0], out_l[1], out_l[2], out_l[3], maps
                else:
                    return out_l[0], out_l[1], out_l[2], out_l[3],sigmoid_cat



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


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class RecUNet(nn.Module):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
        )

        self.down_blocks = nn.ModuleList(
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
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="mean")

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None):

        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                skip = self.temporal_aggregator(
                    feature_maps[-(i + 2)], pad_mask=pad_mask
                )
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if self.return_maps:
                return out, maps
            else:
                return out

# if __name__=="__main__":
#     res = 24
#     # gpu_ids = ["2","3"]
#     # device_ids = [int(i) for i in gpu_ids if i.isnumeric()]
#     # device = get_device(device_ids, allow_cpu=False)
#     device= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#     from SITS.loss_functions.Hitree_loss import hitree_loss
#     x = torch.rand((2, 14, 4, res, res))#.to(device)
#     doy = torch.linspace(1, 356, 14).to(torch.int64).reshape(1, -1)  # .to(device)
#     # print(doy.shape)
#     model=UTAE_HML(input_dim=4,num_labelevel=4,num_classlevel=[4,5,7,8])
#     out_g,out_1,out_2,out_3,sigmoid=model(x,doy)
#     print(out_g.shape)
#     print(out_1.shape)
#     print(out_2.shape)
#     print(out_3.shape)
#     print(sigmoid.shape)
#     # for param_name, param in model.named_parameters():
#     #     print(param_name)
#     #     param.requires_grad = False
#     map_dict={
#             1: {
#             0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
#                 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
#                 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
#             1: [64],
#             2: [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
#             3: [83],
#             4: [84, 85],
#             5: [86]},
#
#           2: {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1: [11, 12, 13, 14, 15], 2: [16], 3: [17],
#               4: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 5: [32], 6: [33, 34],
#               7: [35, 36, 37, 38], 8: [39], 9: [40], 10: [41, 42, 43, 44], 11: [45], 12: [46], 13: [47],
#               14: [48, 49, 50, 51, 52], 15: [53, 54], 16: [55], 17: [56], 18: [57, 58, 59, 60],
#               19: [61, 62], 20: [63], 21: [64], 22: [65, 66, 67, 68, 69, 70, 71], 23: [72, 73, 74, 75, 76],
#               24: [77, 78], 25: [79], 26: [80], 27: [81, 82], 28: [83], 29: [84], 30: [85], 31: [86]},
#
#           3: {0: [0, 1], 1: [2], 2: [3], 3: [4, 5], 4: [6], 5: [7], 6: [8], 7: [9], 8: [10], 9: [11],
#               10: [12], 11: [13], 12: [14], 13: [15], 14: [16], 15: [17], 16: [18], 17: [19], 18: [20, 21],
#               19: [22], 20: [23], 21: [24], 22: [25], 23: [26, 27, 28], 24: [29], 25: [30], 26: [31],
#               27: [32], 28: [34], 29: [35, 36, 37], 30: [38], 31: [39], 32: [40], 33: [41], 34: [42],
#               35: [43], 36: [44], 37: [45], 38: [46], 39: [47], 40: [48], 41: [49, 50, 51, 52], 42: [53],
#               43: [54], 44: [55], 45: [56], 46: [57], 47: [58], 48: [59], 49: [60], 50: [62], 51: [63],
#               52: [64], 53: [66], 54: [67], 55: [68], 56: [69], 57: [70], 58: [71], 59: [72], 60: [73],
#               61: [74], 62: [75], 63: [76], 64: [77], 65: [78], 66: [79], 67: [80], 68: [82], 69: [83],
#               70: [84], 71: [85], 72: [86]}}
    # target=torch.randint(0,10,(2, res, res))#.to(device)
    # loss=hitree_loss(map_dict,[4,5,7,8],[172],device)
    # m=loss(sigmoid,target)
    # print(m.data)