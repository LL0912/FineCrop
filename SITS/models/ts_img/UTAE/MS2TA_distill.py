import sys
sys.path.append("..")
import torch
import torch.nn as nn
from .UTAE import ConvBlock,DownConvBlock,Temporal_Aggregator,ConvLayer
from .ltae import LTAE2d
from SITS.utils.registry import MODELS
@MODELS.register_module(name="ms2ta_distill")
class Multiscale_spatemporal_attention_distill(nn.Module):
    def __init__(self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[64, 64, 64, 128],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        num_labelevel=3,
        num_classlevel_s=[6,20,52],
        num_classlevel_t={"t1":[],"t2":[],"t3":[]}
                 ):
        super(Multiscale_spatemporal_attention_distill, self).__init__()

        self.num_labelevel = num_labelevel
        assert len(num_classlevel_s) == self.num_labelevel
        self.n_stages = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

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

        self.temporal_encoder = nn.ModuleList(
            LTAE2d(
            in_channels=encoder_widths[i],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[i]],
            return_att=True,
            d_k=d_k,
        ) for i in range(self.n_stages)
        )

        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        self.out_conv_dict = nn.ModuleList()
        for i in range(num_labelevel):
            out_conv = [decoder_widths[0]] + [num_classlevel_s[i]]
            self.out_conv_dict.append(ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode))

        self.out_conv_dict_teacher=nn.ModuleDict()
        for key,num_class in num_classlevel_t.items():
            out_classifier_sub_t = nn.ModuleList()
            for i in range(num_labelevel):
                out_conv = [decoder_widths[0]] + [num_class[i]]
                out_classifier_sub_t.append(ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode))
            self.out_conv_dict_teacher[key]=out_classifier_sub_t


    def forward(self, input, batch_positions=None, return_att=False):
        B, T, C, H, W = input.shape
        batch_positions = batch_positions[0, :].reshape(1, -1)
        batch_positions = batch_positions.repeat(B, 1)
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input)

        att_ts=[]
        att_outs=[]

        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out_mul,att_mul=self.temporal_encoder[i](feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
            att_ts.append(att_mul)
            att_outs.append(out_mul)
        # TEMPORAL ENCODER
        out, att_mul = self.temporal_encoder[-1](
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        att_ts.append(att_mul)

        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att_ts[-(i+1)]
            )
            out = self.up_blocks[i](out, skip, att_outs[-(i+1)])

        if self.num_labelevel==3:
            out1_t = {}
            out2_t = {}
            out3_t = {}
            out_1= self.out_conv_dict[0](out)
            out_2 = self.out_conv_dict[1](out)
            out_3 = self.out_conv_dict[2](out)
            for key in self.out_conv_dict_teacher.keys():
                out1_t[key] = self.out_conv_dict_teacher[key][0](out)
                out2_t[key] = self.out_conv_dict_teacher[key][1](out)
                out3_t[key] = self.out_conv_dict_teacher[key][2](out)
            if return_att:
                return out1_t,out2_t,out3_t,(out_1, out_2, out_3), att_ts
            else:
                return out1_t,out2_t,out3_t,(out_1, out_2, out_3)

        if self.num_labelevel == 4:
            out_s={}
            out1_t = {}
            out2_t = {}
            out3_t = {}
            out4_t = {}
            for i in range(self.num_labelevel):
                out_s[i]=self.out_conv_dict[i](out)
            for key in self.out_conv_dict_teacher.keys():
                out1_t[key] = self.out_conv_dict_teacher[key][0](out)
                out2_t[key] = self.out_conv_dict_teacher[key][1](out)
                out3_t[key] = self.out_conv_dict_teacher[key][2](out)
                out4_t[key] = self.out_conv_dict_teacher[key][3](out)
            if return_att:
                return out1_t,out2_t,out3_t,out4_t,(out_s[0], out_s[1], out_s[2], out_s[3]), att_ts
            else:
                return out1_t,out2_t,out3_t,out4_t,(out_s[0], out_s[1], out_s[2], out_s[3])


        if self.num_labelevel==1:
            out_g_s = self.out_conv_dict[0](out)
            out1_t = {}
            for key in self.out_conv_dict_teacher.keys():
                out1_t[key] = self.out_conv_dict_teacher[key][0](out)
            if return_att:
                return out1_t,out_g_s, att_ts
            else:
                return out1_t,out_g_s

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
        self.att_conv=nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )

        self.conv1 = ConvLayer(
            nkernels=[d_out + d + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip, att_out):
        #使用门控机制进行控制，最后输出
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip),self.att_conv(att_out)], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out



if __name__=="__main__":
    res = 24

    x = torch.rand((2, 14, 4, res, res))#.to(device)
    doy = torch.linspace(1, 356, 14).to(torch.int64).reshape(1, -1)  # .to(device)
    model=Multiscale_spatemporal_attention_distill(input_dim=4,num_labelevel=4,num_classlevel_s=[4,5,7,8],num_classlevel_t={"t1":[2,3,3,4],"t2":[2,2,4,4]})


    # z1_t,z2_t,z3_t,z4_t,z_s=model(x,doy)
    # z1_s, z2_s, z3_s, z4_s = z_s
    # for v in z1_t.values():
    #     print(v.shape)
    # print(z1_s.shape)
    # print(z2_s.shape)
    # print(z3_s.shape)
    # print(z4_s.shape)
