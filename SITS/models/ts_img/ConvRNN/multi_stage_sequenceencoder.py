import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convstar import ConvSTAR, ConvSTAR_Res
from .convgru import ConvGRU
from .convlstm import ConvLSTM
from SITS.utils.registry import MODELS
# from SITS.utils.config import configurable

@MODELS.register_module(name="convrnn")
class multistageSTARSequentialEncoder(torch.nn.Module):
    # @configurable
    def __init__(self, height, width, input_dim=4, hidden_dim=64,  kernel_size=(3,3), n_layers=6,nstage=3,
                 use_in_layer_norm=False, viz=False, test=False, wo_softmax=False, cell='star',num_classlevel=[6,20,52],use_channel_fg=False,channel_group=None,use_convblock=False):
        super(multistageSTARSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.viz = viz
        self.test = test
        self.wo_softmax = wo_softmax
        self.cell = cell
        #self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))
        assert len(num_classlevel) == self.nstage
        self.use_in_layer_norm = use_in_layer_norm
        self.use_channel_fg = use_channel_fg
        
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)
        

        if self.cell == 'gru':
            self.rnn = ConvGRU(input_size=input_dim,
                                hidden_sizes=hidden_dim,
                                kernel_sizes=kernel_size[0],
                                n_layers=n_layers)
        elif self.cell == 'star_res':
            self.rnn = ConvSTAR_Res(input_size=input_dim,
                                    hidden_sizes=hidden_dim,
                                    kernel_sizes=kernel_size[0],
                                    n_layers=n_layers)
        elif self.cell == 'star':
            self.rnn = ConvSTAR(input_size=input_dim,
                                    hidden_sizes=hidden_dim,
                                    kernel_sizes=kernel_size[0],
                                    n_layers=n_layers)
        elif self.cell == 'lstm':
            self.rnn = ConvLSTM(input_size=[height, width],
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                kernel_size=kernel_size,
                                num_layers=n_layers)

        self.out_conv_dict = torch.nn.ModuleList()
        for i in range(self.nstage):
            if self.use_channel_fg:
                out_conv = num_classlevel[i] * channel_group
            else:
                out_conv = num_classlevel[i]
            if use_convblock:
                self.out_conv_dict.append(ConvBlock([hidden_dim, out_conv]))
            else:
                self.out_conv_dict.append(torch.nn.Conv2d(hidden_dim, out_conv, (3, 3), padding=1))
            

        if self.use_channel_fg:
            self.classifer = torch.nn.ModuleList()
            for i in range(self.nstage):
                if use_convblock:
                    self.classifer.append(ConvBlock([out_conv, num_classlevel[i]]))
                else:
                    self.classifer.append(torch.nn.Conv2d(out_conv, num_classlevel[i], (3, 3), padding=1))
                

        # self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3), padding=1)
        # self.final_local_1 = torch.nn.Conv2d(hidden_dim, nclasses_l1, (3, 3), padding=1)
        # self.final_local_2 = torch.nn.Conv2d(hidden_dim, nclasses_l2, (3, 3), padding=1)
    # @classmethod
    # def from_config(cls,cfg):
    #     return {
    #         "height":cfg.height,
    #         "width":cfg.width,
    #         "input_dim":cfg.input_dim,
    #         "nstage":cfg.nstage,
    #         "cell":cfg.cell,
    #         "num_classlevel":cfg.num_classlevel,
    #         "n_layers":cfg.n_layers,
    #         "use_channel_fg":cfg.use_channel_fg,
    #         "channel_group":cfg.channel_group,
    #         "use_convblock":cfg.use_convblock,
    #     }

    def forward(self, x,device=None, hiddenS=None,return_fea=False):
    
        if self.use_in_layer_norm:
            #(b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0,1,3,4,2)).permute(0,4,1,2,3)
        else:
            # (b x t x c x h x w) -> (b x c x t x h x w)
            if self.cell!="lstm":
                x = x.permute(0,2,1,3,4)
                b, c, t, h, w = x.shape
            else:
                b, t, c, h, w = x.shape
            
        #x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        #x = self.inconv.forward(x)

        #convRNN step---------------------------------
        #hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]

        if self.cell=="lstm":
            if hiddenS is None:
                hiddenS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
                cellS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
            if torch.cuda.is_available():
                if hiddenS is None:
                    hiddenS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
                    cellS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
                for i in range(self.n_layers):
                    hiddenS[i] = hiddenS[i].to(device)
                    cellS[i] = cellS[i].to(device)
            hiddenS, cellS = self.rnn.forward(x, hiddenS, cellS)

        else:
            if hiddenS is None:
                hiddenS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
            if torch.cuda.is_available():
                if hiddenS is None:
                    hiddenS = [torch.zeros((b, self.hidden_dim, h, w)).to(device)] * self.n_layers
                for i in range(self.n_layers):
                    hiddenS[i] = hiddenS[i].to(device)
            for iter in range(t):
                hiddenS = self.rnn.forward(x[:, :, iter, :, :], hiddenS)

        out_conv={}
        fea_fg={}
        if self.use_channel_fg:
            if self.nstage==3:
                if self.n_layers==3:
                    for i in range(self.n_layers):
                        if self.cell!="lstm":
                            fea_fg[i]=self.out_conv_dict[i](hiddenS[i])
                            out_conv[i]=self.classifer[i](fea_fg[i])
                        else:
                            fea_fg[i] = self.out_conv_dict[i](hiddenS[i][:, -1, :, :, :])
                            out_conv[i] = self.classifer[i](fea_fg[i])
                else:
                    idx_level = np.arange(1, self.n_layers, 2)
                    for i,idx in enumerate(idx_level):
                        if self.cell!="lstm":
                            fea_fg[i]=self.out_conv_dict[i](hiddenS[idx])
                            out_conv[i]=self.classifer[i](fea_fg[i])
                        else:
                            fea_fg[i]=self.out_conv_dict[i](hiddenS[idx][:, -1, :, :, :])
                            out_conv[i]=self.classifer[i](fea_fg[i])

            elif self.nstage==4:
                idx_level = np.arange(1, self.n_layers, 2)
                for i, idx in enumerate(idx_level):
                    if self.cell != "lstm":
                        fea_fg[i] = self.out_conv_dict[i](hiddenS[idx])
                        out_conv[i] = self.classifer[i](fea_fg[i])
                    else:
                        fea_fg[i] = self.out_conv_dict[i](hiddenS[idx][:, -1, :, :, :])
                        out_conv[i] = self.classifer[i](fea_fg[i])

            elif self.nstage==1:
                if self.cell != "lstm":
                    fea_fg[0] = self.out_conv_dict[0](hiddenS[-1])
                    out_conv[0] = self.classifer[0](fea_fg[0])
                else:
                    fea_fg[0] = self.out_conv_dict[0](torch.mean(hiddenS[-1],dim=1))
                    out_conv[0] = self.classifer[0](fea_fg[0])
        else:
            if self.nstage == 3:
                if self.n_layers == 3:
                    for i in range(self.n_layers):
                        if self.cell != "lstm":
                            out_conv[i] = self.out_conv_dict[i](hiddenS[i])
                        else:
                            out_conv[i] = self.out_conv_dict[i](hiddenS[i][:, -1, :, :, :])
                else:
                    idx_level = np.arange(1, self.n_layers, 2)
                    for i, idx in enumerate(idx_level):
                        if self.cell != "lstm":
                            out_conv[i] = self.out_conv_dict[i](hiddenS[idx])
                        else:
                            out_conv[i] = self.out_conv_dict[i](hiddenS[idx][:, -1, :, :, :])

            elif self.nstage == 4:
                idx_level = np.arange(1, self.n_layers, 2)
                for i, idx in enumerate(idx_level):
                    if self.cell != "lstm":
                        out_conv[i] = self.out_conv_dict[i](hiddenS[idx])
                    else:
                        out_conv[i] = self.out_conv_dict[i](hiddenS[idx][:, -1, :, :, :])

            elif self.nstage == 1:
                if self.cell != "lstm":
                    out_conv[0] = self.out_conv_dict[0](hiddenS[-1])
                else:
                    out_conv[0] = self.out_conv_dict[0](hiddenS[-1][:, -1, :, :, :])



        if self.nstage==3:
            if self.use_channel_fg:
                if return_fea:
                    return F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1),fea_fg[2],fea_fg[0],fea_fg[1]
                elif self.test:
                    return F.softmax(out_conv[2], dim=1), F.softmax(out_conv[0], dim=1), F.softmax(out_conv[1], dim=1)
                elif self.wo_softmax:
                    return out_conv[2], out_conv[0], out_conv[1]
                else:
                    return F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1)

            else:
                if self.viz:
                    return  hiddenS[-1]
                elif self.test:
                    return F.softmax(out_conv[2], dim=1), F.softmax(out_conv[0], dim=1), F.softmax(out_conv[1], dim=1)
                elif self.wo_softmax:
                    return out_conv[2], out_conv[0], out_conv[1]
                else:
                    return F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1)

        if self.nstage==4:
            if self.use_channel_fg:
                if return_fea:
                    return F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1), F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[3], dim=1),fea_fg[0],fea_fg[1],fea_fg[2],fea_fg[3]
                elif self.test:
                    return F.softmax(out_conv[0], dim=1), F.softmax(out_conv[1], dim=1), F.softmax(out_conv[2], dim=1), F.softmax(out_conv[3], dim=1)
                elif self.wo_softmax:
                    return out_conv[0], out_conv[1], out_conv[2], out_conv[3]
                else:
                    return F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1), F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[3], dim=1)

            else:
                if return_fea and self.cell!="lstm":
                    return hiddenS[1], hiddenS[3], hiddenS[5], hiddenS[7]
                elif return_fea and self.cell=="lstm":
                    return hiddenS[1][:, -1, :, :, :], hiddenS[3][:, -1, :, :, :], hiddenS[5][:, -1, :, :, :], hiddenS[7][:, -1, :, :, :]
                elif self.test:
                    return F.softmax(out_conv[0], dim=1), F.softmax(out_conv[1], dim=1), F.softmax(out_conv[2], dim=1), F.softmax(out_conv[3], dim=1)
                elif self.wo_softmax:
                    return out_conv[0], out_conv[1], out_conv[2], out_conv[3]
                else:
                    return F.log_softmax(out_conv[0], dim=1), F.log_softmax(out_conv[1], dim=1), F.log_softmax(out_conv[2], dim=1), F.log_softmax(out_conv[3], dim=1)

        if self.nstage==1:
            if return_fea:
                if self.use_channel_fg:
                    return F.log_softmax(out_conv[0], dim=1), fea_fg[0]
                else:
                    if self.cell != "lstm":
                        return F.log_softmax(out_conv[0], dim=1), hiddenS[-1]
                    else:
                        return F.log_softmax(out_conv[0], dim=1), hiddenS[-1][:, -1, :, :, :]
            else:
                if self.test:
                    return F.softmax(out_conv[0], dim=1)
                elif self.wo_softmax:
                    return out_conv[0]
                else:
                    return F.log_softmax(out_conv[0], dim=1)



class multistageLSTMSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=4, hidden_dim=64, nclasses=15,
                 nstage=3, nclasses_l1=3, nclasses_l2=7, kernel_size=(3, 3), n_layers=6,
                 use_in_layer_norm=False, viz=False, test=False, wo_softmax=False):
        super(multistageLSTMSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.viz = viz
        self.test = test
        self.wo_softmax = wo_softmax
        # self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))

        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)

        self.rnn = ConvLSTM(input_size=[24,24],
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=n_layers)

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3), padding=1)
        self.final_local_1 = torch.nn.Conv2d(hidden_dim, nclasses_l1, (3, 3), padding=1)
        self.final_local_2 = torch.nn.Conv2d(hidden_dim, nclasses_l2, (3, 3), padding=1)

    def forward(self, x,device=None, hiddenS=None):

        if self.use_in_layer_norm:
            # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
        # else:
        #     # (b x t x c x h x w) -> (b x c x t x h x w)
        #     x = x.permute(0, 2, 1, 3, 4)

        # x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        # x = self.inconv.forward(x)
        b, t, c, h, w = x.shape

        # convRNN step---------------------------------
        # hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            cellS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers

        if torch.cuda.is_available():
            for i in range(self.n_layers):
                hiddenS[i] = hiddenS[i].to(device)
                cellS[i] = cellS[i].to(device)
                # hiddenS[i] = hiddenS[i].to(device)
                # cellS[i] = cellS[i].to(device)

        #for iter in range(t):
        #    hiddenS, cellS = self.rnn.forward(x[:, :, iter, :, :], hiddenS, cellS)
        hiddenS, cellS = self.rnn.forward(x, hiddenS, cellS)

        if self.n_layers == 3:
            local_1 = hiddenS[0]
            local_2 = hiddenS[1]
        elif self.nstage==3:
            local_1 = hiddenS[1]
            local_2 = hiddenS[3]
        elif self.nstage==2:
            local_1 = hiddenS[1]
            local_2 = hiddenS[2]
        elif self.nstage==1:
            local_1 = hiddenS[-1]
            local_2 = hiddenS[-1]

        last = hiddenS[-1]

        local_1 = local_1[:,-1,:,:,:]
        local_2 = local_2[:,-1,:,:,:]
        last = last[:,-1,:,:,:]

        local_1 = self.final_local_1(local_1)
        local_2 = self.final_local_2(local_2)
        last = self.final(last)

        if self.viz:
            return hiddenS[-1]
        elif self.test:
            return F.softmax(last, dim=1), F.softmax(local_1, dim=1), F.softmax(local_2, dim=1)
        elif self.wo_softmax:
            return last, local_1, local_2
        else:
            return F.log_softmax(last, dim=1), F.log_softmax(local_1, dim=1), F.log_softmax(local_2, dim=1)


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

# if __name__=="__main__":
#     model=multistageSTARSequentialEncoder(32, 32, nstage=1,
#                                                   input_dim=4, hidden_dim=64, n_layers=8,
#                                                   wo_softmax=True,cell="star_res",num_class=[3])
#     input=torch.randn(2,10,4,128,128)
#     out4=model(input)
#
#
#     print(out4.shape)
