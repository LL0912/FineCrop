import torch
from torch import nn
import random
from einops import rearrange
import numpy as np
from torch.autograd import Variable
from SITS.utils.registry import LOSS_FUNCTIONS

"""
channel devil loss
"""

def Mask(nb_batch,cnum, num_class):
    foo = [1] * (cnum-1) + [0] * 1
    bar = []
    for i in range(num_class):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,1, num_class * cnum)
    bar = torch.from_numpy(bar)
    bar = Variable(bar)
    return bar

@LOSS_FUNCTIONS.register_module(name="ChanneldevilLoss")
class ChannelDevLoss(nn.Module):
    def __init__(self, cnum,num_class,use_mask=False,ignore_index=255):
        super(ChannelDevLoss, self).__init__()
        self.ignore_index = ignore_index
        self.use_mask = use_mask
        self.cnum = cnum
        self.num_class = num_class

        self.criterion=nn.CrossEntropyLoss()

    def forward(self, features, target,device):


        """

        :param outputs: [bs,c*s,h,w]
        :param target:
        :return:
        """
        n,c,h,w=features.size()
        fea=rearrange(features,'n c h w -> n (h w) c')

        if self.use_mask:
            mask=Mask(fea.size(0), self.cnum, self.num_class).to(device)
            fea=fea*mask

        branch_dis=nn.MaxPool1d(self.cnum,stride=self.cnum)(fea)
        branch_dis=rearrange(branch_dis,'n (h w) c -> (n h w) c',n=n,h=h,w=w)

        mask = torch.nonzero(target.view(-1) != self.ignore_index)
        target = torch.index_select(target.view(-1, 1), 0, mask.view(-1))
        branch_dis = torch.index_select(branch_dis, 0, mask.view(-1))
        loss_dis = self.criterion(branch_dis, target.view(-1))

        branch_div=torch.softmax(fea,1)
        branch_div = nn.MaxPool1d(self.cnum, stride=self.cnum)(branch_div)
        loss_div=1.0-1.0*torch.mean(torch.sum(branch_div,1))/self.num_class

        return loss_dis,loss_div

@LOSS_FUNCTIONS.register_module(name="ChannelDev_Pixel_Loss")
class ChannelDev_Pixel_Loss(nn.Module):
    def __init__(self, cnum,num_class,use_mask=False,ignore_index=255):
        super(ChannelDev_Pixel_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.use_mask = use_mask
        self.cnum = cnum
        self.num_class = num_class

        self.criterion=nn.CrossEntropyLoss()

    def forward(self, features, target,device):
        """
        :param features: [bs,c*s]
        :param target:
        :return:
        """
        n,c=features.size()
        # fea=rearrange(features,'n c h w -> n (h w) c')

        if self.use_mask:
            mask=Mask(features.size(0), self.cnum, self.num_class).to(device)
            fea=features*mask

        branch_dis=nn.MaxPool1d(self.cnum,stride=self.cnum)(features)
        # branch_dis=rearrange(branch_dis,'n (h w) c -> (n h w) c',n=n,h=h,w=w)

        mask = torch.nonzero(target.view(-1) != self.ignore_index)
        target = torch.index_select(target.view(-1, 1), 0, mask.view(-1))
        branch_dis = torch.index_select(branch_dis, 0, mask.view(-1))
        loss_dis = self.criterion(branch_dis, target.view(-1))

        branch_div=torch.softmax(features,1)
        branch_div = nn.MaxPool1d(self.cnum, stride=self.cnum)(branch_div)
        loss_div=1.0-1.0*torch.mean(torch.sum(branch_div,1))/self.num_class

        return loss_dis,loss_div
        

@LOSS_FUNCTIONS.register_module(name="ChaneldeviParcelLoss")
class ChanelDevParcelLoss(nn.Module):
    def __init__(self, cnum,num_class,use_mask=False,ignore_index=255):
        super(ChanelDevParcelLoss, self).__init__()
        self.ignore_index = ignore_index
        self.use_mask = use_mask
        self.cnum = cnum
        self.num_class = num_class

        self.criterion = nn.CrossEntropyLoss()
    def forward(self,features,target,parcel,device):
        """
        :param feat: [bs,t,c,h,w]
        :param target: [bs,h,w]
        :return:
        """
        n,c,h,w=features.size()
        fea=rearrange(features,'n c h w -> n (h w) c')

        if self.use_mask:
            mask=Mask(fea.size(0), self.cnum, self.num_class).to(device)
            fea=fea*mask

        branch_dis=nn.MaxPool1d(self.cnum,stride=self.cnum)(fea)
        branch_dis=rearrange(branch_dis,'n (h w) c -> (n h w) c',n=n,h=h,w=w)

        mask = torch.nonzero(target.view(-1) != self.ignore_index)
        target = torch.index_select(target.view(-1, 1), 0, mask.view(-1))
        parcel = torch.index_select(parcel.view(-1), 0, mask.view(-1))
        branch_dis = torch.index_select(branch_dis, 0, mask.view(-1))

        unique_parcel=torch.unique(parcel)
        if len(unique_parcel) == 0:
            loss_dis= torch.tensor(0.0).to(device)
        else:
            target_parcel = torch.zeros(len(unique_parcel)).to(device)
            parcel_pred_avg_all = torch.zeros((len(unique_parcel), branch_dis.size(1))).to(device)
            for i, parcel_id in enumerate(unique_parcel):
                parcel_mask = parcel == parcel_id
                target_parcel[i] = target[parcel_mask][0]
                parcel_pred_avg_all[i] = torch.mean(branch_dis[parcel_mask], 0)
            loss_dis = self.criterion(parcel_pred_avg_all, target_parcel.view(-1).long())

        branch_div=torch.softmax(fea,1)
        branch_div = nn.MaxPool1d(self.cnum, stride=self.cnum)(branch_div)
        loss_div=1.0-1.0*torch.mean(torch.sum(branch_div,1))/self.num_class
        loss_div=loss_div.to(device)
        return loss_dis,loss_div

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.randn(6, 40, 24, 24).to(device)
    target = torch.randint(0, 10, (6, 24, 24)).to(device)
    loss_model = ChannelDevLoss(4, 10, True, 255)
    loss_dis,loss_div = loss_model(input, target, device)
    alpha=0.15
    beta=20
    loss_all=loss_dis*alpha+loss_div*beta
    print(loss_dis.data)
    print(loss_div.data)
    print(loss_all.data)