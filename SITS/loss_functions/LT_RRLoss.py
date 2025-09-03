import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from einops import rearrange
from SITS.utils.registry import LOSS_FUNCTIONS


def balanced_softmax_loss(logits,labels, sample_per_class, weight):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, weight=weight)
    return loss
    
    
@LOSS_FUNCTIONS.register_module(name="PRSoftBalanceLoss")
class ParcelRebalancedSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(ParcelRebalancedSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, parcel,cls_num_list,device=None,weight=None):

        self.m_list = torch.from_numpy(cls_num_list).to(device)
        pred = rearrange(pred, "n c h w -> (n h w) c").double()
        num_classes = pred.size(1)
        mask = torch.nonzero(target.view(-1) != self.ignore_index)
        pred = torch.index_select(pred, 0, mask.view(-1))

        target = torch.index_select(target.view(-1), 0, mask.view(-1))
        parcel = torch.index_select(parcel.view(-1), 0, mask.view(-1))
        unique_parcel=torch.unique(parcel)
        
        if len(unique_parcel)==0:
            return torch.tensor(0.0)
        else:
            target_parcel=torch.zeros(len(unique_parcel)).to(device)
            parcel_pred_avg_all=torch.zeros((len(unique_parcel),num_classes)).to(device)

            for i,parcel_id in enumerate(unique_parcel):
                parcel_mask=parcel==parcel_id
                #parcel_target=target[parcel_mask]
                parcel_pred=pred[parcel_mask]

                parcel_pred_avg=torch.mean(parcel_pred,0)
                target_parcel[i]=target[parcel_mask][0]
                parcel_pred_avg_all[i]=parcel_pred_avg
            if weight is not None:
                weight=weight.float()
            parcel_loss = balanced_softmax_loss(
                    parcel_pred_avg_all, target_parcel.long(), self.m_list,weight=weight)
   
            return parcel_loss
            
@LOSS_FUNCTIONS.register_module(name="PRLDAMLoss")            
class ParcelRebalancedLDAM(nn.Module):
    def __init__(self,max_m=0.5, s=30,ignore_index=None):
        super(ParcelRebalancedLDAM, self).__init__()
        self.ignore_index = ignore_index
        self.s=s
        self.max_m = max_m

    def forward(self, pred, target, parcel, cls_num_list, device=None, weight=None):
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))

        # m_list = 1.0 /cls_num_list
        # m_list = 1.0 / np.power(cls_num_list,2)
        m_list = m_list * (self.max_m / np.max(m_list))
        self.m_list = torch.from_numpy(m_list).to(device)

        pred = rearrange(pred, "n c h w -> (n h w) c").double()
        num_classes = pred.size(1)
        
        mask = torch.nonzero(target.view(-1) != self.ignore_index)
        pred = torch.index_select(pred, 0, mask.view(-1))

        target = torch.index_select(target.view(-1), 0, mask.view(-1))
        parcel = torch.index_select(parcel.view(-1), 0, mask.view(-1))
        unique_parcel = torch.unique(parcel)

        if len(unique_parcel) == 0:
            return torch.tensor(0.0)
        else:
            target_parcel = torch.zeros(len(unique_parcel)).to(device)
            parcel_pred_avg_all = torch.zeros((len(unique_parcel), num_classes)).to(device)

            for i, parcel_id in enumerate(unique_parcel):
                parcel_mask = parcel == parcel_id
                # parcel_target=target[parcel_mask]
                parcel_pred = pred[parcel_mask]

                parcel_pred_avg = torch.mean(parcel_pred, 0)
                target_parcel[i] = target[parcel_mask][0]
                parcel_pred_avg_all[i] = parcel_pred_avg

            index = torch.zeros_like(parcel_pred_avg_all, dtype=torch.uint8)
            index.scatter_(1, target_parcel.view(-1, 1).long(), 1)
            index_float = index.double().to(device)
            
            batch_m = torch.matmul(self.m_list[None, :].double(), index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            p_m = parcel_pred_avg_all - batch_m

            output = torch.where(index, p_m.float(), parcel_pred_avg_all)

            if weight is not None:
                weight = weight.float()

            parcel_loss = F.cross_entropy(self.s * output, target_parcel.long(), weight=weight)

            return parcel_loss

if __name__ == "__main__":

    # import pandas as pd
    # # 读取Excel文件
    # excel_file  = pd.ExcelFile(r'G:\01_dataset\public data\Eurocrop\SK\statistic\train_field_count_class_all.xlsx')
    # sheet_name=excel_file .sheet_names[2]
    # df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # column_data = df.iloc[:, 0]
    # # 遍历第一列数据并输出
    # for value in column_data:
    #     print(str(value) + ',', end='')


    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 73, 32, 32)
    label_all = np.load(
        r"G:\01_dataset\public data\Eurocrop\SK\NPY_mini_norm_crop_small_2\train\33UYP-0000010240-0000010240_119.npz")["label"]

    label=label_all[:,:,2]
    parcel=label_all[:,:,4]
    # label=np.ones((1,32,32))*255
    cls_num_list=[1398,831,305,4100,558,676,175,318,117,1,1,6,423,12,244,6,3,1,2264,2026,145,7,14,110,62,100,10,67,2,1813,867,2314,66,7,3,10,207,26,326,9,164,79,11,54,7,22,10,31,51,305,6,281,1336,162,41,25,38,18,66,3,8,9,2,0,1,50,1174,14,3,4,74,12,140]

    l=ParcelRebalancedSoftmaxLoss(ignore_index=255)
    out=l(x,  torch.from_numpy(label), torch.from_numpy(parcel),cls_num_list=np.asarray(cls_num_list),device=device)

    print(out)