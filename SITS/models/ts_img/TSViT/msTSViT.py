import sys
sys.path.append("")

from torch import nn
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from module import Attention, PreNorm, FeedForward
import torch.nn.functional as F
#结构：
#使用三个并行的分支提取多级别的特征，并进行特征间的交互和分类结果的约束
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class msTSViTnet(nn.Module):
    def __init__(self,
                 image_size=24,
                 patch_size=2,
                 dim=128,
                 temporal_depth=4,
                 spatial_depth=4,
                 heads=4,
                 dim_head=32,
                 dropout=0.5,
                 emb_dropout=0.5,
                 pool='cls',
                 scale_dim=4,
                 num_channels=4,
                 num_labelevel=3,
                 num_class=[6,20,52]
                 ):
        super().__init__()
        self.image_size =image_size
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.dim = dim

        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.pool = pool
        self.scale_dim = scale_dim

        self.num_class=num_class
        self.num_labelevel=num_labelevel

        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = num_channels * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            ##rearrange将张量进行变形操作
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)

        if self.num_labelevel==3:
            self.num_classes_g=num_class[2]
            self.num_classes_1 = num_class[0]
            self.num_classes_2 = num_class[1]
            self.temporal_token_g = nn.Parameter(torch.randn(1, self.num_classes_g, self.dim))
            self.temporal_token_1 = nn.Parameter(torch.randn(1, self.num_classes_1, self.dim))
            self.temporal_token_2 = nn.Parameter(torch.randn(1, self.num_classes_2, self.dim))
        elif self.num_labelevel==4:
            self.temporal_token_1 = nn.Parameter(torch.randn(1, self.num_class[0], self.dim))
            self.temporal_token_2 = nn.Parameter(torch.randn(1, self.num_class[1], self.dim))
            self.temporal_token_3 = nn.Parameter(torch.randn(1, self.num_class[2], self.dim))
            self.temporal_token_4 = nn.Parameter(torch.randn(1, self.num_class[3], self.dim))

        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))  # [1,hw,dim]
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                             self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )
    def forward(self,x,doy):

        B, T, C, H, W = x.shape
        doy = doy[0, :].reshape(1, -1)
        doy = doy.repeat(B, 1)

        xt = F.one_hot(doy, num_classes=366).to(torch.float32)  # 时间维度的特征取出来
        xt = xt.reshape(-1, 366)

        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)  ##时间编码

        # 打成patch后计算linear的特征 (b h w) ,t ,dim
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)  # b,(hw),t,dim

        #加入时间编码
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)#(bhw),t,dim

        # #加入分类token
        # cls_temporal_tokens_g = repeat(self.temporal_token_g, '() N d -> b N d', b=B * self.num_patches_1d ** 2)#(bhw),k,dim
        # xg = torch.cat((cls_temporal_tokens_g, x), dim=1) #加入分类的token，输出(bhw),(t+k),dim
        #
        # cls_temporal_tokens_1 = repeat(self.temporal_token_1, '() N d -> b N d', b=B * self.num_patches_1d ** 2)#(bhw),k,dim
        # x1 = torch.cat((cls_temporal_tokens_1, x), dim=1) #加入分类的token，输出(bhw),(t+k),dim
        #
        # cls_temporal_tokens_2 = repeat(self.temporal_token_2, '() N d -> b N d', b=B * self.num_patches_1d ** 2)#(bhw),k,dim
        # x2 = torch.cat((cls_temporal_tokens_2, x), dim=1) #加入分类的token，输出(bhw),(t+k),dim
        #
        # #时间的transformer
        # xg = self.temporal_transformer(xg)#(bhw),(t+k3),dim
        # xg = xg[:, :self.num_classes_g]#(bhw),(k3),dim
        # xg = xg.reshape(B, self.num_patches_1d**2, self.num_classes_g, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim) #(bk),(hw),dim
        #
        # x1 = self.temporal_transformer(x1)#(bhw),(t+k2),dim
        # x1 = x1[:, :self.num_classes_1]#(bhw),(k2),dim
        # x1 = x1.reshape(B, self.num_patches_1d**2, self.num_classes_1, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim) #(bk),(hw),dim
        #
        # x2 = self.temporal_transformer(x2)#(bhw),(t+k1),dim
        # x2 = x2[:, :self.num_classes_2]#(bhw),(k1),dim
        # x2 = x2.reshape(B, self.num_patches_1d**2, self.num_classes_2, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim) #(bk),(hw),dim
        #
        # # 加入空间编码
        # xg += self.space_pos_embedding  # (bk3),(hw),dim
        # xg = self.dropout(xg)
        #
        # x1 += self.space_pos_embedding  # (bk2),(hw),dim
        # x1 = self.dropout(x1)
        #
        # x2 += self.space_pos_embedding  # (bk1),(hw),dim
        # x2 = self.dropout(x2)
        #
        #
        # # 空间transformer
        # xg = self.space_transformer(xg)  # (bk3),(hw),dim
        # x2 = self.space_transformer(x2)  # (bk2),(hw),dim
        # x1 = self.space_transformer(x1)  # (bk1),(hw),dim
        #
        # # 分类头还原
        # xg = self.mlp_head(xg.reshape(-1, self.dim))  # [bkhw,p**2]
        # xg = xg.reshape(B, self.num_classes_g, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3,1)  # [b,k,hw,p**2]->[b,hw,p**2,k]
        # xg = xg.reshape(B,H,W, self.num_classes_g)  # [B,H,W,K]
        # xg = xg.permute(0, 3, 1, 2)  # [B,K,H,W]
        #
        # x2 = self.mlp_head(x2.reshape(-1, self.dim))  # [bkhw,p**2]
        # x2 = x2.reshape(B, self.num_classes_2, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)  # [b,k,hw,p**2]->[b,hw,p**2,k]
        # x2 = x2.reshape(B, H, W, self.num_classes_2)  # [B,H,W,K]
        # x2 = x2.permute(0, 3, 1, 2)  # [B,K,H,W]
        #
        # x1 = self.mlp_head(xg.reshape(-1, self.dim))  # [bkhw,p**2]
        # x1 = x1.reshape(B, self.num_classes_1, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)  # [b,k,hw,p**2]->[b,hw,p**2,k]
        # x1 = x1.reshape(B, H, W, self.num_classes_1)  # [B,H,W,K]
        # x1 = x1.permute(0, 3, 1, 2)  # [B,K,H,W]
        if self.num_labelevel==3:
            xg = self.forward_feature(x, self.temporal_token_g, self.space_pos_embedding, B, H, W, self.num_classes_g)
            x1 = self.forward_feature(x, self.temporal_token_1, self.space_pos_embedding, B, H, W, self.num_classes_1)
            x2 = self.forward_feature(x, self.temporal_token_2, self.space_pos_embedding, B, H, W, self.num_classes_2)
            return xg,x1,x2
        else:
            x1 = self.forward_feature(x, self.temporal_token_1, self.space_pos_embedding, B, H, W, self.num_class[0])
            x2 = self.forward_feature(x, self.temporal_token_2, self.space_pos_embedding, B, H, W, self.num_class[1])
            x3 = self.forward_feature(x, self.temporal_token_3, self.space_pos_embedding, B, H, W, self.num_class[2])
            x4 = self.forward_feature(x, self.temporal_token_4, self.space_pos_embedding, B, H, W, self.num_class[3])
            return x1,x2,x3,x4


    def forward_feature(self,x,temporal_token,space_pos_embedding,B,H,W,num_classes):
        # 加入分类token
        cls_temporal_tokens = repeat(temporal_token, '() N d -> b N d',
                                     b=B * self.num_patches_1d ** 2)  # (bhw),k,dim
        x = torch.cat((cls_temporal_tokens, x), dim=1)  # 加入分类的token，输出(bhw),(t+k),dim

        # 时间的transformer
        x = self.temporal_transformer(x)  # (bhw),(t+k),dim
        x = x[:, : num_classes]  # (bhw),(k),dim
        x = x.reshape(B, self.num_patches_1d ** 2, num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * num_classes, self.num_patches_1d ** 2, self.dim)  # (bk),(hw),dim

        # 加入空间编码
        x += space_pos_embedding  # (bk),(hw),dim
        x = self.dropout(x)

        # 空间transformer
        x = self.space_transformer(x)  # (bk),(hw),dim

        # 分类头还原
        x = self.mlp_head(x.reshape(-1, self.dim))  # [bkhw,p**2]
        x = x.reshape(B, num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3,1)  # [b,k,hw,p**2]->[b,hw,p**2,k]
        x = x.reshape(B, H, W, num_classes)  # [B,H,W,K]
        x = x.permute(0, 3, 1, 2)  # [B,K,H,W]
        return x





if __name__ == "__main__":
    def get_device(device_ids, allow_cpu=False):
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % device_ids[0])
        elif allow_cpu:
            device = torch.device("cpu")
        else:
            sys.exit("No allowed device is found")
        return device


    res = 24
    gpu_ids = ["2","3"]
    # device_ids = [int(i) for i in gpu_ids if i.isnumeric()]
    # device = get_device(device_ids, allow_cpu=False)
    x = torch.rand((2, 16, 4, res, res))#.to(device)

    model = msTSViTnet(num_channels=4,num_labelevel=4,num_class=[3,5,8,6])#.to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)  # 就在这里wrap一下，模型就会使用所有的GPU
    # model.to(device)
    doy = torch.linspace(1, 356, 16).to(torch.int64).reshape(1,-1)#.to(device)
    print(doy.shape)
    x1, x2, x3, x4 = model(x, doy)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)