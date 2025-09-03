import sys
sys.path.append("")
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from module import Attention, PreNorm, FeedForward
import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x,td=None):

        if isinstance(td,list):
            for i, (attn, ff) in enumerate(self.layers):
                x = attn(self.norm(x),td[i])+x
                x = ff(self.norm(x)) + x
        else:
            for attn, ff in self.layers:
                x = attn(self.norm(x),td)+x
                x = ff(self.norm(x)) + x
        return self.norm(x)

class Decode_Block(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.linear = nn.Linear(inplanes, inplanes, bias=False)
        self.linear2 = nn.Linear(inplanes, inplanes, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        return x, out

class msTSViTnetAT(nn.Module):
    def __init__(self,

                 image_size=24,
                 patch_size=2,
                 num_frames=71,
                 dim=128,
                 num_classes_g=52,
                 num_classes_1=6,
                 num_classes_2=20,
                 temporal_depth=4,
                 spatial_depth=4,
                 heads=4,
                 dim_head=32,
                 dropout=0.5,
                 emb_dropout=0.5,
                 pool='cls',
                 scale_dim=4,
                 num_channels=4,
                 dt_model="dt",

                 ):
        super().__init__()
        self.image_size =image_size
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes_g = num_classes_g
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.num_frames = num_frames
        self.dim = dim

        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.pool = pool
        self.scale_dim = scale_dim
        self.dt_mode=dt_model
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = num_channels * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            ##rearrange将张量进行变形操作
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)

        self.temporal_token_g = nn.Parameter(torch.randn(1, self.num_classes_g, self.dim))
        self.temporal_token_1 = nn.Parameter(torch.randn(1, self.num_classes_1, self.dim))
        self.temporal_token_2 = nn.Parameter(torch.randn(1, self.num_classes_2, self.dim))


        self.prompt_g=nn.parameter.Parameter(torch.randn(self.dim), requires_grad=True)
        self.prompt_1=nn.parameter.Parameter(torch.randn(self.dim), requires_grad=True)
        self.prompt_2=nn.parameter.Parameter(torch.randn(self.dim), requires_grad=True)

        if dt_model=="dt":
            self.conv12 =nn.Conv1d(self.num_classes_1,self.num_classes_2+num_frames,1)
            self.conv23 =nn.Conv1d(self.num_classes_2,self.num_classes_g+num_frames,1)
            self.conv31 =nn.Conv1d(self.num_classes_g,self.num_classes_1+num_frames,1)
        if dt_model=="td":
            self.conv32 = nn.Conv1d(self.num_classes_g, self.num_classes_2+num_frames, 1)
            self.conv21 = nn.Conv1d(self.num_classes_2, self.num_classes_1+num_frames, 1)
            self.conv13 = nn.Conv1d(self.num_classes_1, self.num_classes_g+num_frames, 1)

        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)

        self.decoders = nn.ModuleList([Decode_Block(self.dim) for _ in range(self.temporal_depth)])

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

        # xg=self.forward_feature(x,self.temporal_token_g,self.space_pos_embedding,B,H,W,self.num_classes_g)
        # x1 = self.forward_feature(x, self.temporal_token_1, self.space_pos_embedding, B, H, W, self.num_classes_1)
        # x2 = self.forward_feature(x, self.temporal_token_2, self.space_pos_embedding, B, H, W, self.num_classes_2)
        # return xg,x1,x2

        xg = self.forward_tempfeature(x,self.temporal_token_g,B, self.num_classes_g,None)
        x1 = self.forward_tempfeature(x, self.temporal_token_1,  B,  self.num_classes_1,None)
        x2 = self.forward_tempfeature(x, self.temporal_token_2, B,  self.num_classes_2,None)

        #category information to interact with prompt, select the information from last category
        x1_c = self.cate_intro_td(x1,self.prompt_1)
        x2_c = self.cate_intro_td(x2, self.prompt_2)
        x3_c = self.cate_intro_td(xg, self.prompt_g)

        if self.dt_mode=="dt":
            k2_c = self.conv12(x1_c)
            k3_c = self.conv23(x2_c)
            k1_c = self.conv31(x3_c)
        elif self.dt_mode=="td":
            k2_c=self.conv32(x3_c)
            k1_c = self.conv21(x2_c)
            k3_c = self.conv13(x1_c)
        elif self.dt_mode=="self":
            k1_c = x1_c
            k2_c = x2_c
            k3_c = x3_c

        ##decoder解码
        td_2 = self.feedback(k2_c)
        td_3 = self.feedback(k3_c)
        td_1 = self.feedback(k1_c)

        ##层级间的注意力
        xg=self.forward_tempfeature(x, self.temporal_token_g, B, self.num_classes_g, td_3)
        x1 = self.forward_tempfeature(x, self.temporal_token_1, B, self.num_classes_1, td_1)
        x2 = self.forward_tempfeature(x, self.temporal_token_2, B, self.num_classes_2, td_2)

        #空间注意力
        xg = self.forward_spacefeature(xg,self.space_pos_embedding,B,H,W,self.num_classes_g)
        x1 = self.forward_spacefeature(x1, self.space_pos_embedding, B, H, W, self.num_classes_1)
        x2 = self.forward_spacefeature(x2, self.space_pos_embedding, B, H, W, self.num_classes_2)

        return xg,x1,x2
    def feedback(self, x):
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td
        return td

    def cate_intro_td(self,cls_token,prompt):
        #先做相似度计算，再做一个conv
        cos_sim = F.normalize(cls_token, dim=-1) @ F.normalize(prompt[None, ..., None], dim=1) #(bhw),(k),dim
        mask = cos_sim.clamp(0, 1)
        x = cls_token*mask #(bhw),(k),dim
        return x

    def forward_tempfeature(self,x,temporal_token,B,num_classes,td=None):
        # 加入分类token
        cls_temporal_tokens = repeat(temporal_token, '() N d -> b N d',
                                     b=B * self.num_patches_1d ** 2)  # (bhw),k,dim
        x = torch.cat((cls_temporal_tokens, x), dim=1)  # 加入分类的token，输出(bhw),(t+k),dim

        # 时间的transformer
        x = self.temporal_transformer.forward(x,td)  # (bhw),(t+k),dim
        x = x[:, : num_classes]  # (bhw),(k),dim
        return x
        ##k和prompt进行相乘，构成mask
    def forward_spacefeature(self,x,space_pos_embedding,B,H,W,num_classes):
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
    # def get_device(device_ids, allow_cpu=False):
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda:%d" % device_ids[0])
    #     elif allow_cpu:
    #         device = torch.device("cpu")
    #     else:
    #         sys.exit("No allowed device is found")
    #     return device


    res = 24
    gpu_ids = ["2","3"]
    # device_ids = [int(i) for i in gpu_ids if i.isnumeric()]
    # device = get_device(device_ids, allow_cpu=False)
    x = torch.rand((2, 16, 4, res, res))#.to(device)

    model = msTSViTnetAT( num_frames=16, num_channels=4)#.to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)  # 就在这里wrap一下，模型就会使用所有的GPU
    # model.to(device)
    doy = torch.linspace(1, 356, 16).to(torch.int64).reshape(1,-1)#.to(device)
    xg, x1, x2 = model(x,doy)
    print(xg.shape)
    print(x1.shape)
    print(x2.shape)


    # x = torch.rand((2, 4, 16))
    # td = torch.rand((2, 4, 16))
    # # model=Transformer(dim=16, depth=2, heads=4, dim_head=12, mlp_dim=12, dropout=0.)
    # model=Attention(dim=16)
    # out=model(x)