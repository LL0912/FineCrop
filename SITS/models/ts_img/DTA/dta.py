#时间上的偏移
import torch
from torch import nn
from einops import rearrange,repeat
import torch.nn.functional as F
import numpy as np
class Deformable_spatemp_atten(nn.Module):
    def __init__(self,
                 input_channel,
                 n_heads,
                 offset_range_factor=-1,
                 use_pe=True,
                 no_off=False,
                 dwc_pe=False,
                 log_cpb=True,
                 stride=[2,8],
                 ksize=[3,9],
                 reduce_ratio=4

                 ):
        super(Deformable_spatemp_atten,self).__init__()
        self.nc=input_channel
        self.n_heads=n_heads
        self.h_head_channels = self.nc // n_heads
        self.scale = self.h_head_channels ** -0.5
        self.offset_range_factor=offset_range_factor
        self.stride = stride
        self.kk =ksize
        self.no_off=no_off
        pad_size_t = self.kk[0] // 2 if self.kk[0] != stride[0] else 0
        pad_size_hw = self.kk[1] // 2 if self.kk[1] != stride[1] else 0
        self.use_pe=use_pe
        self.dwc_pe=dwc_pe
        self.no_off=no_off
        self.log_cpb = log_cpb

        self.inconv = nn.Conv3d(input_channel, self.nc, kernel_size=1, stride=1, padding=0)

        self.convq = nn.Conv3d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.conv_offset=nn.Sequential(
            nn.Conv3d(self.h_head_channels, self.h_head_channels, (self.kk[0],self.kk[1],self.kk[1]), (stride[0],stride[1],stride[1]),
                      (pad_size_t,pad_size_hw,pad_size_hw), groups=self.h_head_channels),
            LayerNormProxy(self.h_head_channels),
            nn.GELU(),
            nn.Conv3d(self.h_head_channels, 3, 1, 1, 0, bias=False)
        )

        self.conv_k = nn.Conv3d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.conv_v = nn.Conv3d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv3d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                            nn.Linear(3, 32, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(32, 1, bias=False)
                )
        self.t_atten=nn.Sequential(
            nn.Linear(self.nc, self.nc//reduce_ratio, bias=True),
            nn.ReLU(),
            nn.Linear( self.nc // reduce_ratio,1, bias=True),
            nn.Sigmoid()
        )
    @torch.no_grad()
    def _get_ref_points(self, T_key,H_key, W_key, B, dtype, device):

        ref_t, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T_key - 0.5, T_key, dtype=dtype, device=device),
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
        )
        ref = torch.stack((ref_t,ref_y, ref_x), -1)
        ref[..., 0].div_(T_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 1].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 2].div_(W_key - 1.0).mul_(2.0).sub_(1.0)

        ref = ref[None, ...].expand(B * self.n_heads, -1, -1, -1, -1) # B * g, T, H, W, 3

        return ref

    @torch.no_grad()
    def _get_q_grid(self, T, H, W, B, dtype, device):

        ref_t, ref_y, ref_x = torch.meshgrid(
            torch.arange(0, T, dtype=dtype, device=device),
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
        )
        ref = torch.stack((ref_t,ref_y, ref_x), -1)
        ref[..., 0].div_(T - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)  #归一化操作将范围控制在(0,1)之间，到(0,2)之间，再到(-1,1)之间
        ref[..., 2].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_heads,-1, -1, -1, -1) # B * g H W 2

        return ref


    def forward(self,x):
        sz_b, seq_len, dims, h, w = x.shape
        dtype, device = x.dtype, x.device

        x=self.inconv(rearrange(x, "b t c h w -> b c t h w"))
        q = self.convq(x) #q: [b, c, t, h, w]

        q_off=rearrange(q,"b (n_h n_c) t h w -> (b n_h) n_c t h w", n_h=self.n_heads,n_c=self.h_head_channels)

        offset = self.conv_offset(q_off).contiguous()
        Tk,Hk, Wk = offset.size(2),offset.size(3), offset.size(4)

        n_sample=Tk*Hk*Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Tk - 1.0), 1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 3, 1, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)#[b*g,3,t,h,w]

        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tk, Hk, Wk, sz_b, dtype, device) #[B * g, T, H, W, 3],reference的范围是[-1,1]


        if self.no_off:
            offset = offset.fill_(0.0)
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)


        if self.no_off:
            x_sampled = F.avg_pool3d(rearrange(x, "b t c h w -> b c t h w"), kernel_size=(self.stride[0],self.stride[1],self.stride[1]), stride=(self.stride[0],self.stride[1],self.stride[1]),padding=(1,0,0))
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(sz_b * self.n_heads, self.h_head_channels, seq_len, h, w),
                grid=pos[...,(0,2,1)], # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(sz_b, self.nc, Tk, Hk, Wk)

        q = q.reshape(sz_b * self.n_heads,self.h_head_channels,seq_len* h* w)
        k = self.conv_k(x_sampled).reshape(sz_b * self.n_heads, self.h_head_channels, n_sample)
        v = self.conv_v(x_sampled).reshape(sz_b * self.n_heads, self.h_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(sz_b, self.nc,seq_len, h, w))
            elif self.log_cpb:
                q_grid = self._get_q_grid(seq_len, h, w, sz_b, dtype, device)
                displacement = (
                            q_grid.reshape(sz_b * self.n_heads, seq_len*h * w, 3).unsqueeze(2) - pos.reshape(sz_b * self.n_heads, n_sample,
                                                                                                   3).unsqueeze(1)).mul(4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + rearrange(attn_bias, 'b m n h -> (b h) m n', h=1)


        attn = F.softmax(attn, dim=2)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        out=out.reshape(sz_b,self.nc,seq_len, h, w)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        #接一个注意力,消去时间t
        out=rearrange(out,"b c t h w -> b t c h w")

        return out



class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        return rearrange(x, 'b t h w c -> b c t h w')


if __name__=="__main__":
    model=Deformable_spatemp_atten(10,
                 4,
                 16)
    x = torch.randn(2,11,10, 32, 32)
    out=model.forward(x)
    print(out.shape)
