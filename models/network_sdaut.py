'''
# -----------------------------------------
Network
SDAUT m.1.1
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops
import numpy as np


# ------------------------------
# Swin Deformable Attention Transformer Basic
# ------------------------------

class SwinDAttention(nn.Module):
    r""" Shift Windows Deformable Attention

    Args:
        q_size(tuple[int]): Size if query. Here is the window size.
        kv_size(tuple[int]): Size if key and value. Here is the window size.
        dim (int): Number of input channels.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (tuple[int]): Window size for self-attention.
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        stride (int): Stride in offset calculation network
        offset_range_factor (int): Offset range factor in offset calculation network
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, q_size, kv_size, dim, n_head, n_group, window_size,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 use_pe, dwc_pe, no_off, fixed_pe):
        super().__init__()

        self.dim = dim  # input channel
        self.window_size = window_size  # window height Wh, Window width Ww
        self.n_head = n_head  # number of head
        self.n_head_channels = self.dim // self.n_head  # head_dim
        self.scale = self.n_head_channels ** -0.5

        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size

        self.n_group = n_group
        self.n_group_channels = self.dim // self.n_group
        self.n_group_heads = self.n_head // self.n_group

        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        if self.q_h <= 12 or self.q_w <= 12:
            self.kk = 3
        elif 13 <= self.q_h <= 24 or 13 <= self.q_w <= 24:
            self.kk = 5
        elif 25 <= self.q_h <= 48 or 25 <= self.q_w <= 48:
            self.kk = 7
        else:
            self.kk = 9

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, self.kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_group, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, window_size, mask=None):
        H = window_size
        W = window_size
        B, N, C = x.size()
        dtype, device = x.dtype, x.device
        assert H * W == N, "input feature has wrong size"

        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W)
        # calculate query
        q = self.proj_q(x)  # B C H W
        # resize query
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_group, c=self.n_group_channels)

        # use query to calculate offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # get the size of offset
        Hk, Wk = offset.size(2), offset.size(3)
        # sample number
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # resize offset
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # use the number of offset point and batch size to get reference point
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # no offset
        if self.no_off:
            offset = torch.zeros_like(offset)

        # offset + ref
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # embedding query,key,valuse B,C,H,W --> B*head,head_channel,HW
        q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)

        # Q&K
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # use position encoding
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_head, self.n_head_channels,
                                                                              H * W)
            # fix
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_head, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(
                            B * self.n_group, H * W, 2).unsqueeze(2)
                        - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
                attn = attn + attn_bias

        if mask is not None:
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.n_head, H * W, n_sample) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, H * W, n_sample)
            attn = attn.view(-1, H * W, n_sample)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, 0, 0, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, n_head={self.n_head}, n_group={self.n_group}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj_q(x)
        flops += N * self.dim * self.dim * 1
        if not self.no_off:
            # self.conv_offset(q_off)
            flops += self.n_group * N * self.n_group_channels * (
                    self.n_group_channels / self.n_group_channels) * self.kk * self.kk
            flops += self.n_group * N * self.n_group_channels
            flops += self.n_group * N * self.n_group_channels * 2 * 1
        # self.proj_k(x_sampled)
        flops += N * self.dim * self.dim * 1
        # self.proj_v(x_sampled)
        flops += N * self.dim * self.dim * 1
        # torch.einsum('b c m, b c n -> b m n', q, k)
        flops += self.n_group * N * self.n_group_channels * N
        # torch.einsum('b m n, b c n -> b c m', attn, v)
        flops += self.n_group * N * N * self.n_group_channels
        # self.proj_drop(self.proj_out(out))
        flops += N * self.dim * self.dim * 1
        return flops


class SwinDATransformerBlock(nn.Module):
    r""" Swin Deformable Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, n_head, n_group, window_size, shift_size, mlp_ratio=2.,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.n_head_channels = dim // n_head
        self.n_group = n_group
        self.window_size = window_size
        self.q_h, self.q_w = to_2tuple(window_size)
        self.kv_h, self.kv_w = to_2tuple(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SwinDAttention(
            q_size=to_2tuple(window_size), kv_size=to_2tuple(window_size),
            dim=dim, n_head=n_head, n_group=n_group, window_size=to_2tuple(window_size),
            attn_drop=attn_drop, proj_drop=drop, stride=1, offset_range_factor=2,
            use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C)  # x (batch_in_each_GPU, embedding_channel, H, W)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # shifted_x (batch_in_each_GPU, embedding_channel, H, W)

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, _, __, ___ = self.attn(x_windows, self.window_size,
                                                          mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows, _, __, ___ = self.attn(x_windows, self.window_size,
                                                          mask=self.calculate_mask(x_size).to(
                                                              x.device))  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C shifted_x  (batch_in_each_GPU, embedding_channel, H, W)

        # Running only when batch_size=1 (test) to accelerate.
        # Comment out this to calculation the time cost.
        # if shifted_x.shape[0] == 1:
        #     position = window_reverse_deform_point(position.detach().clone(), self.window_size, H, W)
        #     reference = window_reverse_deform_point(reference.detach().clone(), self.window_size, H, W)
        #     attn_map = window_reverse_attn_map(attn_map.detach().clone(), self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # Comment out this to calculation the time cost.
            # position = torch.roll(position, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # reference = torch.roll(reference, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            # TODO ATTN_MAP SHIFT
        else:
            x = shifted_x

        x = x.view(B, H * W, C)  # x (batch_in_each_GPU, H*W, embedding_channel)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel)


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size  # nW: number of windows
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinDATLayer(nn.Module):
    r""" Swin Deformable Attention Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        n_head (int): Number of attention heads.
        window_size (int): Local window size.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default: False.
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.n_group = n_group
        self.use_checkpoint = use_checkpoint

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        # build blocks
        self.blocks = nn.ModuleList([
            SwinDATransformerBlock(dim=dim,
                                   input_resolution=input_resolution,
                                   n_head=n_head,
                                   n_group=n_group,
                                   window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio,
                                   drop=drop,
                                   attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer,
                                   use_pe=use_pe,
                                   dwc_pe=dwc_pe,
                                   no_off=no_off,
                                   fixed_pe=fixed_pe,
                                   )
            for i in range(depth)])

        # patch merging/expend layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        # position = []
        # reference = []
        # attn_map = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, _, __, ___ = checkpoint.checkpoint(blk, x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)
            else:
                x, _, __, ___ = blk(x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)

        if self.downsample is not None:
            x = self.downsample(x)
        # return x, position, reference, attn_map
        return x, 0, 0, 0

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSDATB_ENCODER(nn.Module):
    r"""Residual Swin Deformable Transformer Block in Encoder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depth (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        downsample: Downsampling module. Default: PatchMerging
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4, use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super(RSDATB_ENCODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        self.residual_group = SwinDATLayer(dim=dim,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           n_head=n_head,
                                           window_size=window_size,
                                           n_group=n_group,
                                           mlp_ratio=mlp_ratio,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           downsample=downsample,
                                           use_checkpoint=use_checkpoint,
                                           use_pe=use_pe,
                                           dwc_pe=dwc_pe,
                                           no_off=no_off,
                                           fixed_pe=fixed_pe)

        self.conv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x, _, __, ___ = self.residual_group(x, x_size)
        x_size = (x_size[0] // 2, x_size[1] // 2)

        shortcut = self.downsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        # return x, position, reference, attn_map
        return x, 0, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut downsample
        flops += self.downsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H / 2) * (W / 2) * (self.dim * 2) * (self.dim * 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


class RSDATB_DECODER(nn.Module):
    r"""Residual Swin Deformable Transformer Block in Deconder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depth (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        upsample: Upsamping module. Default: PatchExpand
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4, use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super(RSDATB_DECODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        self.residual_group = SwinDATLayer(dim=dim,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           n_head=n_head,
                                           window_size=window_size,
                                           n_group=n_group,
                                           mlp_ratio=mlp_ratio,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           downsample=upsample,
                                           use_checkpoint=use_checkpoint,
                                           use_pe=use_pe,
                                           dwc_pe=dwc_pe,
                                           no_off=no_off,
                                           fixed_pe=fixed_pe)

        self.conv = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        # patch expending layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x, _, __, ___ = self.residual_group(x, x_size)
        x_size = (x_size[0] * 2, x_size[1] * 2)

        shortcut = self.upsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        # return x, position, reference, attn_map
        return x, 0, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut upsample
        flops += self.upsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H * 2) * (W * 2) * (self.dim / 2) * (self.dim / 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


# ------------------------------
# Swin Transformer Basic
# ------------------------------

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_reverse_deform_point(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size, coordinate)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W, O)
    """
    device = windows.device
    nWH = H // window_size  # number of windows in H direction
    nWW = W // window_size  # number of windows in W direction
    assert (H % window_size == 0) or (W % window_size == 0), "H and W should be the integer multiple of window size!"

    B = int(windows.shape[0] / (nWH * nWW))
    C = windows.shape[1]  # batch number# batch number
    x = windows.reshape(B, C, nWH, nWW, window_size, window_size, 2)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().reshape(B, C, H, W, 2)

    window_coodinate_H_1d = np.linspace(-1, 1, nWH, endpoint=False) + (1 / nWH)
    window_coodinate_W_1d = np.linspace(-1, 1, nWW, endpoint=False) + (1 / nWW)
    X, Y = np.meshgrid(window_coodinate_W_1d, window_coodinate_H_1d)
    window_coodinate_2d = np.stack((Y, X), axis=2)
    window_coodinate_2d = torch.from_numpy(window_coodinate_2d).to(device)

    x[:, :, :, :, 0] = (x[:, :, :, :, 0].clone() / (nWH))
    x[:, :, :, :, 1] = (x[:, :, :, :, 1].clone() / (nWW))

    xx = torch.zeros_like(x)
    for idx_nWH in range(nWH):
        for idx_nWW in range(nWW):
            xx[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
            idx_nWW * window_size:(idx_nWW + 1) * window_size, 1] \
                = - x[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
                    idx_nWW * window_size:(idx_nWW + 1) * window_size, 0] \
                  - window_coodinate_2d[idx_nWH, idx_nWW, 0]

            xx[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
            idx_nWW * window_size:(idx_nWW + 1) * window_size, 0] \
                = x[:, :, idx_nWH * window_size:(idx_nWH + 1) * window_size,
                  idx_nWW * window_size:(idx_nWW + 1) * window_size, 1] \
                  + window_coodinate_2d[idx_nWH, idx_nWW, 1]

    return xx


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        n_head (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, n_head, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # input channel
        self.window_size = window_size  # window height Wh, Window width Ww
        self.n_head = n_head  # number of heads
        head_dim = dim // n_head  # head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_head))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # B: number of Windows * Batch_size in a GPU  N: patch number in a window  C:  embedding channel
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # q,k,v: (number of Windows * Batch_size in a GPU, number of head, number of patch in a wondow, channe/number of head)
        # q,k,v (576,6,64,30) (number of Windows * Batch_size in a GPU, number of head, patch number in a window, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_head, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_head, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, n_head={self.n_head}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += N * self.dim * N
        # x = (attn @ v)
        flops += N * N * self.dim
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
    """

    def __init__(self, dim, input_resolution, n_head, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=to_2tuple(self.window_size), n_head=n_head,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C)  # x (batch_in_each_GPU, embedding_channel, H, W)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # shifted_x (batch_in_each_GPU, embedding_channel, H, W)

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(
                x.device))  # (nW*B, window_size*window_size, C)  nW:number of Windows

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C shifted_x  (batch_in_each_GPU, embedding_channel, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)  # x (batch_in_each_GPU, H*W, embedding_channel)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x  # x (batch_in_each_GPU, H*W, embedding_channel)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size  # nW: number of windows
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        n_head (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 n_head=n_head,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging/expend layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB_ENCODER(nn.Module):
    '''
    Residual Swin Transformer Block in Encoder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depths (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        downsample: Downsampling module. Default: PatchMerging
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
    '''

    def __init__(self, dim, input_resolution, depth, n_head, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4):
        super(RSTB_ENCODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = SwinTLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         n_head=n_head,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop,
                                         attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        self.conv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x = self.residual_group(x, x_size)
        x_size = (x_size[0] // 2, x_size[1] // 2)

        shortcut = self.downsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        return x, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut downsample
        flops += self.downsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H / 2) * (W / 2) * (self.dim * 2) * (self.dim * 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


class RSTB_DECODER(nn.Module):
    '''
    Residual Swin Transformer Block in Deconder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depths (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        upsample: Upsamping module. Default: PatchExpand
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
    '''

    def __init__(self, dim, input_resolution, depth, n_head, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4):
        super(RSTB_DECODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = SwinTLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         n_head=n_head,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop,
                                         attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=upsample,
                                         use_checkpoint=use_checkpoint)

        self.conv = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        # patch expending layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x = self.residual_group(x, x_size)
        x_size = (x_size[0] * 2, x_size[1] * 2)

        shortcut = self.upsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        return x, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut upsample
        flops += self.upsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H * 2) * (W * 2) * (self.dim / 2) * (self.dim / 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


# ------------------------------
# DATransformer Basic
# ------------------------------

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


def reverse_deform_point(points):
    """
    Args:
        points: (B, C, Hk, Wk, 2)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W, O)
    """
    x = torch.zeros_like(points)
    x[:, :, :, :, 1] = - points[:, :, :, :, 0]
    x[:, :, :, :, 0] = points[:, :, :, :, 1]

    return x



class DAttention(nn.Module):
    r""" Deformable Attention

    Args:
        q_size(tuple[int]): Size if query
        kv_size(tuple[int]): Size if key and value
        dim (int): Number of input channels.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        stride (int): Stride in offset calculation network
        offset_range_factor (int): Offset range factor in offset calculation network
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, q_size, kv_size, dim, n_head, n_group,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 use_pe, dwc_pe, no_off, fixed_pe):
        super().__init__()

        self.dim = dim  # input channel
        self.n_head = n_head  # number of head
        self.n_head_channels = self.dim // self.n_head  # head_dim
        self.scale = self.n_head_channels ** -0.5

        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size

        self.n_group = n_group
        self.n_group_channels = self.dim // self.n_group
        self.n_group_heads = self.n_head // self.n_group

        self.dwc_pe = dwc_pe
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        if self.q_h <= 12 or self.q_w <= 12:
            self.kk = 3
        elif 13 <= self.q_h <= 24 or 13 <= self.q_w <= 24:
            self.kk = 5
        elif 25 <= self.q_h <= 48 or 25 <= self.q_w <= 48:
            self.kk = 7
        else:
            self.kk = 9

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, stride, self.kk // 2,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_head, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_group, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, x_size):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        assert x_size[0] == H and x_size[1] == W, "input feature has wrong size"

        # calculate query
        q = self.proj_q(x)  # B C H W
        # resize query
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_group, c=self.n_group_channels)

        # use query to calculate offset
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        # get the size of offset
        Hk, Wk = offset.size(2), offset.size(3)
        # sample number
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # resize offset
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        # use the number of offset point and batch size to get reference point
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # no offset
        if self.no_off:
            offset = torch.zeros_like(offset)

        # offset + ref
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # embedding query,key,valuse B,C,H,W --> B*head,head_channel,HW
        q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)

        # Q&K
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # use position encoding
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_head, self.n_head_channels,
                                                                              H * W)
            # fix
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_head, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(
                            B * self.n_group, H * W, 2).unsqueeze(2)
                        - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, 0, 0, 0

    def extra_repr(self) -> str:
        return f'dim={self.dim}, n_head={self.n_head}, n_group={self.n_group}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj_q(x)
        flops += N * self.dim * self.dim * 1
        if not self.no_off:
            # self.conv_offset(q_off)
            flops += self.n_group * N * self.n_group_channels * (
                    self.n_group_channels / self.n_group_channels) * self.kk * self.kk
            flops += self.n_group * N * self.n_group_channels
            flops += self.n_group * N * self.n_group_channels * 2 * 1
        # self.proj_k(x_sampled)
        flops += N * self.dim * self.dim * 1
        # self.proj_v(x_sampled)
        flops += N * self.dim * self.dim * 1
        # torch.einsum('b c m, b c n -> b m n', q, k)
        flops += self.n_group * N * self.n_group_channels * N
        # torch.einsum('b m n, b c n -> b c m', attn, v)
        flops += self.n_group * N * N * self.n_group_channels
        # self.proj_drop(self.proj_out(out))
        flops += N * self.dim * self.dim * 1
        return flops


class DATransformerBlock(nn.Module):
    r""" Deformable Attention Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        n_head (int): Number of attention heads.
        n_group (int): Offset group.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LaerNorm
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, n_head, n_group, mlp_ratio=2., drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.n_head = n_head
        self.n_head_channels = dim // n_head
        self.n_group = n_group
        self.mlp_ratio = mlp_ratio

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        self.norm1 = norm_layer(dim)
        self.attn = DAttention(
            q_size=input_resolution, kv_size=input_resolution, dim=dim, n_head=n_head, n_group=n_group,
            attn_drop=attn_drop, proj_drop=drop, stride=1, offset_range_factor=2,
            use_pe=use_pe, dwc_pe=dwc_pe, no_off=no_off, fixed_pe=fixed_pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size  # x (batch_in_each_GPU, H*W, embedding_channel)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x (batch_in_each_GPU, H*W, embedding_channel)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B,L,C --> B,H,W,C --> B,C,H,W

        x, position, reference, attn_map = self.attn(x, x_size)  # x (B,C,H,W) _&__ (batch_size, group, 32, 32, 2)

        # position = reverse_deform_point(position)
        # reference = reverse_deform_point(reference)

        x = x.permute(0, 2, 3, 1)  # B,C,H,W --> B,H,W,C
        x = x.view(B, H * W, C)  # B,H,W,C --> B,HW,C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, 0, 0, 0  # x (batch_in_each_GPU, H*W, embedding_channel)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, n_head={self.n_head}, " \
               f"n_group={self.n_group}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # D-MSA
        flops += self.attn.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class DATLayer(nn.Module):
    """ A basic Deformable Attention layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        n_head (int): Number of attention heads.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default: False.
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.n_group = n_group
        self.use_checkpoint = use_checkpoint

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        # build blocks
        self.blocks = nn.ModuleList([
            DATransformerBlock(dim=dim,
                               input_resolution=input_resolution,
                               n_head=n_head,
                               n_group=n_group,
                               mlp_ratio=mlp_ratio,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer,
                               use_pe=use_pe,
                               dwc_pe=dwc_pe,
                               no_off=no_off,
                               fixed_pe=fixed_pe
                               )
            for i in range(depth)])

        # patch merging/expend layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        # position = []
        # reference = []
        # attn_map = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, _, __, ___ = checkpoint.checkpoint(blk, x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)
            else:
                x, _, __, ___ = blk(x, x_size)
                # position.append(pos)
                # reference.append(ref)
                # attn_map.append(att)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, 0, 0, 0

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RDATB_ENCODER(nn.Module):
    '''
    Residual DATransformer Block in Encoder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depth (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        downsample: Downsampling module. Default: PatchMerging
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    '''

    def __init__(self, dim, input_resolution, depth, n_head, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=256, patch_size=8, use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super(RDATB_ENCODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        self.residual_group = DATLayer(dim=dim,
                                       input_resolution=input_resolution,
                                       depth=depth,
                                       n_head=n_head,
                                       n_group=n_group,
                                       mlp_ratio=mlp_ratio,
                                       drop=drop,
                                       attn_drop=attn_drop,
                                       drop_path=drop_path,
                                       norm_layer=norm_layer,
                                       downsample=downsample,
                                       use_checkpoint=use_checkpoint,
                                       use_pe=use_pe,
                                       dwc_pe=dwc_pe,
                                       no_off=no_off,
                                       fixed_pe=fixed_pe)

        self.conv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim * 2,
            norm_layer=None)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x, _, __, ___ = self.residual_group(x, x_size)
        x_size = (x_size[0] // 2, x_size[1] // 2)

        shortcut = self.downsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        return x, 0, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut downsample
        flops += self.downsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H / 2) * (W / 2) * (self.dim * 2) * (self.dim * 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


class RDATB_DECODER(nn.Module):
    '''
    Residual DATransformer Block in Encoder

    Args:
        dim (int): Input patch embedding dimension of this block.
        input_resolution(tuple(int)): Input size of this block.
        depths (int): Depth of this Transformer block (The number of transformer layers).
        n_head (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        downsample: Downsampling module. Default: PatchMerging
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    '''

    def __init__(self, dim, input_resolution, depth, n_head, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 img_size=256, patch_size=8, use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):
        super(RDATB_DECODER, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        self.residual_group = DATLayer(dim=dim,
                                       input_resolution=input_resolution,
                                       depth=depth,
                                       n_head=n_head,
                                       n_group=n_group,
                                       mlp_ratio=mlp_ratio,
                                       drop=drop,
                                       attn_drop=attn_drop,
                                       drop_path=drop_path,
                                       norm_layer=norm_layer,
                                       downsample=upsample,
                                       use_checkpoint=use_checkpoint,
                                       use_pe=use_pe,
                                       dwc_pe=dwc_pe,
                                       no_off=no_off,
                                       fixed_pe=fixed_pe)

        self.conv = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=-1, patch_size=-1, in_chans=-1, embed_dim=dim // 2,
            norm_layer=None)

        # patch expanding layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        assert x.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        shortcut = x
        x, _, __, ___ = self.residual_group(x, x_size)
        x_size = (x_size[0] * 2, x_size[1] * 2)

        shortcut = self.upsample(shortcut)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + shortcut

        return x, 0, 0, 0

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # residual_group
        flops += self.residual_group.flops()
        # shortcut upsample
        flops += self.upsample.flops()
        # patch_unembed
        flops += self.patch_unembed.flops()
        # conv
        flops += (H * 2) * (W * 2) * (self.dim / 2) * (self.dim / 2) * 9
        # patch_embed
        flops += self.patch_embed.flops()
        return flops


# ------------------------------
# Model Architecture
# ------------------------------

class SDAUT(nn.Module):
    """ SDAUT.
    A PyTorch Implement of SDAUT.

    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels (1 for Gray, 3 for RGB). Default: 1
        embed_dims (list): Input patch embedding dimension of every RSTBs and RDATBs. Default: [90, 180, 360, 720, 720, 360]
        types (list): Transformer block types ('s' for RSTB and 'd' for RDATB). Default: ["s", "s", "d", "d", "s", "s"]
        depths (list): Depth of each Transformer block.
        n_heads (list): Number of attention heads in different layers.
        n_groups (list): Number of groups in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_range: Image range. 1. or 255.
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=1,
                 embed_dims=[90, 180, 360, 720, 720, 360], types=["s", "s", "d", "d", "s", "s"],
                 depths=[6, 6, 6, 6, 6, 6], n_heads=[6, 6, 6, 6, 6, 6], n_groups=[-1, -1, 6, 6, -1, -1],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1.,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False,
                 **kwargs):
        super(SDAUT, self).__init__()
        num_in_ch = in_chans  # 1 for Gray
        num_out_ch = in_chans  # 1 for Gray
        self.img_range = img_range  # 1.0 pixel 0~1
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, list):
            img_size = tuple(img_size)
        else:
            raise ('Wrong Image Size Type!')

        self.window_size = window_size
        self.patch_size = patch_size
        #####################################################################################################
        ########################################## INPUT MODULE (IM) ########################################

        self.conv_first = nn.Conv2d(num_in_ch, embed_dims[0],
                                    kernel_size=to_2tuple(patch_size), stride=to_2tuple(patch_size))

        #####################################################################################################
        ################################### FEATURE EXTRACTION MODULE (FEM) #################################
        self.num_layers = len(types)  # [6,6,6,6,6,6]
        assert len(types) == len(depths) == len(embed_dims) == len(n_heads), "Number of blocks mismatches!"
        self.embed_dims = embed_dims

        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dims[0]
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dims[0], embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)  # e.g. img_size:256, patch_size:2, embed_dims[0]:90
        num_patches = self.patch_embed.num_patches  # number of patch. e.g. num_patches: 16384=128*128=(256/2)*(256/2)
        patches_resolution = self.patch_embed.patches_resolution  # resolution of patch. e.g. patches_resolution: (128, 128) = (256/2, 256/2)
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dims[0], embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual S/DA Transformer blocks (RSTB/RDATB)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.conv_fix_chl = nn.ModuleList()
        self.unemb_conv_fix_chl = nn.ModuleList()
        self.emb_conv_fix_chl = nn.ModuleList()
        # Encoder
        for i_layer in range(self.num_layers // 2):

            if types[i_layer] == 's':
                layer = RSTB_ENCODER(dim=embed_dims[i_layer],
                                     input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                       patches_resolution[1] // (2 ** i_layer)),
                                     depth=depths[i_layer],
                                     n_head=n_heads[i_layer],
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=0.,
                                     norm_layer=norm_layer,
                                     downsample=PatchMerging,
                                     use_checkpoint=use_checkpoint,
                                     img_size=img_size,
                                     patch_size=patch_size, )
                self.encoder.append(layer)

            elif types[i_layer] == 'd':
                layer = RDATB_ENCODER(dim=embed_dims[i_layer],
                                      input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      depth=depths[i_layer],
                                      n_head=n_heads[i_layer],
                                      n_group=n_groups[i_layer],
                                      mlp_ratio=mlp_ratio,
                                      drop=drop_rate,
                                      attn_drop=attn_drop_rate,
                                      drop_path=0.,
                                      norm_layer=norm_layer,
                                      downsample=PatchMerging,
                                      use_checkpoint=use_checkpoint,
                                      img_size=img_size,
                                      patch_size=patch_size,
                                      use_pe=use_pe,
                                      dwc_pe=dwc_pe,
                                      no_off=no_off,
                                      fixed_pe=fixed_pe,
                                      )
                self.encoder.append(layer)

            elif types[i_layer] == 'k':
                layer = RSDATB_ENCODER(dim=embed_dims[i_layer],
                                       input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                         patches_resolution[1] // (2 ** i_layer)),
                                       depth=depths[i_layer],
                                       n_head=n_heads[i_layer],
                                       window_size=window_size,
                                       n_group=n_groups[i_layer],
                                       mlp_ratio=mlp_ratio,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       drop_path=0.,
                                       norm_layer=norm_layer,
                                       downsample=PatchMerging,
                                       use_checkpoint=use_checkpoint,
                                       img_size=img_size,
                                       patch_size=patch_size,
                                       use_pe=use_pe,
                                       dwc_pe=dwc_pe,
                                       no_off=no_off,
                                       fixed_pe=fixed_pe,
                                       )
                self.encoder.append(layer)
            else:
                raise "Wrong Block types for Encoder!"

        # Decoder
        for i_layer in range(self.num_layers // 2):
            if types[i_layer + self.num_layers // 2] == 's':
                layer = RSTB_DECODER(dim=embed_dims[i_layer + self.num_layers // 2],
                                     input_resolution=(patches_resolution[0] // (2 ** (self.num_layers // 2 - i_layer)),
                                                       patches_resolution[1] // (
                                                                   2 ** (self.num_layers // 2 - i_layer))),
                                     depth=depths[i_layer + self.num_layers // 2],
                                     n_head=n_heads[i_layer + self.num_layers // 2],
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=0.,
                                     norm_layer=norm_layer,
                                     upsample=PatchExpand,
                                     use_checkpoint=use_checkpoint,
                                     img_size=img_size,
                                     patch_size=patch_size, )
                self.decoder.append(layer)

                conv_fix_chl = nn.Conv2d(embed_dims[i_layer + self.num_layers // 2] * 2,
                                         embed_dims[i_layer + self.num_layers // 2], 3, 1, 1)
                self.conv_fix_chl.append(conv_fix_chl)

                unemb_conv_fix_chl = PatchUnEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                                  embed_dim=embed_dims[i_layer + self.num_layers // 2] * 2,
                                                  norm_layer=None)
                self.unemb_conv_fix_chl.append(unemb_conv_fix_chl)

                emb_conv_fix_chl = PatchEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                              embed_dim=embed_dims[i_layer + self.num_layers // 2], norm_layer=None)
                self.emb_conv_fix_chl.append(emb_conv_fix_chl)

            elif types[i_layer + self.num_layers // 2] == 'd':
                layer = RDATB_DECODER(dim=embed_dims[i_layer + self.num_layers // 2],
                                      input_resolution=(
                                      patches_resolution[0] // (2 ** (self.num_layers // 2 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers // 2 - i_layer))),
                                      depth=depths[i_layer + self.num_layers // 2],
                                      n_head=n_heads[i_layer + self.num_layers // 2],
                                      n_group=n_groups[i_layer + self.num_layers // 2],
                                      mlp_ratio=mlp_ratio,
                                      drop=drop_rate,
                                      attn_drop=attn_drop_rate,
                                      drop_path=0.,
                                      norm_layer=norm_layer,
                                      upsample=PatchExpand,
                                      use_checkpoint=use_checkpoint,
                                      img_size=img_size,
                                      patch_size=patch_size,
                                      use_pe=use_pe,
                                      dwc_pe=dwc_pe,
                                      no_off=no_off,
                                      fixed_pe=fixed_pe,
                                      )
                self.decoder.append(layer)

                conv_fix_chl = nn.Conv2d(embed_dims[i_layer + self.num_layers // 2] * 2,
                                         embed_dims[i_layer + self.num_layers // 2], 3, 1, 1)
                self.conv_fix_chl.append(conv_fix_chl)

                unemb_conv_fix_chl = PatchUnEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                                  embed_dim=embed_dims[i_layer + self.num_layers // 2] * 2,
                                                  norm_layer=None)
                self.unemb_conv_fix_chl.append(unemb_conv_fix_chl)

                emb_conv_fix_chl = PatchEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                              embed_dim=embed_dims[i_layer + self.num_layers // 2], norm_layer=None)
                self.emb_conv_fix_chl.append(emb_conv_fix_chl)

            elif types[i_layer + self.num_layers // 2] == 'k':
                layer = RSDATB_DECODER(dim=embed_dims[i_layer + self.num_layers // 2],
                                       input_resolution=(
                                           patches_resolution[0] // (2 ** (self.num_layers // 2 - i_layer)),
                                           patches_resolution[1] // (2 ** (self.num_layers // 2 - i_layer))),
                                       depth=depths[i_layer + self.num_layers // 2],
                                       n_head=n_heads[i_layer + self.num_layers // 2],
                                       window_size=window_size,
                                       n_group=n_groups[i_layer + self.num_layers // 2],
                                       mlp_ratio=mlp_ratio,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       drop_path=0.,
                                       norm_layer=norm_layer,
                                       upsample=PatchExpand,
                                       use_checkpoint=use_checkpoint,
                                       img_size=img_size,
                                       patch_size=patch_size,
                                       use_pe=use_pe,
                                       dwc_pe=dwc_pe,
                                       no_off=no_off,
                                       fixed_pe=fixed_pe,
                                       )
                self.decoder.append(layer)

                conv_fix_chl = nn.Conv2d(embed_dims[i_layer + self.num_layers // 2] * 2,
                                         embed_dims[i_layer + self.num_layers // 2], 3, 1, 1)
                self.conv_fix_chl.append(conv_fix_chl)

                unemb_conv_fix_chl = PatchUnEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                                  embed_dim=embed_dims[i_layer + self.num_layers // 2] * 2,
                                                  norm_layer=None)
                self.unemb_conv_fix_chl.append(unemb_conv_fix_chl)

                emb_conv_fix_chl = PatchEmbed(img_size=-1, patch_size=-1, in_chans=-1,
                                              embed_dim=embed_dims[i_layer + self.num_layers // 2], norm_layer=None)
                self.emb_conv_fix_chl.append(emb_conv_fix_chl)

            else:
                raise "Wrong Block types for Decoder!"

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1)

        #####################################################################################################
        ######################################## OUTPUT MODULE (OM) #########################################
        # Pixel Shuffle Light
        self.conv_last = UpsampleOneStep(patch_size, embed_dims[0], num_out_ch, patches_resolution)

        self.apply(self._init_weights)

        self.position = 0
        self.reference = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])  # x (batch_size_in_each_GPU, embedding_channel, H/patch_size, W/patch_size)
        x = self.patch_embed(x)  # x (batch_size_in_each_GPU, H*W/patch_size/patch_size, embedding_channel)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_shortcut = []
        # position = []
        # reference = []
        # attn_map = []

        for idx, layer in enumerate(self.encoder):
            assert x.shape[1] == x_size[0] * x_size[1], "Wrong H and W in the encoder!"  # Check the H and W

            output = layer(x, x_size)
            if len(output) == 1:
                x = output
                pos = 0
                ref = 0
                att = 0
            elif len(output) == 3:
                (x, pos, ref) = output
                att = 0
            elif len(output) == 4:
                (x, pos, ref, att) = output
            else:
                raise "WRONG OUTPUT"

            x_size = (x_size[0] // 2, x_size[1] // 2)  # Downsampling in Encoder, size=size/2
            if idx < len(self.encoder) - 1:  # exclude the last layer
                x_shortcut.append(x)
            # position.append(pos)
            # reference.append(ref)
            # attn_map.append(att)

        for idx, layer in enumerate(self.decoder):
            assert x.shape[1] == x_size[0] * x_size[1], "Wrong H and W in the decoder!"  # Check the H and W
            if idx == 0:
                output = layer(x, x_size)
                if len(output) == 1:
                    x = output
                    pos = 0
                    ref = 0
                    att = 0
                elif len(output) == 3:
                    (x, pos, ref) = output
                    att = 0
                elif len(output) == 4:
                    (x, pos, ref, att) = output
                else:
                    raise "WRONG OUTPUT"
            else:
                x = self.unemb_conv_fix_chl[idx](torch.cat([x, x_shortcut[len(self.decoder) - idx - 1]], dim=-1), x_size)
                x = self.conv_fix_chl[idx](x)
                x = self.emb_conv_fix_chl[idx](x)
                output = layer(x, x_size)
                if len(output) == 1:
                    x = output
                    pos = 0
                    ref = 0
                    att = 0
                elif len(output) == 3:
                    (x, pos, ref) = output
                    att = 0
                elif len(output) == 4:
                    (x, pos, ref, att) = output
                else:
                    raise "WRONG OUTPUT"
                # x, pos, ref, att = layer(x, x_size)
            x_size = (x_size[0] * 2, x_size[1] * 2)  # Upsampling in Decoder, size=size*2
            # position.append(pos)
            # reference.append(ref)
            # attn_map.append(att)

        x = self.norm(x)  # x (batch_size_in_each_GPU, H*W, embedding_channel180)
        x = self.patch_unembed(x, x_size)  # x (batch_size_in_each_GPU, embedding_channel, H/patch_size, W/patch_size)

        return x, 0, 0, 0

    def forward(self, x):
        H, W = x.shape[2:]  # x (batch_size_in_each_GPU, input_image_channel,  H/patch_size, W/patch_size)
        x = self.check_image_size(x)  # Check for Window Attention

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(
            x)  # x_first (batch_size_in_each_GPU, input_image_channel,  H/patch_size, W/patch_size)
        xx, _, __, ___ = self.forward_features(x_first)
        # self.position = position
        # self.reference = reference
        # self.attn_map = attn_map
        res = self.conv_after_body(xx) + x_first  # res (batch_size_in_each_GPU, input_image_channel,  H/patch_size, W/patch_size)
        x = x + self.conv_last(res)  # x (batch_size_in_each_GPU, input_image_channel,  H/patch_size, W/patch_size)

        x = x / self.img_range + self.mean

        return x

    def get_position(self):
        return self.position

    def get_reference(self):
        return self.reference

    def get_attn_map(self):
        return self.attn_map

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        # self.conv_first(x)
        flops += H * W * 1 * self.num_features * self.patch_size * self.patch_size

        # self.forward_features(x_first)
        flops += self.patch_embed.flops()
        for idx, layer in enumerate(self.encoder):
            flops += layer.flops()
        for idx, layer in enumerate(self.decoder):
            flops += layer.flops()
        flops += H * W * self.num_features
        flops += self.patch_unembed.flops()

        # self.conv_after_body(xx) + x_first
        flops += H * W * self.num_features * self.num_features * 9

        # x = x + self.conv_last(res)
        flops += self.conv_last.flops()

        return flops


# ------------------------------
# Common Function
# ------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # norm: norm_layer(4 * dim)
        flops = (H // 2) * (W // 2) * 4 * self.dim
        # reduction: nn.Linear(4 * dim, 2 * dim, bias=False)
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    r""" Patch Expand Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = einops.rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # expand: nn.Linear(dim, 2 * dim, bias=False)
        flops = H * W * self.dim * 2 * self.dim
        # norm: norm_layer(dim // dim_scale)
        flops += (2 * H) * (2 * W) * (self.dim // 2)
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution):
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.input_resolution = input_resolution
        self.scale = scale
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        m.append(nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += H * W * self.num_feat * (self.scale ** 2) * self.num_out_ch * 9
        flops += self.scale * H * self.scale * W * self.num_out_ch * self.num_out_ch * 9
        return flops




if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    batch = 1
    height = 256
    width = 256  # 192
    patch_size = 1
    device = 'cuda'

    print('kkddkk')
    model = SDAUT(img_size=[height, width],
                    patch_size = patch_size,
                    in_chans=1,
                    embed_dim=90,
                    embed_dims=[90,180,360,720,360,180],
                    types=["k", "k", "d", "d", "k", "k"],
                    depths=[6, 6, 6, 6, 6, 6],
                    n_heads=[6, 12, 24, 24, 24, 12],
                    n_groups=[1, 3, 6, 6, 6, 3],
                    window_size=8,
                    mlp_ratio=2.,
                    img_range=1.,
                    use_pe=True,
                    dwc_pe=False,
                    no_off=False,
                    fixed_pe=False,
                    ).to(device)

    # print(model)
    print('FLOPs: {}G'.format(round((model.flops() * 1e-9),3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')


    print('kkddkk-nooffset')
    model = SDAUT(img_size=[height, width],
                    patch_size = patch_size,
                    in_chans=1,
                    embed_dim=90,
                    embed_dims=[90,180,360,720,360,180],
                    types=["k", "k", "d", "d", "k", "k"],
                    depths=[6, 6, 6, 6, 6, 6],
                    n_heads=[6, 12, 24, 24, 24, 12],
                    n_groups=[1, 3, 6, 6, 6, 3],
                    window_size=8,
                    mlp_ratio=2.,
                    img_range=1.,
                    use_pe=True,
                    dwc_pe=False,
                    no_off=True,
                    fixed_pe=False,
                    ).to(device)

    # print(model)
    print('FLOPs: {}G'.format(round((model.flops() * 1e-9),3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')

    print('kkkkkk')
    model = SDAUT(img_size=[height, width],
                    patch_size = patch_size,
                    in_chans=1,
                    embed_dim=90,
                    embed_dims=[90, 180, 360, 720, 360, 180],
                    types=["k", "k", "k", "k", "k", "k"],
                    depths=[6, 6, 6, 6, 6, 6],
                    n_heads=[6, 12, 24, 24, 24, 12],
                    n_groups=[1, 3, 6, 6, 3, 1],
                    window_size=8,
                    mlp_ratio=2.,
                    img_range=1.,
                    use_pe=True,
                    dwc_pe=False,
                    no_off=False,
                    fixed_pe=False,
                    ).to(device)

    # print(model)
    print('FLOPs: {}G'.format(round((model.flops() * 1e-9), 3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')
    #
    #
    print('kkkkkk-nooffset')
    model = SDAUT(img_size=[height, width],
                    patch_size = patch_size,
                    in_chans=1,
                    embed_dim=90,
                    embed_dims=[90, 180, 360, 720, 360, 180],
                    types=["k", "k", "k", "k", "k", "k"],
                    depths=[6, 6, 6, 6, 6, 6],
                    n_heads=[6, 12, 24, 24, 24, 12],
                    n_groups=[1, 3, 6, 6, 3, 1],
                    window_size=8,
                    mlp_ratio=2.,
                    img_range=1.,
                    use_pe=True,
                    dwc_pe=False,
                    no_off=True,
                    fixed_pe=False,
                    ).to(device)

    # print(model)
    print('FLOPs: {}G'.format(round((model.flops() * 1e-9), 3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')

    # print('ssddss')
    # model = SDAUT(img_size=[height, width],
    #                 patch_size = patch_size,
    #                 in_chans=1,
    #                 embed_dim=90,
    #                 embed_dims=[90,180,360,720,360,180],
    #                 types=["s", "s", "d", "d", "s", "s"],
    #                 depths=[6, 6, 6, 6, 6, 6],
    #                 n_heads=[6, 6, 6, 6, 6, 6],
    #                 n_groups=[-1, -1, 6, 6, -1, -1],
    #                 window_size=8,
    #                 mlp_ratio=2.,
    #                 img_range=1.,
    #                 use_pe=True,
    #                 dwc_pe=False,
    #                 no_off=False,
    #                 fixed_pe=False,
    #                 ).to(device)
    #
    # # print(model)
    # print('FLOPs: {}G'.format(round((model.flops() * 1e-9),3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')
    #
    #
    #
    # print('ssssss')
    # model = SDAUT(img_size=[height, width],
    #                 patch_size = patch_size,
    #                 in_chans=1,
    #                 embed_dim=90,
    #                 embed_dims=[90,180,360,720,360,180],
    #                 types=["s", "s", "s", "s", "s", "s"],
    #                 depths=[6, 6, 6, 6, 6, 6],
    #                 n_heads=[6, 6, 6, 6, 6, 6],
    #                 n_groups=[-1, -1, 6, 6, -1, -1],
    #                 window_size=8,
    #                 mlp_ratio=2.,
    #                 img_range=1.,
    #                 use_pe=True,
    #                 dwc_pe=False,
    #                 no_off=False,
    #                 fixed_pe=False,
    #                 ).to(device)
    #
    # # print(model)
    # print('FLOPs: {}G'.format(round((model.flops() * 1e-9),3)))
    # x = torch.randn((batch, 1, height, width)).to(device)
    # print(f'Input shape: {x.shape}')
    # x = model(x)
    # print(f'Output shape: {x.shape}')
    # print('-------------------------------')

    # macs, params = profile(model, inputs=(x, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs)
    # print(params)
