import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
from einops import rearrange

from Unetplusplus import *
from Res import ResBlock, ResBlock_noPadding, stadge_block, DeFusion

GPUno = 0
useAllAvailableGPU = False
device = torch.device("cuda:" + str(GPUno) if torch.cuda.is_available() else "cpu")


##### XAVIER WEIGHT INITIALIZATION FOR NETWORK PARAMETER INITIALIZATION #####
def init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


nonlinearity = partial(F.relu, inplace=True)

####################################################################################################
# Component from UTNet about the transformer part

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

#######################################################################################################################
#Multi-scale fusion function
class Mlp_fusion(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FusionAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.relative_position_encoding = RelativePositionBias(heads, 8, 8)

        #self.base_proj = nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, bias=False)

        #self.semantic_proj = nn.Conv1d(inner_dim, 64, kernel_size=3, padding=1, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: B, L, C.   Batch, sequence length, dim
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        '''
        k1 = k[:, 0:x1, :]
        k2 = k[:, x1:x1 + x2, :]
        k3 = k[:, x1 + x2:x1 + x2 + x3, :]
        k4 = k[:, x1 + x2 + x3:x1 + x2 + x3 + x4, :]

        k1, k2, k3, k4 = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b (heads dim_head) l ', heads=16),
                             (k1, k2, k3, k4))

        k1, k2, k3, k4 = map(lambda t: F.interpolate(t, size=8 * 8, mode='linear', align_corners=True),
                             (k1, k2, k3, k4))
        k1, k2, k3, k4 = map(lambda t: rearrange(t, 'b (heads dim_head) l -> b heads l dim_head ', heads=16),
                             (k1, k2, k3, k4))

        k_f = torch.cat([k1, k2, k3, k4], 2)

        v1 = v[:, 0:x1, :]
        v2 = v[:, x1:x1 + x2, :]
        v3 = v[:, x1 + x2:x1 + x2 + x3, :]
        v4 = v[:, x1 + x2 + x3:x1 + x2 + x3 + x4, :]

        v1, v2, v3, v4 = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b (heads dim_head) l ', heads=16),
                             (v1, v2, v3, v4))

        v1, v2, v3, v4 = map(lambda t: F.interpolate(t, size=8 * 8, mode='linear', align_corners=True),
                             (v1, v2, v3, v4))
        v1, v2, v3, v4 = map(lambda t: rearrange(t, 'b (heads dim_head) l -> b heads l dim_head ', heads=16),
                             (v1, v2, v3, v4))

        v_f = torch.cat([v1, v2, v3, v4], 2)

        k = k_f
        v = v_f
        '''
        '''
        weight_k = k.permute(0, 2, 1)  # B, C, L
        feat_k = self.base_proj(weight_k)  # B, C, L
        weight_map_k = self.semantic_proj(weight_k)  # B, 64, L
        semantic_k = torch.einsum('bij,bkj->bik', feat_k, weight_map_k)  # B, C, 64
        k = semantic_k

        weight_v = v.permute(0, 2, 1)  # B, C, L
        feat_v = self.base_proj(weight_v)  # B, C, L
        weight_map_v = self.semantic_proj(weight_v)  # B, 64, L
        semantic_v = torch.einsum('bij,bkj->bik', feat_v, weight_map_v)  # B, C, 64
        v = semantic_v
        '''

        k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b (heads dim_head) l ', heads=self.heads), (k, v))

        k, v = map(lambda t: F.interpolate(t, size= 4 * 8 * 8, mode='linear', align_corners=True), (k, v))
        k, v = map(lambda t: rearrange(t, 'b (heads dim_head) l -> b heads l dim_head ', heads=self.heads), (k, v))

        q = rearrange(q, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads)
        # q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])

        attn = torch.einsum('bhid,bhjd->bhij', q, k)

        '''
        # add position
        re0 = self.relative_position_encoding(80, 80)
        re1 = self.relative_position_encoding(40, 40)
        re2 = self.relative_position_encoding(20, 20)
        re3 = self.relative_position_encoding(10, 10)
        #re4 = self.relative_position_encoding(20, 20)
        relative_position_bias = torch.cat([re0, re1, re2, re3], dim=2)
        attn = attn + relative_position_bias
        '''

        attn = attn * self.scale

        attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')

        attned = self.to_out(attned)

        attned = self.proj_drop(attned)

        return attned


class FusionTransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.attn = FusionAttention(dim, heads, dim_head, attn_drop, proj_drop)
        self.norm = nn.LayerNorm(dim)

        self.mlp = Mlp_fusion(dim, mlp_dim, dim, drop=proj_drop)

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FusionAttention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp_fusion(dim, mlp_dim, dim, drop=proj_drop))
            ]))


    def forward(self, x):
        '''

        x = self.attn(self.norm(x)) + x
        x = self.mlp(self.norm(x)) + x
        '''

        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x


        return x


class FeatureMapFusion(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.depth = depth
        if depth == 0:
            pass
        else:
            self.dim = dim

            # project all maps to the same channel num
            self.in_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.in_proj.append(nn.Conv2d(in_dim_list[i], dim, kernel_size=1, bias=False))

            self.fusion = FusionTransformerBlock(dim, depth, heads, dim // heads, dim, attn_drop=0., proj_drop=0.)

            self.conv = nn.Conv1d(dim, dim, 1)

            # project all maps back to their origin channel num
            self.out_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.out_proj.append(nn.Conv2d(dim, in_dim_list[i], kernel_size=1, bias=False))

    def forward(self, map_list):

        if self.depth == 0:
            return map_list
        else:
            B, _, H, W = map_list[0].shape
            B1, _, H1, W1 = map_list[1].shape
            B2, _, H2, W2 = map_list[2].shape
            B3, _, H3, W3 = map_list[3].shape
            #B4, _, H4, W4 = map_list[4].shape
            proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in range(len(map_list))]
            # B, L, C where L=HW

            x1 = H * W
            x2 = H1 * W1
            x3 = H2 * W2
            #x4 = H3 * W3

            proj_maps = torch.cat(proj_maps, dim=1)

            attned_maps = self.fusion(proj_maps)
            #attned_maps = self.conv(attned_maps.permute(0, 2, 1))
            #attned_maps = attned_maps.permute(0, 2, 1)

            # attned_maps = attned_maps.chunk(len(map_list), dim=1)
            out_maps0 = attned_maps[:, 0:H * W, :]
            out_maps1 = attned_maps[:, H * W:H * W + H1 * W1, :]
            out_maps2 = attned_maps[:, H * W + H1 * W1:H * W + H1 * W1 + H2 * W2, :]
            out_maps3 = attned_maps[:, H * W + H1 * W1 + H2 * W2:H * W + H1 * W1 + H2 * W2 + H3 * W3, :]
            #out_maps4 = attned_maps[:, H * W + H1 * W1 + H2 * W2 + H3 * W3:H * W + H1 * W1 + H2 * W2 + H3 * W3 + H4 * W4, :]

            out_maps = [out_maps0, out_maps1, out_maps2, out_maps3]

            # maps_out = [self.out_proj[i](attned_maps[i].permute(0, 2, 1).view(B, self.dim, H, W)) for i in range(len(map_list))]

            maps_out0 = out_maps[0].permute(0, 2, 1).view(B, self.dim, H, W)
            maps_out1 = out_maps[1].permute(0, 2, 1).view(B1, self.dim, H1, W1)
            maps_out2 = out_maps[2].permute(0, 2, 1).view(B2, self.dim, H2, W2)
            maps_out3 = out_maps[3].permute(0, 2, 1).view(B3, self.dim, H3, W3)
            #maps_out4 = out_maps[4].permute(0, 2, 1).view(B4, self.dim, H4, W4)

            maps_out = [maps_out0, maps_out1, maps_out2, maps_out3]
            maps_out_re = [self.out_proj[i](maps_out[i]) for i in range(len(map_list))]

            return maps_out_re


class FeatureMapFusion4L(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.depth = depth
        if depth == 0:
            pass
        else:
            self.dim = dim

            # project all maps to the same channel num
            self.in_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.in_proj.append(nn.Conv2d(in_dim_list[i], dim, kernel_size=1, bias=False))

            self.fusion = FusionTransformerBlock(dim, depth, heads, dim // heads, dim, attn_drop=0., proj_drop=0.)

            self.conv = nn.Conv1d(dim, dim, 1)

            # project all maps back to their origin channel num
            self.out_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.out_proj.append(nn.Conv2d(dim, in_dim_list[i], kernel_size=1, bias=False))

    def forward(self, map_list):

        if self.depth == 0:
            return map_list
        else:
            B, _, H, W = map_list[0].shape
            B1, _, H1, W1 = map_list[1].shape
            B2, _, H2, W2 = map_list[2].shape
            B3, _, H3, W3 = map_list[3].shape
            #B4, _, H4, W4 = map_list[4].shape
            proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in range(len(map_list))]
            # B, L, C where L=HW

            x1 = H * W
            x2 = H1 * W1
            x3 = H2 * W2
            #x4 = H3 * W3

            proj_maps = torch.cat(proj_maps, dim=1)

            attned_maps = self.fusion(proj_maps)
            #attned_maps = self.conv(attned_maps.permute(0, 2, 1))
            #attned_maps = attned_maps.permute(0, 2, 1)

            # attned_maps = attned_maps.chunk(len(map_list), dim=1)
            out_maps0 = attned_maps[:, 0:H * W, :]
            out_maps1 = attned_maps[:, H * W:H * W + H1 * W1, :]
            out_maps2 = attned_maps[:, H * W + H1 * W1:H * W + H1 * W1 + H2 * W2, :]
            out_maps3 = attned_maps[:, H * W + H1 * W1 + H2 * W2:H * W + H1 * W1 + H2 * W2 + H3 * W3, :]
            #out_maps4 = attned_maps[:, H * W + H1 * W1 + H2 * W2 + H3 * W3:H * W + H1 * W1 + H2 * W2 + H3 * W3 + H4 * W4, :]

            out_maps = [out_maps0, out_maps1, out_maps2, out_maps3]

            # maps_out = [self.out_proj[i](attned_maps[i].permute(0, 2, 1).view(B, self.dim, H, W)) for i in range(len(map_list))]

            maps_out0 = out_maps[0].permute(0, 2, 1).view(B, self.dim, H, W)
            maps_out1 = out_maps[1].permute(0, 2, 1).view(B1, self.dim, H1, W1)
            maps_out2 = out_maps[2].permute(0, 2, 1).view(B2, self.dim, H2, W2)
            maps_out3 = out_maps[3].permute(0, 2, 1).view(B3, self.dim, H3, W3)
            #maps_out4 = out_maps[4].permute(0, 2, 1).view(B4, self.dim, H4, W4)

            maps_out = [maps_out0, maps_out1, maps_out2, maps_out3]
            maps_out_re = [self.out_proj[i](maps_out[i]) for i in range(len(map_list))]

            return maps_out_re

#######################################################################################################################
# semantic_map
class BasicBlockSe(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm2d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):

        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1),
            act(),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out


class DropPath(nn.Module):
    """
    Drop connection with pobability p
    """

    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x


class FusedMBConv(nn.Module):
    """
    MBConv with an expansion factor of N, and squeeze-and-excitation module
    """

    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch

        self.stride = stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1,
                                   norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)
        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x


class MapLayer(nn.Module):
    """
    A basic transformer layer for one stage
    No downsample of upsample operation in this layer, they are wraped in the down_block or up_block of UTNet
    """

    def __init__(self, feat_dim, map_dim, out_dim, num_blocks, heads=4, dim_head=64, expansion=1, attn_drop=0.,
                 proj_drop=0., map_size=8, proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()

        dim1 = feat_dim
        dim2 = out_dim

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.blocks.append(BidirectionAttentionBlock(dim1, map_dim, dim2, heads, dim_head, expansion=4,
                               attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, norm=norm, act=act))
            dim1 = out_dim

    def forward(self, x, semantic_map):
        for block in self.blocks:
            x, semantic_map = block(x, semantic_map)

        return x, semantic_map


class BidirectionAttentionBlock(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads, dim_head, expansion=4, norm=nn.BatchNorm2d,
                 act=nn.GELU, attn_drop=0., proj_drop=0., map_size=8, ):
        super().__init__()

        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        # assert proj_type in ['linear', 'depthwise']

        self.norm1 = norm(feat_dim) if norm else nn.Identity()  # norm layer for feature map
        self.norm2 = norm(map_dim) if norm else nn.Identity()  # norm layer for semantic map

        self.attn = BidirectionAttention(feat_dim, map_dim, out_dim, heads=heads, dim_head=dim_head,
                                         attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size)

        self.shortcut = nn.Sequential()
        if feat_dim != out_dim:
            self.shortcut = ConvNormAct(feat_dim, out_dim, kernel_size=1, padding=0, norm=norm, act=act, preact=True)

        # if proj_type == 'linear':
        self.feedforward = FusedMBConv(out_dim, out_dim, expansion=expansion, kernel_size=1, act=act, norm=norm) # 2 conv1x1
        # else:
        # self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=3, act=act, norm=norm, p=proj_drop) # depthwise conv

    def forward(self, x, semantic_map):
        feat = self.norm1(x)
        mapp = self.norm2(semantic_map)

        out, mapp = self.attn(feat, mapp)

        out += self.shortcut(x)
        out = self.feedforward(out)

        mapp += semantic_map

        return out, mapp


class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., map_size=16, proj_type='depthwise'):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size

        # assert proj_type in ['linear', 'depthwise']

        # if proj_type == 'linear':
        self.feat_qv = nn.Conv2d(feat_dim, self.inner_dim * 2, kernel_size=1, bias=False)
        self.feat_out = nn.Conv2d(self.inner_dim, out_dim, kernel_size=1, bias=False)

        # else:
        # self.feat_qv = DepthwiseSeparableConv(feat_dim, self.inner_dim * 2)
        # self.feat_out = DepthwiseSeparableConv(self.inner_dim, out_dim)

        self.map_qv = nn.Conv2d(map_dim, self.inner_dim * 2, kernel_size=1, bias=False)
        self.map_out = nn.Conv2d(self.inner_dim, map_dim, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, semantic_map):
        B, C, H, W = feat.shape

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1)  # B, inner_dim, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1)  # B, inner_dim, rs, rs

        feat_q, feat_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=H, w=W), [feat_q, feat_v])
        map_q, map_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=self.map_size, w=self.map_size), [map_q, map_v])

        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale

        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout
        # add dropout migth cause unstable during training
        map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                             dim_head=self.dim_head, heads=self.heads)

        map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        map_out = rearrange(map_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', b=B, dim_head=self.dim_head,
                            heads=self.heads, h=self.map_size, w=self.map_size)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        map_out = self.proj_drop(self.map_out(map_out))

        return feat_out, map_out


class SEAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: B, L, C.   Batch, sequence length, dim
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')

        attned = self.to_out(attned)

        return attned

class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SEAttention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp_fusion(dim, mlp_dim, dim, drop=proj_drop))
            ]))

    def forward(self, x):

        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x

        return x


class SemanticMapGeneration(nn.Module):
    def __init__(self, feat_dim, map_dim, map_size):
        super().__init__()

        self.map_size = map_size
        self.map_dim = map_dim

        self.map_code_num = map_size * map_size

        self.base_proj = nn.Conv2d(feat_dim, map_dim, kernel_size=3, padding=1, bias=False)
        self.semantic_proj = nn.Conv2d(feat_dim, self.map_code_num, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.base_proj(x)  # B, map_dim, h, w
        weight_map = self.semantic_proj(x)  # B, map_code_num, h, w

        weight_map = weight_map.view(B, self.map_code_num, -1)
        weight_map = F.softmax(weight_map, dim=2)  # B, map_code_num, hw
        feat = feat.view(B, self.map_dim, -1)  # B, map_dim, hw

        semantic_map = torch.einsum('bij,bkj->bik', feat, weight_map)

        return semantic_map.view(B, self.map_dim, self.map_size, self.map_size)


class SemanticMapFusion(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.depth = depth
        if depth == 0:
            pass
        else:
            self.dim = dim

            # project all maps to the same channel num
            self.in_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.in_proj.append(nn.Conv2d(in_dim_list[i], dim, kernel_size=1, bias=False))

            self.fusion = TransformerBlock(dim, depth, heads, dim // heads, dim, attn_drop=0., proj_drop=0.)

            # project all maps back to their origin channel num
            self.out_proj = nn.ModuleList([])
            for i in range(len(in_dim_list)):
                self.out_proj.append(nn.Conv2d(dim, in_dim_list[i], kernel_size=1, bias=False))

    def forward(self, map_list):

        if self.depth == 0:
            return map_list
        else:
            B, _, H, W = map_list[0].shape
            proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in
                         range(len(map_list))]
            # B, L, C where L=HW

            proj_maps = torch.cat(proj_maps, dim=1)

            attned_maps = self.fusion(proj_maps)

            attned_maps = attned_maps.chunk(len(map_list), dim=1)

            maps_out = [self.out_proj[i](attned_maps[i].permute(0, 2, 1).view(B, self.dim, H, W)) for i in
                        range(len(map_list))]

            return maps_out


class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """

    def __init__(self, dim, out_dim, norm=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Conv2d(4*dim, out_dim, kernel_size=1, bias=False)

        self.norm = norm(4*dim)

    def forward(self, x, semantic_map=None):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], 1) # B, 4C, H, W

        x = self.norm(x)
        x = self.reduction(x)

        return x


class down_block_trans_WASE(nn.Module):
    def __init__(self, in_ch, out_ch, input_resolution, window_size, shift_size, num_block, bottleneck=False,
                 maxpool=True, heads=4, attn_drop=0., map_generate=True, map_size=8):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block_add = BasicBlock_UTNet

        attn_block = SwinTransformerBlock4T

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        block_list.append(block_add(out_ch, out_ch, stride=1))

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, input_resolution, heads, window_size, shift_size, 4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=attn_drop))
        self.blocks = nn.Sequential(*block_list)

        map_dim = out_ch
        self.map_generate = map_generate

        if map_generate:
            self.map_gen = SemanticMapGeneration(out_ch, map_dim, map_size)

        self.trans_blocks = MapLayer(out_ch, map_dim, out_ch, num_blocks=1,
                                     heads=heads, dim_head=16, norm=nn.BatchNorm2d, act=nn.GELU, expansion=4,
                                     attn_drop=attn_drop, proj_drop=0, map_size=map_size)

    def forward(self, x):

        out = self.blocks(x)

        if self.map_generate:
            semantic_map = self.map_gen(out)
        else:
            semantic_map = None

        out, semantic_map = self.trans_blocks(out, semantic_map)

        return out, semantic_map


class down_Se(nn.Module):
    def __init__(self, in_ch, out_ch, trans_num, conv_block=BasicBlockSe,
                 heads=4, dim_head=16, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                 norm=nn.BatchNorm2d, act=nn.GELU, map_generate=False, map_dim=None):
        super().__init__()

        map_dim = out_ch if map_dim is None else map_dim
        self.map_generate = map_generate
        if map_generate:
            self.map_gen = SemanticMapGeneration(out_ch, map_dim, map_size)
        '''
        block_list = []
        
        block = BasicBlock
        
        block_list.append(nn.MaxPool2d(2))
        block_list.append(block(in_ch, out_ch))
        
        self.conv = nn.Sequential(*block_list)
        '''
        self.patch_merging = PatchMerging(in_ch, out_ch, norm=norm)

        self.conv_blocks = conv_block(out_ch, out_ch, norm=norm, act=act)

        self.trans_blocks = MapLayer(out_ch, map_dim, out_ch, num_blocks=trans_num,
                                       heads=heads, dim_head=dim_head, norm=norm, act=act, expansion=expansion,
                                       attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size)

    def forward(self, x):

        #x = self.conv(x)
        x = self.patch_merging(x)

        out = self.conv_blocks(x)
        if self.map_generate:
            semantic_map = self.map_gen(out)
        else:
            semantic_map = None

        out, semantic_map = self.trans_blocks(out, semantic_map)

        return out, semantic_map

class up_Se(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim=2, pad=False, map_shortcut=False):
        super(up_Se, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)

        map_dim = out_ch

        self.map_shortcut = map_shortcut

        if map_shortcut:
            self.map_reduction = nn.Conv2d(in_ch_1 + in_ch_2, map_dim, kernel_size=1, bias=False)
        else:
            self.map_reduction = nn.Conv2d(in_ch_1, map_dim, kernel_size=1, bias=False)

        #if map_gen:
            #self.map_gene = SemanticMapGeneration(in_ch_2, in_ch_2, map_size)

        if pad:
            self.conv = conv_block(in_ch_1 + in_ch_2, out_ch, modelDim)
        else:
            self.conv = conv_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)


        self.trans = MapLayer(out_ch, map_dim, out_ch, num_blocks=1,
                              heads=4, dim_head=16, norm=nn.BatchNorm2d, act=nn.GELU,
                              attn_drop=0, proj_drop=0, map_size=8)
        self.conv_block = BasicBlockSe(out_ch, out_ch, norm=nn.BatchNorm2d, act=nn.GELU)

    def forward(self, x1, x2, map1, map2=None):  # x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)

        startIndexDim2 = (x2.size()[2] - x1.size()[2]) // 2
        startIndexDim3 = (x2.size()[3] - x1.size()[3]) // 2
        x = torch.cat(
            [x2[:, :, startIndexDim2:x1.size()[2] + startIndexDim2, startIndexDim3:x1.size()[3] + startIndexDim3], x1],
            dim=1)

        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.map_shortcut and map2 is not None:
            semantic_map = torch.cat([map1, map2], dim=1)
        else:
            semantic_map = map1

        if semantic_map is not None:
            semantic_map = self.map_reduction(semantic_map)

        x, semantic_map = self.trans(x, semantic_map)
        x = self.conv_block(x)

        return x, semantic_map


#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# UTNet Window attention part


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


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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

        #trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class WinMlp(nn.Module):
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


class SwinTransformerBlock4T(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
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
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(0)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = WinMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
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
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        x_B, x_C, x_H, x_W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(x_B, x_H * x_W, x_C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)

        return x


#######################################################################################################################

class Mlp(nn.Module):
    def __init__(self, in_ch, hid_ch=None, out_ch=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hid_ch = hid_ch or in_ch

        self.fc1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*2, stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue

        return out

class BasicBlock_UTNet(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.ln1 = nn.LayerNorm(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    #nn.LayerNorm(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):

        residue = x

        #out = self.bn1(x)
        #out = self.relu(out)
        #out = self.conv1(out)

        #out = self.bn2(out)
        #out = self.relu(out)
        #out = self.conv2(out)

        #out += self.shortcut(residue)
        out = self.shortcut(residue)

        return out

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.InstanceNorm2d(planes)  # Test for Unet
        self.relu =nn.LeakyReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.InstanceNorm2d(planes)  # Test for Unet

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):
        '''
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)
        '''

        #Test for Unet
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)



        return out


class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch // heads, attn_drop=attn_drop,
                                    proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                    rel_pos=rel_pos)

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):
        out = self.bn1(x)
        out, q_k_attn = self.attn(out)

        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class BasicTransDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True):
        super().__init__()

        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = LinearAttentionDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch // heads, attn_drop=attn_drop,
                                           proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                           rel_pos=rel_pos)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        # x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)

        out, q_k_attn = self.attn(x2, x1)

        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out

############################################################################################################
# Transformer


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):

        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class LinearAttentionDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16,
                 projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim * 2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):

        B, C, H, W = x.shape  # low-res feature shape
        BH, CH, HH, WH = q.shape  # high-res feature shape

        k, v = self.to_kv(x).chunk(2, dim=1)  # B, inner_dim, H, W
        q = self.to_q(q)  # BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(HH, WH)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class RelativePositionEmbedding(nn.Module):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None]  # h, h
        relative_coords += self.shape - 1  # shift to start from 0

        self.register_buffer('relative_position_index', relative_coords)

    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index = torch.repeat_interleave(self.relative_position_index, W // self.shape, dim=0)  # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits


class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)
        if H % self.h != 0:
            add_h = torch.zeros(H-(H//self.h)*self.h, self.w, self.h * self.w, self.num_heads).to(device)
            relative_position_bias_expand_h = torch.cat((relative_position_bias_expand_h, add_h), 0)
        else:
            relative_position_bias_expand_h = relative_position_bias_expand_h

        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH
        if W % self.w != 0:
            added = torch.zeros(relative_position_bias_expanded.shape[0], W-(W//self.w)*self.w, self.h * self.w, self.num_heads).to(device)
            relative_position_bias_expanded = torch.cat((relative_position_bias_expanded, added), 1)
        else:
            relative_position_bias_expanded = relative_position_bias_expanded








        #relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               #self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        relative_position_bias_expanded = relative_position_bias_expanded.view(-1, self.h * self.w,
                                                                               self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)


        return relative_position_bias_expanded
######################################################################################################################
# Unet Transformer component
class down_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, maxpool=True, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):

        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block_add = BasicBlock_UTNet

        attn_block = BasicTransBlock


        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        block_list.append(block_add(out_ch, out_ch, stride=1))

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out = self.blocks(x)

        return out


class down_block_trans_WA(nn.Module):
    def __init__(self, in_ch, out_ch, input_resolution, window_size, shift_size, num_block, bottleneck=False, maxpool=True, heads=4, attn_drop=0.):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block_add = BasicBlock_UTNet

        attn_block = SwinTransformerBlock4T

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        #self.blocks = nn.Sequential(*block_list)

        #block_list_trans = []

        #block_list.append(block_add(out_ch, out_ch, stride=1))
        block_list.append(block_add(out_ch, out_ch, stride=1))

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, input_resolution, heads, window_size, shift_size, 4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out = self.blocks(x)
        #out_cnn = out
        #out = self.blocks_trans(out)

        return out


class down_block_trans_WA_Para(nn.Module):
    def __init__(self, in_ch, out_ch, input_resolution, window_size, shift_size, num_block, maxpool=True, heads=4, attn_drop=0.):
        super().__init__()

        block_list = []

        #block = BasicBlock
        block = BasicBlock_UTNet

        attn_block = SwinTransformerBlock4T

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, input_resolution, heads, window_size, shift_size, 4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out = self.blocks(x)

        return out



class up_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                 reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop,
                                                   proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                                   rel_pos=rel_pos)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock
        attn_block = BasicTransBlock

        block_list = []

        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos))

        block_list.append(block(2 * out_ch, out_ch, stride=1))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x1: low-res feature, x2: high-res feature
        out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)

        return out


class block_trans(nn.Module):
    def __init__(self, in_ch, num_block, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16,
                 projection='interp', rel_pos=True):
        super().__init__()

        block_list = []

        attn_block = BasicTransBlock

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(in_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):
        out = self.blocks(x)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock


        if pool:
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2,2),bottleneck=False):
        super().__init__()
        self.scale=scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock


        block_list = []
        block_list.append(block(2*out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


############################################################################################################
# Custom represents our utilized and developed deep learning model. It is based on the U-Net architecture:
# ----- Custom Unet 2D - Pooling-Encoder + (Transposed/Upsampling)-Decoder + DoubleConvs ----- #
class Custom(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(Custom, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        #self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up0 = upWithPad(1024, 512, 512, 2)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False, conv5=False)
        #Test for the size of 320 input
        self.outadd = nn.Conv2d(32, 32, kernel_size=3, bias=True)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)



    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        #Test for 320
        x = self.outadd(x)
        x = self.outc(x)
        return x

# This class represents the UTNet
class UTNet(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=1, reduce_size=8, block_list='234', num_blocks=[1, 2, 4],
                 projection='interp', num_heads=[2, 4, 8], fusion_dim=512, fusion_heads=16, fusion_depth=2,
                 attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        # Unet++
        nb_filter = [16, 32, 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = UpPlus()

        self.conv0_0 = Block_plus(32, nb_filter[1], nb_filter[1])
        self.conv1_0 = Block_plus(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv2_0 = Block_plus(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv3_0 = Block_plus(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv4_0 = Block_plus(nb_filter[4], nb_filter[5], nb_filter[5])

        self.conv0_1 = Block_plus(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_1 = Block_plus(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_1 = Block_plus(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv3_1 = Block_plus(nb_filter[4] + nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_2 = Block_plus(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2 = Block_plus(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_2 = Block_plus(nb_filter[3] * 2 + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_3 = Block_plus(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_3 = Block_plus(nb_filter[2] * 3 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_4 = Block_plus(nb_filter[1] * 4 + nb_filter[2], nb_filter[1], nb_filter[1])

        # Multi-scale fusion part
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan,
                        32*base_chan, 16*base_chan, 8*base_chan, 4*base_chan]

        #0 for 5 level and 1 for 4 level
        self.map_fusion = FeatureMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth,
                                           norm=nn.BatchNorm2d)

        self.semanticmap_fusion = SemanticMapFusion(chan_num[2:5], fusion_dim, fusion_heads, depth=fusion_depth,
                                                    norm=nn.BatchNorm2d)

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        #self.inc = [BasicBlock_UTNet(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                            attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                            projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4],
                                      dim_head=base_chan // num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop,
                                      reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            #self.inc.append(BasicBlock_UTNet(base_chan, base_chan))
            #self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
            #Unet decoder
            self.up4 = up_noAttn(2 * base_chan, 1 * base_chan, 1 * base_chan, num_block=1)
            #self.up4 = up(2 * base_chan, 1 * base_chan, 1 * base_chan, 2)
            #self.up4 = up_Se(2 * base_chan, 1 * base_chan, 1 * base_chan, 2, pad=False, map_shortcut=False)
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-4], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-4], dim_head=2 * base_chan // num_heads[-4],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            #self.down1 = down_block_trans_WA(base_chan, 2 * base_chan, (160, 160), 32, 8, num_block=num_blocks[-4],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], attn_drop=attn_drop)
            self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos)
            #self.down1 = down(base_chan, 2 * base_chan, 2)
            #self.down1 = down_Se(base_chan, 2 * base_chan, 1)
            #self.up3 = up_noAttn(4 * base_chan, 2 * base_chan, 2 * base_chan, num_block=1)
            #self.up3 = up_Se(4 * base_chan, 2 * base_chan, 2 * base_chan, 2, pad=False, map_shortcut=False)
        else:
            self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
            self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-3], dim_head=4 * base_chan // num_heads[-3],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            #self.down2 = down_block_trans_WA(2 * base_chan, 4 * base_chan, (80, 80), 20, 5, num_block=num_blocks[-3],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3],
                                             #attn_drop=attn_drop)
            #self.down2 = down_block_trans_WASE(2 * base_chan, 4 * base_chan, (80, 80), 20, 5, num_block=num_blocks[-3],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-3],
                                             #attn_drop=attn_drop, map_generate=True, map_size=8)
            #self.down2 = down_Se(2 * base_chan, 4 * base_chan, 1, conv_block=BasicBlockSe,
                                 #heads=4, dim_head=16, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                                 #norm=nn.BatchNorm2d, act=nn.GELU, map_generate=True, map_dim=None)
            self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos)
            #self.down2 = down(2 * base_chan, 4 * base_chan, 2)
            #self.up2 = up_noAttn(8 * base_chan, 4 * base_chan, 4 * base_chan, num_block=1)
            #self.up2 = up_Se(8 * base_chan, 4 * base_chan, 4 * base_chan, 2, pad=False, map_shortcut=False)

        else:
            self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(4 * base_chan, 8 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-2], dim_head=8 * base_chan // num_heads[-2],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            #self.down3 = down_block_trans_WA(4 * base_chan, 8 * base_chan, (40, 40), 20, 5, num_block=num_blocks[-2],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2],
                                             #attn_drop=attn_drop)
            #self.down3 = down_block_trans_WASE(4 * base_chan, 8 * base_chan, (40, 40), 20, 5, num_block=num_blocks[-2],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-2],
                                             #attn_drop=attn_drop, map_generate=True, map_size=8)
            #self.down3 = down_Se(4 * base_chan, 8 * base_chan, 1, conv_block=BasicBlockSe,
                                 #heads=4, dim_head=16, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                                 #norm=nn.BatchNorm2d, act=nn.GELU, map_generate=True, map_dim=None)
            #self.up1 = up_block_trans(16 * base_chan, 8 * base_chan, num_block=0, bottleneck=bottleneck,
                                      #heads=num_heads[-1], dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                      #proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      #rel_pos=rel_pos)
            #self.down3 = down(4 * base_chan, 8 * base_chan, 2)
            self.up1 = up_noAttn(16 * base_chan, 8 * base_chan, 8 * base_chan, num_block=1)
            #self.up1 = up_Se(16 * base_chan, 8 * base_chan, 8 * base_chan, 2, pad=False, map_shortcut=True)

        else:
            self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)
            self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)


        if '4' in block_list:
            self.down4 = down_block_trans(8 * base_chan, 16 * base_chan, num_block=num_blocks[-1],
                                          bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                          dim_head=16 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                          proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                          rel_pos=rel_pos)
            #self.down4 = down_block_trans_WA(8 * base_chan, 16 * base_chan, (20, 20), 20, 0, num_block=num_blocks[-1],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                             #attn_drop=attn_drop)
            #self.down4 = down_block_trans_WASE(8 * base_chan, 16 * base_chan, (20, 20), 20, 0, num_block=num_blocks[-1],
                                             #bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                             #attn_drop=attn_drop, map_generate=True, map_size=8)
            #self.down4 = down_Se(8 * base_chan, 16 * base_chan, 1, conv_block=BasicBlockSe,
                                 #heads=4, dim_head=16, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                                 #norm=nn.BatchNorm2d, act=nn.GELU, map_generate=True, map_dim=None)
            #self.down4 = down(8 * base_chan, 16 * base_chan, 2)
        else:
            self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=2)

        #For the test of size of 320 input
        self.outadd = nn.Conv2d(base_chan, base_chan, kernel_size=3, bias=True)
        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        ###############################################################################################################
        #Try to add a level with small resolution to improve the capture of features of low_level information
        self.down5 = down(16 * base_chan, 32 * base_chan, 2)
        self.up0 = upWithPad(32 * base_chan, 16 * base_chan, 16 * base_chan, 2)
        #self.down5 = down_Se(16 * base_chan, 32 * base_chan, 1, conv_block=BasicBlockSe,
                 #heads=4, dim_head=16, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                 #norm=nn.BatchNorm2d, act=nn.GELU, map_generate=True, map_dim=None)
        #self.up0 = up_Se(32 * base_chan, 16 * base_chan, 16 * base_chan, 2, pad=True, map_shortcut=True)

        #add skip_connection_fusion_res
        self.skip_fusion = nn.ModuleList([])
        for i in range(len(chan_num[1:4])):
            self.skip_fusion.append(nn.Conv2d(2 * chan_num[i+1], chan_num[i+1], kernel_size=1, bias=False))

        ###############################################################################################################

        if aux_loss:
            self.out1 = nn.Conv2d(8 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2 * base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        #x0_1 = self.conv0_1(self.up(x2, x1))

        x3 = self.down2(x2)
        #x1_1 = self.conv1_1(self.up(x3, x2))
        #x0_2 = self.conv0_2(self.up(x1_1, torch.cat([x1, x0_1], dim=1)))

        x4 = self.down3(x3)
        #x2_1 = self.conv2_1(self.up(x4, x3))
        #x1_2 = self.conv1_2(self.up(x2_1, torch.cat([x2, x1_1], dim=1)))
        #x0_3 = self.conv0_3(self.up(x1_2, torch.cat([x1, x0_1, x0_2], dim=1)))

        x5 = self.down4(x4)
        #x3_1 = self.conv3_1(self.up(x5, x4))
        #x2_2 = self.conv2_2(self.up(x3_1, torch.cat([x3, x2_1], dim=1)))
        #x1_3 = self.conv1_3(self.up(x2_2, torch.cat([x2, x1_1, x1_2], dim=1)))
        x#0_4 = self.conv0_4(self.up(x1_3, torch.cat([x1, x0_1, x0_2, x0_3], dim=1)))
#
        x6 = self.down5(x5)
        #x4, map4 = self.down3(x3)
        #x5, map5 = self.down4(x4)
        #Try to add a level
        #x6, map6 = self.down5(x5)

        #####################################################
        #x2_r = self.down1_r(x1)
        x#2 = torch.cat([x2, x2_r], dim=1)
        #x2 = self.convres1(x2)

        #x3_r = self.down2_r(x2)
        #x3 = torch.cat([x3, x3_r], dim=1)
        #x3 = self.convres2(x3)

        #x4_r = self.down3_r(x3)
        #x4 = torch.cat([x4, x4_r], dim=1)
        #x4 = self.convres3(x4)
        #####################################################

        #Multi-scale fusion part
        #map_list = [x2, x3, x4, x5, x6]
        #map_list = [x3, x4, x5]
        #map_list = self.map_fusion(map_list)

        #x6_f = torch.cat([x6, map_list[3]], dim=1)
        #x6_f = self.skip_fusion[3](x6_f)
        #x5_f = torch.cat([x5, map_list[2]], dim=1)
        #x5_f = self.skip_fusion[2](x5_f)
        #x4_f = torch.cat([x4, map_list[1]], dim=1)
        #x4_f = self.skip_fusion[1](x4_f)
        #x3_f = torch.cat([x3, map_list[0]], dim=1)
        #x3_f = self.skip_fusion[0](x3_f)
        #x2_f = torch.cat([x2, map_list[0]], dim=1)
        #x2_f = self.skip_fusion[0](x2_f)

        #semanticmap_list = [map4, map5, map6]
        #semanticmap_list = self.semanticmap_fusion(semanticmap_list)


        if self.aux_loss:
            #out = self.up0(x6, x5)
            out = self.up1(x5, x4, x4_cnn)
            #out = self.up0(x6_f, x5_f)
            #out = self.up1(out, x4)               #For added level
            #out = self.up1(out, x4_f)  # For added level
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            #out = self.up2(out, x3)
            out = self.up2(out, x3, x3_cnn)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            #out = self.up3(out, x2)
            out = self.up3(out, x2, x2_cnn)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:

            out = self.up0(x6, x5)
            out = self.up1(out, x4)  # For added level
            out = self.up2(out, x3)

            #out, semantic_map = self.up0(x6, x5, semanticmap_list[2], semanticmap_list[1])
            #out = self.up1(x5, x4)
            #out, semantic_map = self.up1(out, x4, semantic_map, semanticmap_list[0])              #For added level
            #out, semantic_map = self.up2(out, x3, semantic_map, None)


            #for multi-scale fusion
            #out = self.up0(x6_f, x5_f)
            #out = self.up1(out, x4_f)  # For added level
            #out = self.up2(out, x3_f)

            out = self.up3(out, x2)
            #out = self.up3(out, map_list[0])
            out = self.up4(out, x1)
            #add to reduce resolution to 258
            #out = self.outadd(out)
            out = self.outc(out)
            #out = out[:, :, 31:289, 31:289]

            return out[:, :, 1:259, 1:259]


class UTNetPara(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=1, reduce_size=8, block_list='234', num_blocks=[1, 2, 4],
                 projection='interp', num_heads=[2, 4, 8], fusion_dim=512, fusion_heads=16, fusion_depth=2,
                 attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        # Multi-scale fusion part
        chan_num = [2 * base_chan, 4 * base_chan, 8 * base_chan, 16 * base_chan,
                    32 * base_chan, 16 * base_chan, 8 * base_chan, 4 * base_chan]

        # 0 for 5 level and 1 for 4 level
        self.map_fusion = FeatureMapFusion(chan_num[1:5], fusion_dim, fusion_heads, depth=fusion_depth,
                                           norm=nn.BatchNorm2d)

        self.semanticmap_fusion = SemanticMapFusion(chan_num[2:5], fusion_dim, fusion_heads, depth=fusion_depth,
                                                    norm=nn.BatchNorm2d)

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        #self.inc = [BasicBlock(in_chan*3, base_chan)]
        # self.inc = [BasicBlock_UTNet(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                            attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                            projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4],
                                      dim_head=base_chan // num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop,
                                      reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            # self.inc.append(BasicBlock_UTNet(base_chan, base_chan))
            # self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
            # Unet decoder
            self.up4 = up_noAttn(2 * base_chan, 1 * base_chan, 1 * base_chan, num_block=1)
            # self.up4 = up(2 * base_chan, 1 * base_chan, 1 * base_chan, 2)
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1_trans = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-4], bottleneck=bottleneck,
                                                maxpool=maxpool, heads=num_heads[-4], dim_head=2 * base_chan // num_heads[-4],
                                                attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                                projection=projection, rel_pos=rel_pos)
            #self.down1_trans = down_block_trans_WA_Para(base_chan, 2 * base_chan, (160, 160), 32, 8, num_block=num_blocks[-4],
                                                #maxpool=maxpool, heads=num_heads[-4], attn_drop=attn_drop)
            # self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down1_cnn = down(base_chan, 2 * base_chan, 2)
            self.up3 = up_noAttn(4 * base_chan, 2 * base_chan, 2 * base_chan, num_block=1)
            self.stadge1 = stadge_block(2 * 2 * base_chan, 2 * base_chan)
        else:
            self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
            self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)

        if '2' in block_list:
            # self.down2 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
            # maxpool=maxpool, heads=num_heads[-3], dim_head=4 * base_chan // num_heads[-3],
            # attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
            # projection=projection, rel_pos=rel_pos)
            self.down2_trans = down_block_trans_WA_Para(2 * base_chan, 4 * base_chan, (80, 80), 20, 5,
                                                        num_block=num_blocks[-3],maxpool=maxpool, heads=num_heads[-3],
                                                        attn_drop=attn_drop)
            # self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down2_cnn = down(2 * base_chan, 4 * base_chan, 2)
            self.up2 = up_noAttn(8 * base_chan, 4 * base_chan, 4 * base_chan, num_block=1)
            self.up2_trans = upwithTrans(8 * base_chan, 4 * base_chan, 4 * base_chan, 4)
            self.defusion2 = DeFusion(8 * base_chan, 4 * base_chan)
            self.stadge2 = stadge_block(2 * 4 * base_chan, 4 * base_chan)

        else:
            self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)

        if '3' in block_list:
            # self.down3 = down_block_trans(4 * base_chan, 8 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
            # maxpool=maxpool, heads=num_heads[-2], dim_head=8 * base_chan // num_heads[-2],
            # attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
            # projection=projection, rel_pos=rel_pos)
            self.down3_trans = down_block_trans_WA_Para(4 * base_chan, 8 * base_chan, (40, 40), 20, 5,
                                                        num_block=num_blocks[-2],
                                                        maxpool=maxpool, heads=num_heads[-2],
                                                        attn_drop=attn_drop)
            # self.up1 = up_block_trans(16 * base_chan, 8 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-1], dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down3_cnn = down(4 * base_chan, 8 * base_chan, 2)
            self.up1 = up_noAttn(16 * base_chan, 8 * base_chan, 8 * base_chan, num_block=1)
            self.up1_trans = upwithTrans(16 * base_chan, 8 * base_chan, 8 * base_chan, 4)
            self.defusion1 = DeFusion(16 * base_chan, 8 * base_chan)
            self.stadge3 = stadge_block(2 * 8 * base_chan, 8 * base_chan)

        else:
            self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)
            self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        if '4' in block_list:
            # self.down4 = down_block_trans(8 * base_chan, 16 * base_chan, num_block=num_blocks[-1],
            # bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
            # dim_head=16 * base_chan // num_heads[-1], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down4_trans = down_block_trans_WA_Para(8 * base_chan, 16 * base_chan, (20, 20), 20, 0,
                                                        num_block=num_blocks[-1],
                                                        maxpool=maxpool, heads=num_heads[-1],
                                                        attn_drop=attn_drop)
            self.down4_cnn = down(8 * base_chan, 16 * base_chan, 2)
            self.stadge4 = stadge_block(2 * 16 * base_chan, 16 * base_chan)
        else:
            self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=2)

        # For the test of size of 320 input
        self.outadd = nn.Conv2d(base_chan, base_chan, kernel_size=3, bias=True)
        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        ###############################################################################################################
        # Try to add a level with small resolution to improve the capture of features of low_level information
        self.down5_cnn = down(16 * base_chan, 32 * base_chan, 2)
        self.down5_trans = down_block_trans_WA_Para(16 * base_chan, 32 * base_chan, (10, 10), 10, 0,
                                                    num_block=num_blocks[-1],
                                                    maxpool=maxpool, heads=num_heads[-1],
                                                    attn_drop=attn_drop)
        self.up0 = upWithPad(32 * base_chan, 16 * base_chan, 16 * base_chan, 2)
        self.up0_trans = upwithTrans(32 * base_chan, 16 * base_chan, 16 * base_chan, 4, True)
        self.defusion0 = DeFusion(32 * base_chan, 16 * base_chan)
        self.stadge5 = stadge_block(2 * 32 * base_chan, 32 * base_chan)

        # add skip_connection_fusion_res
        self.skip_fusion = nn.ModuleList([])
        for i in range(len(chan_num[1:5])):
            self.skip_fusion.append(nn.Conv2d(2 * chan_num[i + 1], chan_num[i + 1], kernel_size=1, bias=False))

        ###############################################################################################################

        if aux_loss:
            self.out1 = nn.Conv2d(8 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2 * base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):

        x1 = self.inc(x)

        x2_cnn = self.down1_cnn(x1)
        x2_trans = self.down1_trans(x1)
        x2 = self.stadge1(torch.cat([x2_cnn, x2_trans], dim=1))

        x3_cnn = self.down2_cnn(x2)
        #x3_trans = self.down2_trans(x2)
        x3_trans = self.down2_trans(x2_trans)
        x3 = self.stadge2(torch.cat([x3_cnn, x3_trans], dim=1))

        x4_cnn = self.down3_cnn(x3)
        #x4_trans = self.down3_trans(x3)
        x4_trans = self.down3_trans(x3_trans)
        x4 = self.stadge3(torch.cat([x4_cnn, x4_trans], dim=1))

        x5_cnn = self.down4_cnn(x4)
        #x5_trans = self.down4_trans(x4)
        x5_trans = self.down4_trans(x4_trans)
        x5 = self.stadge4(torch.cat([x5_cnn, x5_trans], dim=1))

        x6_cnn = self.down5_cnn(x5)
        #x6_trans = self.down5_trans(x5)
        x6_trans = self.down5_trans(x5_trans)
        x6 = self.stadge5(torch.cat([x6_cnn, x6_trans], dim=1))
        #####################################################

        # Multi-scale fusion part
        # map_list = [x2, x3, x4, x5, x6]
        map_list = [x3, x4, x5, x6]
        map_list = self.map_fusion(map_list)

        x6_f = torch.cat([x6, map_list[3]], dim=1)
        x6_f = self.skip_fusion[3](x6_f)
        x5_f = torch.cat([x5, map_list[2]], dim=1)
        x5_f = self.skip_fusion[2](x5_f)
        x4_f = torch.cat([x4, map_list[1]], dim=1)
        x4_f = self.skip_fusion[1](x4_f)
        x3_f = torch.cat([x3, map_list[0]], dim=1)
        x3_f = self.skip_fusion[0](x3_f)
        # x2_f = torch.cat([x2, map_list[0]], dim=1)
        # x2_f = self.skip_fusion[0](x2_f)

        if self.aux_loss:
            out = self.up0(x6, x5)
            out = self.up1(out, x4)
            # out = self.up0(x6_f, x5_f)
            # out = self.up1(out, x4)               #For added level
            # out = self.up1(out, x4_f)  # For added level
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            # out = self.up2(out, x3)
            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            # out = self.up3(out, x2)
            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:

            out = self.up0(x6_f, x5_f)
            #out = self.up0(x6, x5_f)
            out_trans = self.up0_trans(x6, x5)
            out = self.defusion0(out, out_trans)

            out = self.up1(out, x4_f)  # For added level
            #out = self.up1(out, x4)  # For added level
            out_trans = self.up1_trans(out_trans, x4)
            out = self.defusion1(out, out_trans)

            out = self.up2(out, x3_f)
            #out = self.up2(out, x3)
            out_trans = self.up2_trans(out_trans, x3)
            out = self.defusion2(out, out_trans)

            # for multi-scale fusion
            # out = self.up0(x6_f, x5_f)
            # out = self.up1(out, x4_f)  # For added level
            # out = self.up2(out, x3_f)

            out = self.up3(out, x2)
            # out = self.up3(out, map_list[0])
            out = self.up4(out, x1)
            # add to reduce resolution to 258
            # out = self.outadd(out)
            out = self.outc(out)
            # out = out[:, :, 31:289, 31:289]

            return out[:, :, 1:259, 1:259]


class UTNetParaV2(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=1, reduce_size=8, block_list='234', num_blocks=[1, 2, 4],
                 projection='interp', num_heads=[2, 4, 8], fusion_dim=512, fusion_heads=16, fusion_depth=2,
                 attn_drop=0., proj_drop=0., bottleneck=False, maxpool=True, rel_pos=True, aux_loss=False):
        super().__init__()

        # Multi-scale fusion part
        chan_num = [2 * base_chan, 4 * base_chan, 8 * base_chan, 16 * base_chan,
                    32 * base_chan, 16 * base_chan, 8 * base_chan, 4 * base_chan]

        # 0 for 5 level and 1 for 4 level
        self.map_fusion = FeatureMapFusion4L(chan_num[1:5], fusion_dim, fusion_heads, depth=fusion_depth,
                                           norm=nn.BatchNorm2d)

        self.semanticmap_fusion = SemanticMapFusion(chan_num[2:5], fusion_dim, fusion_heads, depth=fusion_depth,
                                                    norm=nn.BatchNorm2d)

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        # self.inc = [BasicBlock_UTNet(in_chan, base_chan)]
        if '0' in block_list:
            self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                            attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                            projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4],
                                      dim_head=base_chan // num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop,
                                      reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            # self.inc.append(BasicBlock_UTNet(base_chan, base_chan))
            # self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
            # Unet decoder
            self.up4 = up_noAttn(2 * base_chan, 1 * base_chan, 1 * base_chan, num_block=1)
            # self.up4 = up(2 * base_chan, 1 * base_chan, 1 * base_chan, 2)
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1_trans = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-4],
                                                bottleneck=bottleneck,
                                                maxpool=maxpool, heads=num_heads[-4],
                                                dim_head=2 * base_chan // num_heads[-4],
                                                attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                                projection=projection, rel_pos=rel_pos)
            # self.down1 = down_block_trans_WA(base_chan, 2 * base_chan, (160, 160), 32, 8, num_block=num_blocks[-4],
            # bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-4], attn_drop=attn_drop)
            # self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down1_cnn = down(base_chan, 2 * base_chan, 2)
            self.up3 = up_noAttn(4 * base_chan, 2 * base_chan, 2 * base_chan, num_block=1)
            self.stadge1 = stadge_block(2 * 2 * base_chan, 2 * base_chan)
        else:
            self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
            self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)

        if '2' in block_list:
            # self.down2 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
            # maxpool=maxpool, heads=num_heads[-3], dim_head=4 * base_chan // num_heads[-3],
            # attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
            # projection=projection, rel_pos=rel_pos)
            self.down2_trans = down_block_trans_WA_Para(2 * base_chan, 4 * base_chan, (80, 80), 20, 5,
                                                        num_block=num_blocks[-3],
                                                        maxpool=maxpool, heads=num_heads[-3],
                                                        attn_drop=attn_drop)
            # self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down2_cnn = down(2 * base_chan, 4 * base_chan, 2)
            self.up2 = up_noAttn(8 * base_chan, 4 * base_chan, 4 * base_chan, num_block=1)
            self.up_skip2 = up_V2(8 * base_chan, 4 * base_chan, 4 * base_chan)
            self.stadge2 = stadge_block(2 * 4 * base_chan, 4 * base_chan)
            self.catconv = conv_block(2 * 4 * base_chan, 4 * base_chan, 2)

        else:
            self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)

        if '3' in block_list:
            # self.down3 = down_block_trans(4 * base_chan, 8 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
            # maxpool=maxpool, heads=num_heads[-2], dim_head=8 * base_chan // num_heads[-2],
            # attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
            # projection=projection, rel_pos=rel_pos)
            self.down3_trans = down_block_trans_WA_Para(4 * base_chan, 8 * base_chan, (40, 40), 20, 5,
                                                        num_block=num_blocks[-2],
                                                        maxpool=maxpool, heads=num_heads[-2],
                                                        attn_drop=attn_drop)
            # self.up1 = up_block_trans(16 * base_chan, 8 * base_chan, num_block=0, bottleneck=bottleneck,
            # heads=num_heads[-1], dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down3_cnn = down(4 * base_chan, 8 * base_chan, 2)
            self.up1 = up_noAttn(16 * base_chan, 8 * base_chan, 8 * base_chan, num_block=1)
            self.up_skip1 = up_V2(16 * base_chan, 8 * base_chan, 8 * base_chan)
            self.stadge3 = stadge_block(2 * 8 * base_chan, 8 * base_chan)

        else:
            self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)
            self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        if '4' in block_list:
            # self.down4 = down_block_trans(8 * base_chan, 16 * base_chan, num_block=num_blocks[-1],
            # bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
            # dim_head=16 * base_chan // num_heads[-1], attn_drop=attn_drop,
            # proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
            # rel_pos=rel_pos)
            self.down4_trans = down_block_trans_WA_Para(8 * base_chan, 16 * base_chan, (20, 20), 20, 0,
                                                        num_block=num_blocks[-1],
                                                        maxpool=maxpool, heads=num_heads[-1],
                                                        attn_drop=attn_drop)
            self.down4_cnn = down(8 * base_chan, 16 * base_chan, 2)
            self.stadge4 = stadge_block(2 * 16 * base_chan, 16 * base_chan)
        else:
            self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=2)

        # For the test of size of 320 input
        self.outadd = nn.Conv2d(base_chan, base_chan, kernel_size=3, bias=True)
        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        ###############################################################################################################
        # Try to add a level with small resolution to improve the capture of features of low_level information
        self.down5_cnn = down(16 * base_chan, 32 * base_chan, 2)
        self.down5_trans = down_block_trans_WA_Para(16 * base_chan, 32 * base_chan, (10, 10), 10, 0,
                                                    num_block=num_blocks[-1],
                                                    maxpool=maxpool, heads=num_heads[-1],
                                                    attn_drop=attn_drop)
        self.up0 = upWithPad(32 * base_chan, 16 * base_chan, 16 * base_chan, 2)
        self.up_skip0 = up_V2(32 * base_chan, 16 * base_chan, 16 * base_chan, True)
        self.stadge5 = stadge_block(2 * 32 * base_chan, 32 * base_chan)

        # add skip_connection_fusion_res
        self.skip_fusion = nn.ModuleList([])
        for i in range(len(chan_num[1:5])):
            self.skip_fusion.append(nn.Conv2d(2 * chan_num[i + 1], chan_num[i + 1], kernel_size=1, bias=False))

        ###############################################################################################################

        if aux_loss:
            self.out1 = nn.Conv2d(8 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2 * base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):

        x1 = self.inc(x)

        x2_cnn = self.down1_cnn(x1)
        x2_trans = self.down1_trans(x1)
        x2 = self.stadge1(torch.cat([x2_cnn, x2_trans], dim=1))

        x3_cnn = self.down2_cnn(x2)
        x3_trans = self.down2_trans(x2_trans)
        x3 = self.stadge2(torch.cat([x3_cnn, x3_trans], dim=1))

        x4_cnn = self.down3_cnn(x3)
        x4_trans = self.down3_trans(x3_trans)
        x4 = self.stadge3(torch.cat([x4_cnn, x4_trans], dim=1))

        x5_cnn = self.down4_cnn(x4)
        x5_trans = self.down4_trans(x4_trans)
        x5 = self.stadge4(torch.cat([x5_cnn, x5_trans], dim=1))

        x6_cnn = self.down5_cnn(x5)
        x6_trans = self.down5_trans(x5_trans)
        x6 = self.stadge5(torch.cat([x6_cnn, x6_trans], dim=1))
        #####################################################

        # Multi-scale fusion part
        # map_list = [x2, x3, x4, x5, x6]
        map_list = [x3, x4, x5, x6]
        map_list = self.map_fusion(map_list)

        # x6_f = torch.cat([x6, map_list[3]], dim=1)
        # x6_f = self.skip_fusion[3](x6_f)
        # x5_f = torch.cat([x5, map_list[2]], dim=1)
        # x5_f = self.skip_fusion[2](x5_f)
        # x4_f = torch.cat([x4, map_list[1]], dim=1)
        # x4_f = self.skip_fusion[1](x4_f)
        # x3_f = torch.cat([x3, map_list[0]], dim=1)
        # x3_f = self.skip_fusion[0](x3_f)
        # x2_f = torch.cat([x2, map_list[0]], dim=1)
        # x2_f = self.skip_fusion[0](x2_f)

        if self.aux_loss:
            out = self.up0(x6, x5)
            out = self.up1(out, x4)
            # out = self.up0(x6_f, x5_f)
            # out = self.up1(out, x4)               #For added level
            # out = self.up1(out, x4_f)  # For added level
            out1 = F.interpolate(self.out1(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            # out = self.up2(out, x3)
            out = self.up2(out, x3)
            out2 = F.interpolate(self.out2(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            # out = self.up3(out, x2)
            out = self.up3(out, x2)
            out3 = F.interpolate(self.out3(out), size=x.shape[-2:], mode='bilinear', align_corners=True)

            out = self.up4(out, x1)
            out = self.outc(out)

            return out, out3, out2, out1

        else:

            out = self.up0(x6, x5)
            skip0 = self.up_skip0(map_list[-1], map_list[-2])
            #print(skip0.shape)

            out = self.up1(out, x4)  # For added level
            skip1 = self.up_skip1(skip0, map_list[-3])
            #print(skip1.shape)

            out = self.up2(out, x3)
            skip2 = self.up_skip2(skip1, map_list[-4])
            #print(skip2.shape)
            out = self.catconv(torch.cat([out, skip2], dim=1))

            # for multi-scale fusion
            # out = self.up0(x6_f, x5_f)
            # out = self.up1(out, x4_f)  # For added level
            # out = self.up2(out, x3_f)

            out = self.up3(out, x2)
            # out = self.up3(out, map_list[0])
            out = self.up4(out, x1)
            # add to reduce resolution to 258
            # out = self.outadd(out)
            out = self.outc(out)
            # out = out[:, :, 31:289, 31:289]

            return out[:, :, 1:259, 1:259]


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                # nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True)
                # nn.SELU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



class conv_block_noPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_noPadding, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True)
                # nn.SELU(inplace=True)

            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



class conv5_block_noPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv5_block_noPadding, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=5),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=5),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_noPadding1x1(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_noPadding1x1, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch//4, kernel_size=1),
                nn.InstanceNorm2d(in_ch//4),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_ch//4, in_ch//4, kernel_size=3),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(in_ch//4),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_ch//4, in_ch//4, kernel_size=3),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(in_ch//4),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True)
                nn.Conv2d(in_ch//4, out_ch, kernel_size=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_padding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_padding, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu1(self.norm1(self.conv1(x)))
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu2(self.norm2(self.conv2(x)))
        return x



class initialconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(initialconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, modelDim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(down, self).__init__()
        if modelDim == 2:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                # nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch),
                conv_block(in_ch, out_ch, modelDim)
            )
        elif modelDim == 3:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv_block(in_ch, out_ch, modelDim)
            )
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class downNoPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(downNoPadding, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block_noPadding(in_ch, out_ch, modelDim)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class downPadded(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(downPadded, self).__init__()
        if modelDim == 2:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                conv_block_padding(in_ch, out_ch, modelDim)
            )
        elif modelDim == 3:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv_block_padding(in_ch, out_ch, modelDim)
            )
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False, conv5=False):
        super(up, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        # self.conv = conv_block_noPadding1x1(in_ch_1//2 + in_ch_2, out_ch, modelDim)
        if conv5:
            # self.conv = conv5_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
            self.conv = conv5_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)
        else:
            # self.conv = conv_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
            self.conv = conv_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        # if self.modelDim == 2: #2D
        #     x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        # else: #3D
        #     x1 = F.pad(x1, (0, x2.size()[4] - x1.size()[4], 0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class upSparse(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upSparse, self).__init__()

        self.conv1 = nn.Conv2d(in_ch_1, in_ch_1 // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_ch_1 // 4)

        self.deconv2 = nn.ConvTranspose2d(in_ch_1 // 4, in_ch_1 // 4, 4, stride=2, padding=3)
        self.norm2 = nn.BatchNorm2d(in_ch_1 // 4)

        self.conv3 = nn.Conv2d(in_ch_1 // 4, in_ch_2, 1)
        self.norm3 = nn.BatchNorm2d(in_ch_2)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = nonlinearity(self.norm3(self.conv3(nonlinearity(self.norm2(self.deconv2(nonlinearity(self.norm1(self.conv1(x1)))))))))
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x1 = x1 + x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3]

        return x1

class upPadded(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upPadded, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        self.conv = conv_block(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        if self.modelDim == 2: #2D
            x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        else: #3D
            x1 = F.pad(x1, (0, x2.size()[4] - x1.size()[4], 0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        # startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        # startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        # x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class upWithPad(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upWithPad, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        self.conv = conv_block(in_ch_1 + in_ch_2, out_ch, modelDim)
        #self.conv = conv_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        #x = torch.cat([x2, x1], dim=1)
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)

        x = self.conv(x)
        return x


class up_noAttn(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, num_block, modelDim=2, heads=4, dim_head=16, attn_drop=0.,
                 proj_drop=0., reduce_size=8, projection='interp', rel_pos=True):
        super(up_noAttn, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)

        #self.block_add = ResBlock(in_ch_1+in_ch_2, in_ch_1+in_ch_2)

        self.conv = conv_block_noPadding(in_ch_1+in_ch_2,  out_ch, modelDim)

        '''
        # Trans_Decoder
        self.attn_decoder = BasicTransDecoderBlock(in_ch_1, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop,
                                                   proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                                   rel_pos=rel_pos)

        attn_block = BasicTransBlock

        block_list = []

        for i in range(num_block):
            block_list.append(attn_block(in_ch_1+in_ch_2, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop,
                                         reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
        self.blocks = nn.Sequential(*block_list)
        '''

    def forward(self, x1, x2):  # x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        #out = self.attn_decoder(x1, x2)
        startIndexDim2 = (x2.size()[2] - x1.size()[2]) // 2
        startIndexDim3 = (x2.size()[3] - x1.size()[3]) // 2
        x = torch.cat([x2[:, :, startIndexDim2:x1.size()[2] + startIndexDim2, startIndexDim3:x1.size()[3] + startIndexDim3], x1],
            dim=1)
        # x = torch.cat([x2, x1], dim=1)
        #x = self.blocks(x)
        #x = self.block_add(x)
        x = self.conv(x)
        return x


class up_V2(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, pad=False):
        super(up_V2, self).__init__()

        self.pad = pad

        self.up = nn.ConvTranspose2d(in_ch_1, in_ch_2, 2, padding=0, stride=2)

        if pad:
            self.res = ResBlock(in_ch_2, out_ch)
        else:
            self.res = ResBlock_noPadding(in_ch_2, out_ch)

    def forward(self, x1, x2):  # x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        startIndexDim2 = (x2.size()[2] - x1.size()[2]) // 2
        startIndexDim3 = (x2.size()[3] - x1.size()[3]) // 2
        x = x2[:, :, startIndexDim2:x1.size()[2] + startIndexDim2, startIndexDim3:x1.size()[3] + startIndexDim3] + x1
        x = self.res(x)
        return x


class upwithTrans(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, heads, pad=False, attn_drop=0., proj_drop=0., reduce_size=8,
                 projection='interp', rel_pos=True):
        super(upwithTrans, self).__init__()

        self.bn_l = nn.BatchNorm2d(in_ch_1)
        self.bn_h = nn.BatchNorm2d(in_ch_2)

        self.conv_ch = nn.Conv2d(in_ch_1, out_ch, kernel_size=1)

        self.up = nn.ConvTranspose2d(in_ch_1, in_ch_2, 2, padding=0, stride=2)

        self.attn = LinearAttentionDecoder(in_ch_1, out_ch, heads=heads,
                                           dim_head=out_ch // heads, attn_drop=attn_drop, proj_drop=proj_drop,
                                           reduce_size=reduce_size,
                                           projection=projection, rel_pos=rel_pos)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

        self.block = BasicBlock_UTNet(2*out_ch, out_ch, 1)

        if pad:
            self.conv = conv_block_padding(in_ch_2 + in_ch_2, out_ch, 2)
        else:
            # self.conv = conv_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
            self.conv = conv_block_noPadding(in_ch_2 + in_ch_2, out_ch, 2)

    def forward(self, x1, x2):  # x2 provides equal/decreased by 1 axis sizes

        x1_up = self.up(x1)

        startIndexDim2 = (x2.size()[2] - x1_up.size()[2]) // 2
        startIndexDim3 = (x2.size()[3] - x1_up.size()[3]) // 2

        x2 = x2[:, :, startIndexDim2:x1_up.size()[2] + startIndexDim2, startIndexDim3:x1_up.size()[3] + startIndexDim3]

        #residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        residue = x1_up


        #x1 = self.bn_l(x1)
        #x2 = self.bn_h(x2)

        out, q_k_attn = self.attn(x2, x1)


        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        out = torch.cat([out, x2], dim=1)

        out = self.conv(out)

        return out


'''
class up_noAttn(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, num_block, modelDim=2, heads=4, dim_head=16, attn_drop=0.,
                 proj_drop=0., reduce_size=8, projection='interp', rel_pos=True):
        super(up_noAttn, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)

        self.attn_decoder = BasicTransDecoderBlock(in_ch_1, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop,
                                                   proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                                   rel_pos=rel_pos)

        block = BasicBlock
        attn_block = BasicTransBlock

        block_list = []

        #for i in range(num_block):
            #block_list.append(attn_block(2 * out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop,
                                         #reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))

        block_list.append(conv_block(in_ch_1 + in_ch_2, out_ch, modelDim))

        block_list.append(block(out_ch, out_ch, stride=1))

        self.blocks = nn.Sequential(*block_list)

        self.conv = conv_block_noPadding(out_ch, out_ch, modelDim)

        self.conv_cnn = conv_block(2 * out_ch, out_ch, 2)

    def forward(self, x1, x2, x_cnn):  # x2 provides equal/decreased by 1 axis sizes
        #out = self.attn_decoder(x1, x2)
        x1 = self.up(x1)

        startIndexDim2 = (x2.size()[2] - x1.size()[2]) // 2
        startIndexDim3 = (x2.size()[3] - x1.size()[3]) // 2
        x = torch.cat(
            [x2[:, :, startIndexDim2:x1.size()[2] + startIndexDim2, startIndexDim3:x1.size()[3] + startIndexDim3],
             x1], dim=1)
        #x = torch.cat([x2, x], dim=1)
        x = self.blocks(x)

        x = torch.cat([x, x_cnn[:, :, startIndexDim2:x1.size()[2] + startIndexDim2,
                          startIndexDim3:x1.size()[3] + startIndexDim3]], dim=1)
        x = self.conv_cnn(x)

        x = self.conv(x)

        return x
'''


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(outconv, self).__init__()
        if modelDim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        elif modelDim == 3:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



class DACblock(nn.Module):
    def __init__(self, channel, modelDim, withAtrous=True):
        super(DACblock, self).__init__()
        if modelDim == 2:
            if withAtrous:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
            else:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        elif modelDim == 3:
            if withAtrous:
                self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
                self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
                self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
            else:
                self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out









