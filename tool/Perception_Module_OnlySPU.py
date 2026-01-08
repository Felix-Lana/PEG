# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, groups=1):
        super().__init__()
        p = ks // 2
        # depthwise
        self.dw = nn.Conv2d(in_channels, in_channels, ks, stride, p, groups=in_channels, bias=False)
        # pointwise
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SpatialAttentionUnit(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class SPU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels % 2 == 0, f"SPU 需要偶数通道以便二分；got in_channels={in_channels}"
        self.c1 = Conv(in_channels // 2, in_channels // 2, 3)
        self.c2 = Conv(in_channels // 2, in_channels // 2, 5)
        self.c3 = Conv(in_channels, out_channels, 1)
        self.spatial_attention = SpatialAttentionUnit(kernel_size=3)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.c3(x_out)
        x_out = self.spatial_attention(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out



class EPALite(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        channel_attn_drop: float = 0.1,
        spatial_attn_drop: float = 0.1,
        proj_ratio: float = 1/16,
        min_proj: int = 4,
        max_proj: int = 256,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)

        self.proj_ratio = float(proj_ratio)
        self.min_proj = int(min_proj)
        self.max_proj = int(max_proj)

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    @staticmethod
    def _pool_tokens_1d(x: torch.Tensor, P: int) -> torch.Tensor:

        B, H, d, N = x.shape
        x_ = x.reshape(B * H * d, 1, N)
        x_ = F.adaptive_avg_pool1d(x_, P)
        return x_.reshape(B, H, d, P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        d = C // H

        P = max(self.min_proj, int(round(self.proj_ratio * N)))
        P = min(P, self.max_proj)
        P = min(P, N)

        qkvv = self.qkvv(x).reshape(B, N, 4, H, d).permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)  # [B,H,d,N]
        k_shared = k_shared.transpose(-2, -1)  # [B,H,d,N]
        v_CA = v_CA.transpose(-2, -1)          # [B,H,d,N]
        v_SA = v_SA.transpose(-2, -1)          # [B,H,d,N]

        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        # Channel Attn
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # Spatial Attn (low-rank)
        k_proj = self._pool_tokens_1d(k_shared, P)   # [B,H,d,P]
        v_proj = self._pool_tokens_1d(v_SA, P)       # [B,H,d,P]

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_proj) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_proj.transpose(-2, -1)).permute(0, 2, 1, 3).reshape(B, N, C)

        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        y = torch.cat((x_SA, x_CA), dim=-1)
        return y

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class EPA2D_Lite(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        proj_ratio: float = 1/16,
        min_proj: int = 4,
        max_proj: int = 256,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        spatial_drop: float = 0.1,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.sr_ratio = int(sr_ratio)

        self.epa = EPALite(
            hidden_size=self.channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            channel_attn_drop=attn_drop,
            spatial_attn_drop=spatial_drop,
            proj_ratio=proj_ratio,
            min_proj=min_proj,
            max_proj=max_proj,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"channels mismatch: {C} vs {self.channels}"

        if self.sr_ratio > 1 and (H >= self.sr_ratio and W >= self.sr_ratio):
            xp = F.avg_pool2d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            Hp, Wp = xp.shape[-2:]
        else:
            xp = x
            Hp, Wp = H, W

        x_seq = xp.flatten(2).transpose(1, 2)  # [B,N,C]
        y_seq = self.epa(x_seq)                # [B,N,C]
        y = y_seq.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

        if (Hp, Wp) != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class EPA2D_Dynamic(EPA2D_Lite):
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
    ):
        blk_prefix = prefix + "blocks."
        to_delete = [k for k in list(state_dict.keys()) if k.startswith(blk_prefix)]
        for k in to_delete:
            del state_dict[k]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


# =========================
# SFS_Conv（SPU-only ablation）
# =========================

class SFS_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter="FrGT",
        order=0.25,
        use_epa: bool = True,
        epa_heads: int = 2,
        epa_proj_ratio: float = 1/8,
        epa_min_proj: int = 8,
    ):
        super().__init__()

        assert in_channels % 4 == 0, (
            f"SFS_Conv(SPU-only): in_channels % 4 == 0；got {in_channels}"
        )

        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.SPU  = SPU(in_channels // 2, out_channels)

        self.epa_dyn = EPA2D_Dynamic(
            channels=out_channels,
            num_heads=epa_heads,
            proj_ratio=epa_proj_ratio,
            min_proj=epa_min_proj,
        )
        self.use_epa = bool(use_epa)

        self.PWC_o = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        out = self.SPU(self.PWC0(x))  # [B, out_ch, H, W]
        if self.use_epa:
            out = out + self.epa_dyn(out)
        return self.PWC_o(out)
