# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, stride=1, groups=1):
        super().__init__()
        p = ks // 2
        self.dw = nn.Conv2d(in_channels, in_channels, ks, stride, p, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CurvatureAttention(nn.Module):

    def __init__(self, guidance="feature", multiscale=True, out_channels=None):
        super().__init__()
        self.guidance = guidance
        self.multiscale = multiscale
        assert out_channels is not None, "CurvatureAttention out_channels"
        self.out_channels = int(out_channels)

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        lap = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)

        self.fuse = nn.Conv2d(3, 1, 1, bias=False)
        self.out_proj = nn.LazyConv2d(self.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        lap = F.conv2d(x_gray, self.lap, padding=1)

        guide = torch.cat([gx.abs(), gy.abs(), lap.abs()], dim=1)  # [B,3,H,W]
        guide_w = torch.sigmoid(self.fuse(guide))                  # [B,1,H,W]
        y = x * (1.0 + guide_w)
        y = self.out_proj(y)                                       # [B,out_ch,H,W]
        return y




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
        assert hidden_size % num_heads == 0, "hidden_size num_heads"
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)

        self.proj_ratio = float(proj_ratio)
        self.min_proj = int(min_proj)
        self.max_proj = int(max_proj)

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        # 生成 [Q, K, V_CA, V_SA]
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        # 两个分支各自压缩到 C/2，最后拼接回 C
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

        # [B,N,4C] -> [4,B,H,N,d]
        qkvv = self.qkvv(x).reshape(B, N, 4, H, d).permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # [B,H,N,d] -> [B,H,d,N]
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        # 通道注意力（在 d 维）
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature  # [B,H,d,d]
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)          # [B,N,C]

        # 空间注意力（在 N 维，低秩 P）
        k_proj = self._pool_tokens_1d(k_shared, P)                            # [B,H,d,P]
        v_proj = self._pool_tokens_1d(v_SA, P)                                # [B,H,d,P]
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_proj) * self.temperature2  # [B,H,N,P]
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_proj.transpose(-2, -1)).permute(0, 2, 1, 3).reshape(B, N, C)

        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        y = torch.cat((x_SA, x_CA), dim=-1)  # [B,N,C]
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



class SFS_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, filter="FrGT", order=0.25,
                 use_epa: bool = True,
                 epa_heads: int = 2,
                 epa_proj_ratio: float = 1/8,
                 epa_min_proj: int = 8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # CPU：曲率引导分支（直接把输入投影到 out_channels）
        self.Cur = CurvatureAttention(guidance="feature", multiscale=True, out_channels=self.out_channels)

        # 末端保持一致：可选 EPA + 1x1 输出投影
        self.epa_dyn = EPA2D_Dynamic(
            channels=self.out_channels,
            num_heads=epa_heads,
            proj_ratio=epa_proj_ratio,
            min_proj=epa_min_proj,
        )
        self.use_epa = bool(use_epa)
        self.PWC_o = Conv(self.out_channels, self.out_channels, 1)

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

        drop_prefixes = (
            prefix + "PWC0.",
            prefix + "PWC1.",
            prefix + "SPU.",
            prefix + "FPU.",
            prefix + "fuse_cat.",
            prefix + "advavg.",
        )
        to_delete = [k for k in list(state_dict.keys()) if k.startswith(drop_prefixes)]
        for k in to_delete:
            del state_dict[k]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        out = self.Cur(x)  # [B, out_ch, H, W]
        if self.use_epa:
            out = out + self.epa_dyn(out)
        return self.PWC_o(out)

