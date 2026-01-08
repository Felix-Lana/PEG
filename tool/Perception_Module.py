# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Wavelet-自己修改-深度可分离-小波变换
"""

"""
使用了：
（1）：空间和频域感知单元(Gabor 变换和小波变换的集成)：     eg：SFS_Conv(in_channel, out_channel, filter="'FrFT' / 'FrGT' / 'Wavelet'")，但是通常使用 'FrGT' 和'Wavelet'
空间和频域感知单元：Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection
（2）：基于自注意力的通道和空间注意力：    自注意力的通道和空间注意力：UNETR++: Delving Into Efficient and Accurate 3D Medical image Segmentation
（3）：曲率注意力
（4）：深度可分离卷积
（5）：新添加了空间注意力
（6）：旋转位置编码：RoFormer:Enhanced 
"""
"""
改进版，降低了显存
"""

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    轻量 depthwise separable 卷积：DW(ks) + PW(1x1) + BN + SiLU
    形状保持：padding = ks//2
    """
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
    """
    简化版 CBAM Spatial Attention：沿通道做 avg/max → concat → 3x3 → sigmoid 加权
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CurvatureAttention(nn.Module):
    """
    简洁“曲率/二阶”引导模块：
    Sobel 一阶 + Laplacian 二作为引导特征，融合后投影回通道数。
    ✅ 修复：out_proj 使用 LazyConv2d 并在 __init__ 注册，避免 unexpected keys
    """
    def __init__(self, guidance="feature", multiscale=True, out_channels=None):
        super().__init__()
        self.guidance = guidance
        self.multiscale = multiscale
        assert out_channels is not None, "CurvatureAttention 需要指定 out_channels"
        self.out_channels = int(out_channels)

        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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
        y = self.out_proj(y)
        return y


# =========================
# FrFT 相关
# =========================

class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super(FrFTFilter, self).__init__()
        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x = kernel[0]
        d_y = kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        X, Y = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(-f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p)))
            real_FrFT_filterY[:, :, i] = np.cos(-f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p)))
            real_FrFT_filter[:, :, i] = (real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i])
        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(real_FrFT_filter[:, :, i : i + 1], in_channels, axis=2)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))
        return torch.tensor(g_f_real, dtype=torch.float32)

    def forward(self, x):
        return x * self.weight


class FrFTSingle(nn.Module):
    """
    可学习核 t，经 FrFT 掩膜“门控”，再做 conv2d 等
    """
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t, std=0.02)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3, training=self.training)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    """
    分频多分支（FrFT）
    """
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "FourierFPU: 通道需能被4整除"
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat([self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1)
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


# =========================
# 波段/空间感知
# =========================

class SPU(nn.Module):
    """
    空间感知单元：两路不同感受野 + 空间注意力
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
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


# =========================
# EPA（优化版：不再有 O(N^2) 参数爆炸）
# =========================
#
# 原版 EPA 的参数爆炸来自：
#   E / Fp: nn.Linear(input_size=N, proj_size=P)  ==> 参数 ~ N*P（随 H*W 增长，甚至接近 N^2）
# 且 EPA2D_Dynamic 会按不同 HxW 缓存多个 EPA 子块，进一步导致参数“越跑越大”。
#
# 本优化版的核心改动：
# 1) 用“无参数的 token 维压缩（adaptive_avg_pool1d）”替代 E/Fp 的可学习投影；
# 2) 提供 max_proj 上限，控制空间注意力低秩维度 P 的计算量（但参数量与 N 无关）；
# 3) 提供 sr_ratio，可选在进入注意力前做空间下采样，进一步降低激活显存。
#
# ✅ 保留的基本功能：
# - 共享 Q/K 的 Paired Attention 思路
# - 通道注意力（CA，沿 d 维）
# - 低秩空间注意力（SA，沿 N 维，但低秩为 P）
# - 两分支各压到 C/2 后 concat 回 C

class EPALite(nn.Module):
    """
    Efficient Paired Attention Block（参数安全版）
    输入:  x [B, N, C]
    输出:  y [B, N, C]
    """
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

        # 生成 [Q, K, V_CA, V_SA]
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        # 两个分支各自压缩到 C/2，最后拼接回 C
        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    @staticmethod
    def _pool_tokens_1d(x: torch.Tensor, P: int) -> torch.Tensor:
        """
        无参数 token 维压缩（替代 E/Fp learnable 投影）
        x: [B, H, d, N] -> [B, H, d, P]
        """
        B, H, d, N = x.shape
        # [B*H*d, 1, N] -> pool -> [B*H*d, 1, P] -> [B,H,d,P]
        x_ = x.reshape(B * H * d, 1, N)
        x_ = F.adaptive_avg_pool1d(x_, P)
        return x_.reshape(B, H, d, P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        d = C // H

        # 低秩维度 P（只影响算量，不影响参数量）
        P = max(self.min_proj, int(round(self.proj_ratio * N)))
        P = min(P, self.max_proj)
        P = min(P, N)  # 防御：避免 P > N

        # [B,N,4C] -> [4,B,H,N,d]
        qkvv = self.qkvv(x).reshape(B, N, 4, H, d).permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # 交换最后两维，便于做通道维注意力（在 d 维上做）
        q_shared = q_shared.transpose(-2, -1)  # [B,H,d,N]
        k_shared = k_shared.transpose(-2, -1)  # [B,H,d,N]
        v_CA = v_CA.transpose(-2, -1)          # [B,H,d,N]
        v_SA = v_SA.transpose(-2, -1)          # [B,H,d,N]

        # 归一化 Q/K（cosine 相似度风格）
        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        # ------- 通道注意力 CA：在 d 维做 self-attn -------
        # [B,H,d,N] @ [B,H,N,d] -> [B,H,d,d]
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        # [B,H,d,d] @ [B,H,d,N] -> [B,H,d,N] -> [B,N,C]
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        # ------- 空间注意力 SA：在 N 维做，低秩（P） -------
        # 用无参数 pool 压缩 token 维（替代 learnable 的 E/Fp）
        k_proj = self._pool_tokens_1d(k_shared, P)   # [B,H,d,P]
        v_proj = self._pool_tokens_1d(v_SA, P)       # [B,H,d,P]

        # [B,H,N,d] @ [B,H,d,P] -> [B,H,N,P]
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_proj) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        # [B,H,N,P] @ [B,H,P,d] -> [B,H,N,d] -> [B,N,C]
        x_SA = (attn_SA @ v_proj.transpose(-2, -1)).permute(0, 2, 1, 3).reshape(B, N, C)

        # 融合：各压到 C/2 再拼接
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        y = torch.cat((x_SA, x_CA), dim=-1)  # [B,N,C]
        return y

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class EPA2D_Lite(nn.Module):
    """
    2D 适配器（参数安全版）：
    - 不再按 HxW 缓存子块（避免“越跑越大”）
    - 可选 sr_ratio 在进入注意力前做空间下采样，降低激活显存
    输入/输出: [B,C,H,W]
    """
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

        # 可选空间下采样（仅影响算量/激活显存，不影响参数）
        if self.sr_ratio > 1 and (H >= self.sr_ratio and W >= self.sr_ratio):
            xp = F.avg_pool2d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            Hp, Wp = xp.shape[-2:]
        else:
            xp = x
            Hp, Wp = H, W

        x_seq = xp.flatten(2).transpose(1, 2)  # [B,N,C]
        y_seq = self.epa(x_seq)                # [B,N,C]
        y = y_seq.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

        # 若下采样过，插值回原尺寸
        if (Hp, Wp) != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class EPA2D_Dynamic(EPA2D_Lite):
    """
    兼容旧名字：以前的实现会按 HxW 动态创建并缓存 EPA 子块，导致参数暴涨。
    现在 EPA2D_Dynamic 直接等价于参数安全版 EPA2D_Lite。

    额外：为了尽量兼容旧 checkpoint（含 blocks.* 键），这里在加载时会忽略旧格式的 blocks 参数。
    """
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
        # 忽略旧版本的 blocks.*（动态分辨率缓存）参数
        blk_prefix = prefix + "blocks."
        to_delete = [k for k in list(state_dict.keys()) if k.startswith(blk_prefix)]
        for k in to_delete:
            del state_dict[k]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

# =========================
# Wavelet：Stationary DWT + FPU
# =========================

class DWT2DStationary(nn.Module):
    """
    不下采样 2D DWT（Haar），输出 [B, 4*C, H, W]
    """
    def __init__(self, wave='haar'):
        super().__init__()
        assert wave.lower() == 'haar', "示例使用 Haar；可按需替换为其它正交滤波器"
        h = torch.tensor([1/math.sqrt(2),  1/math.sqrt(2)], dtype=torch.float32)
        g = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)], dtype=torch.float32)
        ll = torch.outer(h, h); lh = torch.outer(h, g); hl = torch.outer(g, h); hh = torch.outer(g, g)
        bank = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer('bank', bank)
        self.pad2d = (0, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        c = int(c)
        weight = self.bank.repeat_interleave(c, dim=0)  # [4*c,1,2,2]
        x_pad = F.pad(x, self.pad2d, mode='reflect')
        y = F.conv2d(x_pad, weight, stride=1, padding=0, groups=c)  # [B,4C,H,W]
        return y


class WaveletFPU(nn.Module):
    """
    Wavelet 频域单元：Stationary DWT → depthwise 3x3 → BN/ReLU → 1x1 → Dropout → 残差（可选）
    """
    def __init__(self, in_channels, out_channels, wave='haar', drop=0.1):
        super().__init__()
        self.dwt = DWT2DStationary(wave=wave)
        mid_ch = 4 * in_channels
        self.dw = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False)
        self.bn = nn.BatchNorm2d(mid_ch)
        self.act = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(mid_ch, out_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(drop)
        self.use_skip = (in_channels == out_channels)

    def forward(self, x):
        y = self.dwt(x)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.drop(y)
        if self.use_skip:
            y = y + x
        return y


# =========================
# Gabor：滤波器组 + FPU
# =========================

def make_gabor_kernel(ks: int, sigma: float, theta: float, lambd: float, gamma: float = 0.5, psi: float = 0.0):
    half = ks // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2)) * np.cos(2 * np.pi * x_theta / lambd + psi)
    return torch.from_numpy(gb.astype(np.float32))


class FixedGaborBank2d(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 7,
                 n_orient: int = 4, lambdas=(3.0, 5.0),
                 sigma: float = 2.0, gamma: float = 0.5, psi: float = 0.0):
        super().__init__()
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)

        thetas = [i * np.pi / n_orient for i in range(n_orient)]
        kernels = []
        for th in thetas:
            for lb in lambdas:
                k = make_gabor_kernel(self.kernel_size, sigma=sigma, theta=th, lambd=lb, gamma=gamma, psi=psi)
                kernels.append(k)

        bank = torch.stack(kernels, dim=0)             # [K, ks, ks]
        weight = bank.unsqueeze(1)                      # [K, 1, ks, ks]
        weight = weight.repeat_interleave(self.in_channels, dim=0)  # [K*in, 1, ks, ks]
        self.out_channels = weight.shape[0]
        self.register_buffer("weight", weight)
        self.padding = self.kernel_size // 2

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        c = int(c)
        y = F.conv2d(x, self.weight, padding=self.padding, groups=c)  # [B, K*C, H, W]
        return y


class GaborFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25,
                 kernel_size=7, n_orient=4, lambdas=(3.0, 5.0), sigma=2.0, gamma=0.5):
        super().__init__()
        sigma_eff = sigma * (1.0 + float(order))
        self.bank = FixedGaborBank2d(in_channels, kernel_size, n_orient, lambdas, sigma_eff, gamma, psi=0.0)
        mid_ch = self.bank.out_channels
        self.bn = nn.BatchNorm2d(mid_ch)
        self.act = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(mid_ch, out_channels, 1, bias=False)
        self.use_skip = (in_channels == out_channels)

    def forward(self, x):
        y = self.bank(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw(y)
        if self.use_skip:
            y = y + x
        return y


# =========================
# SFS_Conv（支持 Wavelet / FrGT / FrFT）——总是注册 epa_dyn
# =========================

"""
使用了SPU / FPU / CSU 是用来构成 SFS-Conv
1）SPU：空间感知单元
2）FPU：频率感知单元
3）改进版CSU：通道选择单元：有一段“三路融合 + 通道加权”的逻辑，实现将SPU、FPU、曲率注意力 这三个分支做通道级别的注意力加权 + 1×1 卷积融合
"""

class SFS_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, filter="FrGT", order=0.25,
                 use_epa: bool = True,
                 epa_heads: int = 2,
                 epa_proj_ratio: float = 1/8,
                 epa_min_proj: int = 8):
        super().__init__()
        assert filter in ("FrFT", "FrGT", "Wavelet"), "filter 必须是 'FrFT' | 'FrGT' | 'Wavelet'"

        # 两个 1x1 将输入分为两路：空间（SPU）与频域（FPU）
        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.PWC1 = Conv(in_channels, in_channels // 2, 1)
        self.SPU  = SPU(in_channels // 2, out_channels)

        if filter == "FrFT":
            self.FPU = FourierFPU(in_channels // 2, out_channels, order)
        elif filter == "FrGT":
            self.FPU = GaborFPU(in_channels // 2, out_channels, order)
        else:
            self.FPU = WaveletFPU(in_channels // 2, out_channels, wave='haar')

        # 曲率引导
        self.Cur = CurvatureAttention(guidance="feature", multiscale=True, out_channels=out_channels)

        # 三路融合
        self.fuse_cat = Conv(out_channels * 3, out_channels, 1)
        self.PWC_o = Conv(out_channels, out_channels, 1)

        # 零参数通道注意力
        self.advavg = nn.AdaptiveAvgPool2d(1)

        # ✅ 总是注册 epa_dyn：这样无论 use_epa True/False，都能匹配 state_dict 中的 epa_dyn.*
        self.epa_dyn = EPA2D_Dynamic(
            channels=out_channels,
            num_heads=epa_heads,
            proj_ratio=epa_proj_ratio,
            min_proj=epa_min_proj,
        )
        self.use_epa = bool(use_epa)
        self.out_channels = out_channels

    def forward(self, x):
        x_spa = self.SPU(self.PWC0(x))      # [B, out_ch, H, W]
        x_fre = self.FPU(self.PWC1(x))      # [B, out_ch, H, W]
        x_Cur = self.Cur(x)                 # [B, out_ch, H, W]

        out = torch.cat([x_spa, x_fre, x_Cur], dim=1)       # [B, 3*out_ch, H, W]
        ch_score = self.advavg(out)
        out = F.softmax(ch_score, dim=1) * out
        out = self.fuse_cat(out)                            # [B, out_ch, H, W]

        if self.use_epa:
            out = out + self.epa_dyn(out)
        return self.PWC_o(out)
