# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Basic Conv (depthwise separable)
# =========================
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


# =========================
# FrFT (FourierFPU) related
# =========================
class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x, d_y = kernel[0], kernel[1]
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
            g_f[:, :, :, i] = np.repeat(real_FrFT_filter[:, :, i:i + 1], in_channels, axis=2)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))
        return torch.tensor(g_f_real, dtype=torch.float32)

    def forward(self, x):
        return x * self.weight


class FrFTSingle(nn.Module):
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


class DWT2DStationary(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        assert wave.lower() == 'haar', "示例使用 Haar；可按需替换为其它正交滤波器"
        h = torch.tensor([1 / math.sqrt(2),  1 / math.sqrt(2)], dtype=torch.float32)
        g = torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)], dtype=torch.float32)
        ll = torch.outer(h, h)
        lh = torch.outer(h, g)
        hl = torch.outer(g, h)
        hh = torch.outer(g, g)
        bank = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer('bank', bank)
        self.pad2d = (0, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        c = int(c)
        weight = self.bank.repeat_interleave(c, dim=0)  # [4*c,1,2,2]
        x_pad = F.pad(x, self.pad2d, mode='reflect')
        y = F.conv2d(x_pad, weight, stride=1, padding=0, groups=c)  # [B,4C,H,W]
        return y


class WaveletFPU(nn.Module):
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
# GaborFPU
# =========================
def make_gabor_kernel(ks: int, sigma: float, theta: float, lambd: float,
                      gamma: float = 0.5, psi: float = 0.0):
    half = ks // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (sigma ** 2)) * \
         np.cos(2 * np.pi * x_theta / lambd + psi)
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

        bank = torch.stack(kernels, dim=0)                       # [K, ks, ks]
        weight = bank.unsqueeze(1)                               # [K, 1, ks, ks]
        weight = weight.repeat_interleave(self.in_channels, dim=0)  # [K*in, 1, ks, ks]
        self.out_channels = weight.shape[0]
        self.register_buffer("weight", weight)
        self.padding = self.kernel_size // 2

    def forward(self, x: torch.Tensor):
        _, c, _, _ = x.shape
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
# EPA (lite, parameter-safe)
# =========================
class EPALite(nn.Module):
    """
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
        assert hidden_size % num_heads == 0, "hidden_size: num_heads"
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

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        k_proj = self._pool_tokens_1d(k_shared, P)   # [B,H,d,P]
        v_proj = self._pool_tokens_1d(v_SA, P)       # [B,H,d,P]

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_proj) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_proj.transpose(-2, -1)).permute(0, 2, 1, 3).reshape(B, N, C)

        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        y = torch.cat((x_SA, x_CA), dim=-1)  # [B,N,C]
        return y


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
# FPU-only Ablation: keep output shape same as original SFS_Conv
# =========================
class SFS_Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter: str = "FrGT",   # "FrFT" | "FrGT" | "Wavelet"
        order: float = 0.25,
        use_epa: bool = True,
        epa_heads: int = 2,
        epa_proj_ratio: float = 1/8,
        epa_min_proj: int = 8,
    ):
        super().__init__()
        assert filter in ("FrFT", "FrGT", "Wavelet"), "filter: 'FrFT' | 'FrGT' | 'Wavelet'"
        assert in_channels % 2 == 0, "False"

        self.PWC1 = Conv(in_channels, in_channels // 2, 1)

        if filter == "FrFT":
            self.FPU = FourierFPU(in_channels // 2, out_channels, order)
        elif filter == "FrGT":
            self.FPU = GaborFPU(in_channels // 2, out_channels, order)
        else:
            self.FPU = WaveletFPU(in_channels // 2, out_channels, wave='haar')

        # 输出 1x1（原版最后也有 PWC_o）
        self.PWC_o = Conv(out_channels, out_channels, 1)

        # 为了 checkpoint 兼容：总是注册 epa_dyn（与原版一致）
        self.epa_dyn = EPA2D_Dynamic(
            channels=out_channels,
            num_heads=epa_heads,
            proj_ratio=epa_proj_ratio,
            min_proj=epa_min_proj,
        )
        self.use_epa = bool(use_epa)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.FPU(self.PWC1(x))  # [B, out_ch, H, W]
        if self.use_epa:
            out = out + self.epa_dyn(out)
        return self.PWC_o(out)


