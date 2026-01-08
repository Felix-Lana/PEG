import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class _Norm2d(nn.Module):
    """Channel-first LayerNorm-like module using GroupNorm under the hood.
    Safer than BatchNorm for small batch sizes and compatible with AMP.
    """
    def __init__(self, num_channels: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__()
        g = min(num_groups, 32)
        while num_channels % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class LinearGatedMHSA2d(nn.Module):
    """
    Linear (O(N)) multi-head self-attention for 2D feature maps with head-wise sigmoid gating
    applied AFTER attention output (paper-style gated attention idea, but using linear attention).

    Input/Output: [B, C, H, W]

    Core idea:
      Replace softmax(QK^T)V (O(N^2)) with kernel feature map attention (O(N)):
        phi = elu(x) + 1  (positive feature map)
        out_mask = (phi(Q) @ (sum_n phi(K_n)^T V_n)) / (phi(Q) · sum_n phi(K_n) + eps)

    References (conceptually):
      - Linear Transformers / feature-map attention reduces O(N^2) to O(N).  :contentReference[oaicite:2]{index=2}
      - Performer (FAVOR+) is a principled softmax-kernel approximation with linear complexity. :contentReference[oaicite:3]{index=3}
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,   # optional spatial reduction to further save memory
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        assert sr_ratio >= 1
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.eps = eps

        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # head-wise gate from tokens: [B, N, C] -> [B, N, heads]
        self.gate = nn.Linear(dim, num_heads, bias=True)

        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        # Positive feature map (common linear-attention choice)
        return F.elu(x, inplace=False) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # optional spatial reduction (still linear, but smaller N)
        if self.sr_ratio > 1 and (H >= self.sr_ratio and W >= self.sr_ratio):
            xp = F.avg_pool2d(x, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            Hp, Wp = xp.shape[-2:]
        else:
            xp = x
            Hp, Wp = H, W

        N = Hp * Wp

        # [B, C, Hp, Wp] -> [B, N, C]
        xt = xp.flatten(2).transpose(1, 2).contiguous()
        xt = self.norm(xt)

        # [B, N, 3C] -> q,k,v each [B, N, C]
        qkv = self.qkv(xt)
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, N, C] -> [B, heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # feature map
        q_phi = self._phi(q)  # [B,h,N,d]
        k_phi = self._phi(k)  # [B,h,N,d]

        # Compute KV = sum_n k_phi[n]^T v[n]  => [B,h,d,d]
        # einsum: (B,h,N,d) x (B,h,N,d) -> (B,h,d,d)
        kv = torch.einsum("b h n d, b h n e -> b h d e", k_phi, v)

        # Normalizer: z = 1 / (q_phi · sum_n k_phi + eps)  => [B,h,N,1]
        k_sum = k_phi.sum(dim=2)  # [B,h,d]
        z = 1.0 / (torch.einsum("b h n d, b h d -> b h n", q_phi, k_sum) + self.eps)
        z = z.unsqueeze(-1)  # [B,h,N,1]

        # Attention output: out_mask = (q_phi @ kv) * z  => [B,h,N,d]
        out = torch.einsum("b h n d, b h d e -> b h n e", q_phi, kv)
        out = out * z

        # ----- head-wise sigmoid gating AFTER attention output -----
        gate = torch.sigmoid(self.gate(xt))              # [B, N, heads]
        gate = gate.transpose(1, 2).unsqueeze(-1)        # [B, heads, N, 1]
        out = out * gate                                  # [B, heads, N, head_dim]

        # merge heads -> [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj_drop(self.proj(out))

        # back to [B, C, Hp, Wp]
        out = out.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

        # upsample if pooled
        if (Hp, Wp) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        # residual
        return x + out


class ChannelGateGuided(nn.Module):
    """Guided Channel Attention (SE/CBAM style) for skip gating.

    Uses both encoder (x) and decoder guide (g) to produce a channel _mask m_c in [Analysis_Gating,1].
    m_c has shape [B, Cx, 1, 1]. Lightweight and AMP-friendly.

    NEW:
    - Adds linear O(N) self-attention (with gating) before pooling, so channel selection
      is informed by global self-attention context without N^2 memory.
    """
    def __init__(
        self,
        Cx: int,
        Cg: int,
        reduction: int = 16,
        bias: bool = True,
        use_gated_sa: bool = True,
        sa_heads: int = 4,
        sa_sr_ratio: int = 1,
        sa_proj_drop: float = 0.0,
    ):
        super().__init__()
        d = max(Cx // reduction, 1)

        self.proj_g = nn.Conv2d(Cg, Cx, kernel_size=1, bias=False)

        self.use_gated_sa = use_gated_sa
        self.sa = (
            LinearGatedMHSA2d(
                dim=Cx,
                num_heads=sa_heads,
                proj_drop=sa_proj_drop,
                sr_ratio=sa_sr_ratio,
            )
            if use_gated_sa
            else None
        )

        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(Cx, d, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(d, Cx, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        g = self.proj_g(g)

        # fuse encoder/decoder signals
        f = x + g

        # NEW: linear gated self-attention context step (O(N))
        if self.sa is not None:
            f = self.sa(f)

        # global descriptors
        z = self.pool_avg(f) + self.pool_max(f)
        m = self.sigmoid(self.fc2(self.act(self.fc1(z))))  # [B, Cx, 1, 1]
        return m


class SpatialGateGuided(nn.Module):
    """Guided Spatial Attention for skip gating (Attention U-Net inspired, CBAM-refined).

    Produces a spatial _mask m_s in [Analysis_Gating,1] with shape [B, 1, H, W]. Includes a small
    multi-branch refinement (3x3 + dilated 5x5) for robustness.
    """
    def __init__(self, Cx: int, Cg: int, inter_ratio: int = 2, use_depthwise: bool = True):
        super().__init__()
        Ci = max(Cx // inter_ratio, 1)
        self.theta_x = nn.Conv2d(Cx, Ci, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(Cg, Ci, kernel_size=1, bias=False)

        if use_depthwise:
            self.refine1 = nn.Sequential(
                nn.Conv2d(Ci, Ci, kernel_size=3, padding=1, groups=Ci, bias=False),
                nn.Conv2d(Ci, Ci, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.refine2 = nn.Sequential(
                nn.Conv2d(Ci, Ci, kernel_size=3, padding=2, dilation=2, groups=Ci, bias=False),
                nn.Conv2d(Ci, Ci, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
        else:
            self.refine1 = nn.Sequential(
                nn.Conv2d(Ci, Ci, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.refine2 = nn.Sequential(
                nn.Conv2d(Ci, Ci, kernel_size=5, padding=2, bias=False),
                nn.ReLU(inplace=True),
            )

        self.combine = nn.Conv2d(Ci * 2, 1, kernel_size=1, bias=True)
        self.norm_x = _Norm2d(Cx)
        self.norm_g = _Norm2d(Cg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_n = self.norm_x(x)
        g_n = self.norm_g(g)
        q = self.theta_x(x_n)
        k = self.phi_g(g_n)
        s1 = self.refine1(q + k)
        s2 = self.refine2(q + k)
        s = torch.cat([s1, s2], dim=1)
        m = self.sigmoid(self.combine(s))  # [B,1,H,W]
        return m


class GuidedAttentionGate(nn.Module):
    """
    UNet 3+ decoder fusion gate:
    - Input: List[Tensor] length 5, each [B, Cx, H, W] (same spatial size)
    - Output: cat([x_i * gate], dim=1) -> [B, 5*Cx, H, W]
    - If return_mask=True: returns (out_mask, _mask)

    NEW:
    - Channel gating includes linear O(N) self-attention step + head-wise gated output.
    - Average _ablation_masks across 5 inputs (prevents amplification).
    - gate_strength (gamma) is used as soft residual gating strength.
    """
    def __init__(
        self,
        Cx: int,
        Cg: int,
        reduction: int = 16,
        inter_ratio: int = 2,
        mode: str = "cbam",
        gate_strength: float = 1.0,
        return_mask: bool = False,
        # linear self-attention options
        use_gated_sa: bool = True,
        sa_heads: int = 4,
        sa_sr_ratio: int = 1,
        sa_proj_drop: float = 0.0,
    ):
        super().__init__()
        assert mode in {"cbam", "channel", "spatial"}
        self.mode = mode
        self.return_mask = return_mask

        if mode in {"cbam", "channel"}:
            self.cgate = ChannelGateGuided(
                Cx=Cx,
                Cg=Cg,
                reduction=reduction,
                use_gated_sa=use_gated_sa,
                sa_heads=sa_heads,
                sa_sr_ratio=sa_sr_ratio,
                sa_proj_drop=sa_proj_drop,
            )
        else:
            self.cgate = None

        if mode in {"cbam", "spatial"}:
            self.sgate = SpatialGateGuided(Cx, Cg, inter_ratio=inter_ratio)
        else:
            self.sgate = None

        if mode == "cbam":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_parameter("alpha", None)

        # gate strength (soft residual)
        if 0 < gate_strength < 1:
            g0 = torch.log(torch.tensor(gate_strength) / (1 - torch.tensor(gate_strength) + 1e-6))
        else:
            g0 = torch.tensor(6.0) if gate_strength >= 1 else torch.tensor(-6.0)
        self.gamma = nn.Parameter(g0.float())

    def _combine_masks(
        self,
        mc: Optional[torch.Tensor],
        ms: Optional[torch.Tensor],
        alpha: Optional[nn.Parameter],
    ) -> torch.Tensor:
        if mc is None:
            return ms
        if ms is None:
            return mc
        a = torch.sigmoid(alpha)
        return a * mc + (1 - a) * ms  # broadcast to [B,C,H,W] as needed

    def forward(self, x: List[torch.Tensor]):
        assert len(x) == 5, "Expecting 5 input features for gating"

        mc_sum = None
        ms_sum = None

        for xi in x:
            mc_feature = self.cgate(xi, xi) if self.cgate else None
            ms_feature = self.sgate(xi, xi) if self.sgate else None

            if mc_feature is not None:
                mc_sum = mc_feature if mc_sum is None else (mc_sum + mc_feature)
            if ms_feature is not None:
                ms_sum = ms_feature if ms_sum is None else (ms_sum + ms_feature)

        n = len(x)
        mc = (mc_sum / n) if mc_sum is not None else None
        ms = (ms_sum / n) if ms_sum is not None else None

        m = self._combine_masks(mc, ms, self.alpha)

        # soft residual gating strength
        s = torch.sigmoid(self.gamma)
        gated_features = [xi * ((1 - s) + s * m) for xi in x]

        out = torch.cat(gated_features, dim=1)
        return out if not self.return_mask else (out, m)
