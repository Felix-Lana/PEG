import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding2D(nn.Module):

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim 必须能被 4 整除（2D RoPE 约定）"
        self.head_dim = head_dim
        self.half = head_dim // 2
        self.pairs_per_dir = self.half // 2


        inv_freq = 1.0 / (base ** (torch.arange(self.pairs_per_dir).float() / self.pairs_per_dir))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.reshape(x.shape)

    def _build_sin_cos_1d(self, n_pos: int, device, dtype):

        t = torch.arange(n_pos, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("n,f->nf", t, self.inv_freq)  # [n_pos, pairs_per_dir]
        sin = torch.sin(freqs).to(dtype)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)  # [n_pos, half]
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)  # [n_pos, half]
        return sin, cos  # [n_pos, half], [n_pos, half]

    def apply_2d(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int):

        B, h, H_in, W_in, d = q.shape
        assert H_in == H and W_in == W and d == self.head_dim, \
            f"False: Input=({H_in},{W_in},{d}) exp=({H},{W},{self.head_dim})"

        device = q.device
        dtype = q.dtype

        sin_y, cos_y = self._build_sin_cos_1d(H, device, dtype)  # [H, half]
        sin_x, cos_x = self._build_sin_cos_1d(W, device, dtype)  # [W, half]
        sin_y = sin_y.view(1, 1, H, 1, self.half)
        cos_y = cos_y.view(1, 1, H, 1, self.half)
        sin_x = sin_x.view(1, 1, 1, W, self.half)
        cos_x = cos_x.view(1, 1, 1, W, self.half)

        q_y, q_x = torch.split(q, [self.half, self.half], dim=-1)
        k_y, k_x = torch.split(k, [self.half, self.half], dim=-1)

        q_y = q_y * cos_y + self._rotate_half(q_y) * sin_y
        k_y = k_y * cos_y + self._rotate_half(k_y) * sin_y

        q_x = q_x * cos_x + self._rotate_half(q_x) * sin_x
        k_x = k_x * cos_x + self._rotate_half(k_x) * sin_x

        q = torch.cat([q_y, q_x], dim=-1)
        k = torch.cat([k_y, k_x], dim=-1)

        return q, k


class SelfAttentionUncertainty(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.,
                 use_rope_2d=False, rope_base=10000.0):
        super().__init__()
        assert dim % num_heads == 0, "dim : num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.use_rope_2d = use_rope_2d
        if use_rope_2d:
            assert self.head_dim % 4 == 0, " 2D RoPE: head_dim % 4 == Analysis_Gating"
            self.rope2d = RotaryEmbedding2D(self.head_dim, base=rope_base)

        self.last_uncertainty = None

    def _flatten_4d_to_3d(self, x: torch.Tensor):
        # x: (B, C, H, W) -> (B, N, C) with N = H*W
        B, C, H, W = x.shape
        x3 = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return x3, (B, C, H, W)

    def _restore_3d_to_4d(self, x3: torch.Tensor, shape_4d):
        # x3: (B, N, C) -> (B, C, H, W)
        B, C, H, W = shape_4d
        x4 = x3.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x4

    def forward(self, x: torch.Tensor, H: int = None, W: int = None):

        self.last_uncertainty = None
        orig_4d = False
        if x.dim() == 4:
            # (B, C, H, W) -> (B, N, C)
            orig_4d = True
            x3, shape_4d = self._flatten_4d_to_3d(x)
            B, N, C = x3.shape
            H = shape_4d[2]
            W = shape_4d[3]
        elif x.dim() == 3:
            x3 = x
            B, N, C = x3.shape
            if self.use_rope_2d:
                # 3D 序列输入想用 2D RoPE，必须给 H,W 且 H*W=N
                assert (H is not None) and (W is not None) and (H * W == N), \
                    "RoPE-2D 3D: H*W=N"
        else:
            raise ValueError(f"SelfAttentionUncertainty only 3D or 4D 输入，Now: {x.dim()}D")


        qkv = self.qkv(x3).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, heads, N, head_dim)

        if self.use_rope_2d:
            q_hw = q.view(B, self.num_heads, H, W, self.head_dim)
            k_hw = k.view(B, self.num_heads, H, W, self.head_dim)
            q_hw, k_hw = self.rope2d.apply_2d(q_hw, k_hw, H, W)
            q = q_hw.view(B, self.num_heads, H * W, self.head_dim)
            k = k_hw.view(B, self.num_heads, H * W, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)  # (B, heads, N)
        uncertainty_map = entropy.mean(dim=1)                        # (B, N)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)      # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if orig_4d:
            out = self._restore_3d_to_4d(out, shape_4d)         # (B, C, H, W)
            self.last_uncertainty = uncertainty_map.view(B, H, W)  # (B, H, W)
        else:
            self.last_uncertainty = uncertainty_map             # (B, N)

        return out
