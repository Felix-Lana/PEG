import numpy as np
from scipy import ndimage as ndi
import heapq
from typing import Tuple, List, Optional, Dict

# ----------------------
# Utilities
# ----------------------

def ensure_float01(img: np.ndarray) -> np.ndarray:
    """Return float image scaled to [Analysis_Gating,1]. Accepts uint8/uint16/float.*
    If RGB, convert to simple luminance."""
    arr = img.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] > 1:
        # Normalize if looks like 8-bit
        if arr.max() > 1.5:
            arr /= 255.0
        # Y-like luminance weights
        arr = 0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]
    m, M = float(arr.min()), float(arr.max())
    if M > m:
        arr = (arr - m) / (M - m)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.float32)


def sobel_gradmag(gray: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    g = np.hypot(gx, gy)
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    return g.astype(np.float32)


def disk_footprint(r: int) -> np.ndarray:
    """Create a disk structuring element with radius r."""
    if r < 1:
        return np.ones((1,1), dtype=bool)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= r*r
    return mask


def white_tophat(gray: np.ndarray, r: int) -> np.ndarray:
    """Morphological white top-hat using grey opening with a disk footprint."""
    fp = disk_footprint(r)
    opened = ndi.grey_opening(gray, footprint=fp)
    wth = gray - opened
    wth[wth < 0] = 0
    return wth


def normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a /= m
    return a


def binary_hysteresis(R: np.ndarray, q_high: float, q_low: float) -> np.ndarray:
    """Hysteresis threshold using quantiles on positive responses."""
    pos = R[R > 0]
    if pos.size == 0:
        return np.zeros_like(R, dtype=bool)
    th_hi = np.quantile(pos, q_high / 100.0)
    th_lo = np.quantile(pos, q_low / 100.0)
    B_hi = R >= th_hi
    B_lo = R >= th_lo
    if not B_lo.any():
        return B_hi
    labeled, n = ndi.label(B_lo)
    if n == 0:
        return B_hi
    keep = np.zeros(n + 1, dtype=bool)
    # keep components that intersect B_hi
    comp_ids = np.unique(labeled[B_hi])
    keep[comp_ids] = True
    return keep[labeled]


def dilate_ring(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.zeros_like(mask, dtype=bool)
    fp = disk_footprint(r)
    dil = ndi.binary_dilation(mask.astype(bool), structure=fp)
    ring = np.logical_and(dil, ~mask.astype(bool))
    return ring


def allowed_background_band(coarse_fg: np.ndarray, r_allow: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return allowed background band and deep background Net_masks from a coarse foreground _mask."""
    fg = coarse_fg.astype(bool)
    fp = disk_footprint(max(1, r_allow))
    dil = ndi.binary_dilation(fg, structure=fp)
    allow_band = np.logical_and(dil, ~fg)
    deep_bg = ~np.logical_or(dil, fg)
    return allow_band, deep_bg


# ----------------------
# Region-level salient highlight (SHR, Sal)
# ----------------------

def detect_region_saliency(
    gray: np.ndarray,
    scales: Tuple[int, ...] = (3, 5, 7, 9),
    q_high: float = 99.0,
    q_low: float = 95.0,
    area_min_ratio: float = 0.0002,   # Analysis_Gating.02% of image area
    contrast_ring_r: int = 3,
    contrast_min: float = 0.12,
    sal_alpha: float = 0.7,
    sal_beta: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build region-level bright-saliency:
    - R: multi-scale white top-hat (max over scales)
    - SHR: binary salient highlight regions after hysteresis and region filtering
    - Sal: continuous saliency in [Analysis_Gating,1] combining normalized R and region contrast

    Returns
    -------
    R : float32 (H,W)   white-tophat max response (normalized to [Analysis_Gating,1])
    SHR : bool (H,W)    strong highlight regions (continuous, sizeable, high contrast)
    Sal : float32 (H,W) continuous saliency map in [Analysis_Gating,1]
    """
    H, W = gray.shape
    # Multi-scale white top-hat
    Rs = [white_tophat(gray, r) for r in scales]
    R = np.max(np.stack(Rs, axis=0), axis=0)
    R = normalize01(R)

    # Hysteresis threshold on R
    B = binary_hysteresis(R, q_high=q_high, q_low=q_low)

    # Connected components + filtering by area and outer-ring contrast
    lbl, n = ndi.label(B)
    SHR = np.zeros_like(B, dtype=bool)
    Sal = np.zeros_like(gray, dtype=np.float32)

    A_min = max(1, int(area_min_ratio * H * W))
    fp_ring = disk_footprint(max(1, contrast_ring_r))

    for i in range(1, n+1):
        comp = (lbl == i)
        area = comp.sum()
        if area < A_min:
            continue
        # Outer ring
        ring = np.logical_and(ndi.binary_dilation(comp, structure=fp_ring), ~comp)
        if ring.sum() == 0:
            continue
        mean_in = float(gray[comp].mean())
        mean_out = float(gray[ring].mean())
        delta = mean_in - mean_out
        if delta < contrast_min:
            continue
        # Keep component
        SHR[comp] = True
        # Fill saliency inside region: blend R and (normalized) contrast
        delta_norm = float(np.clip(delta, 0.0, 1.0))
        Sal[comp] = np.clip(sal_alpha * R[comp] + sal_beta * delta_norm, 0.0, 1.0)

    # Slightly diffuse Sal at region boundary for smooth distances (optional)
    if Sal.any():
        Sal = ndi.gaussian_filter(Sal, sigma=0.5)
        Sal = normalize01(Sal)

    return R.astype(np.float32), SHR.astype(bool), Sal.astype(np.float32)


# ----------------------
# Density map
# ----------------------

def build_density_map(
    gray: np.ndarray,
    grad: np.ndarray,
    Sal: np.ndarray,
    SHR: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    w_g: float = 0.5,
    w_h: float = 1.1,
    w_e: float = 0.8,
    w_u: float = 0.5,
    hl_ring_w: float = 0.7,
    ring_r: int = 3,
    coarse_fg: Optional[np.ndarray] = None,
    gamma_bg: float = 1.3,
    r_allow: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine features into a seed-density map M.
    M = w_g*grad + w_h*Sal + w_e*EdgeBand + w_u*U + hl_ring_w*SHR_ring
    Then apply background suppression (deep background gets stronger suppression).
    Returns (M, allow_bg_band, deep_bg)
    """
    # EdgeBand: emphasize mid-strong gradients (sigmoid compression)
    edge_band = grad.copy()
    edge_band = edge_band**0.8  # gently flatten extremes

    SHR_ring = dilate_ring(SHR, ring_r)

    U = np.zeros_like(gray, dtype=np.float32) if uncertainty is None else normalize01(uncertainty)
    M = (w_g * grad + w_h * Sal + w_e * edge_band + w_u * U + hl_ring_w * SHR_ring.astype(np.float32))
    M = normalize01(M)

    # Background suppression if coarse _mask provided
    if coarse_fg is not None:
        allow_band, deep_bg = allowed_background_band(coarse_fg, r_allow=r_allow)
        P_bg = (~coarse_fg.astype(bool)).astype(np.float32)
        BgSupp = (1 - P_bg) ** gamma_bg  # fg ~1 => keep; bg ~Analysis_Gating => suppress
        M = M * BgSupp
        # Additionally damp deep background density
        M[deep_bg] *= 0.1
    else:
        allow_band = np.zeros_like(gray, dtype=bool)
        deep_bg = np.zeros_like(gray, dtype=bool)

    M = normalize01(M)
    return M.astype(np.float32), allow_band, deep_bg


# ----------------------
# Seed placement
# ----------------------

def sample_from_density(
    density: np.ndarray,
    k: int,
    forbid: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None
) -> List[Tuple[int,int]]:
    """Sample k seeds from a probability map (without replacement)."""
    if rng is None:
        rng = np.random.RandomState(123)
    H, W = density.shape
    D = density.astype(np.float64).copy()

    # 允许区域：如果 forbid 给了，就把禁止区置 Analysis_Gating
    valid = np.ones_like(D, dtype=bool)
    if forbid is not None:
        valid &= ~forbid.astype(bool)

    D[~valid] = 0.0
    D[D < 0] = 0.0

    # 非零概率的索引
    flat = D.ravel()
    nz_idx = np.where(flat > 0)[0]
    nz_cnt = nz_idx.size

    # 如果全部为 Analysis_Gating，退化为在 valid 区域均匀采样
    if nz_cnt == 0:
        valid_flat = valid.ravel()
        valid_idx = np.where(valid_flat)[0]
        if valid_idx.size == 0 or k <= 0:
            return []
        size = min(k, valid_idx.size)
        idx_sel = rng.choice(valid_idx, size=size, replace=False)
        ys, xs = np.unravel_index(idx_sel, (H, W))
        return list(zip(ys.tolist(), xs.tolist()))

    # 这里保证 size 不会超过非零概率的个数（避免报错）
    if k <= 0:
        return []
    size = min(k, nz_cnt)

    # 只在 nz_idx 上归一化概率
    p = np.zeros_like(flat, dtype=np.float64)
    p[nz_idx] = flat[nz_idx]
    p_sum = p[nz_idx].sum()
    if p_sum <= 0:
        # 理论上不会到这里，但以防万一，再退化成均匀
        valid_flat = valid.ravel()
        valid_idx = np.where(valid_flat)[0]
        size = min(k, valid_idx.size)
        idx_sel = rng.choice(valid_idx, size=size, replace=False)
    else:
        p[nz_idx] /= p_sum
        idx_sel = rng.choice(flat.size, size=size, replace=False, p=p)

    ys, xs = np.unravel_index(idx_sel, (H, W))
    return list(zip(ys.tolist(), xs.tolist()))


def allocate_seeds_by_regions(
    Sal: np.ndarray,
    SHR: np.ndarray,
    gray: np.ndarray,
    k_target: int,
    gamma_area: float = 0.8,
    eta_contrast: float = 0.2,
    min_per_region: int = 1,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[int,int]]:
    """Deterministically allocate seeds to SHR regions, proportional to area^gamma * contrast^eta.
    Within each region, sample positions with probability ~ Sal.
    """
    if rng is None:
        rng = np.random.RandomState(123)
    lbl, n = ndi.label(SHR.astype(bool))
    if n == 0 or k_target <= 0:
        return []

    fp_ring = disk_footprint(3)

    # Collect per-region stats
    areas, contrasts = [], []
    comp_ids = list(range(1, n+1))
    for i in comp_ids:
        comp = (lbl == i)
        area = comp.sum()
        ring = np.logical_and(ndi.binary_dilation(comp, structure=fp_ring), ~comp)
        if ring.sum() == 0:
            delta = 0.0
        else:
            delta = float(gray[comp].mean() - gray[ring].mean())
        areas.append(area)
        contrasts.append(max(0.0, delta))

    areas = np.asarray(areas, dtype=np.float64)
    contrasts = np.asarray(contrasts, dtype=np.float64)
    # Normalize for stability
    a_norm = areas / max(1.0, areas.sum())
    c_norm = contrasts / max(1e-6, contrasts.max()) if contrasts.max() > 0 else contrasts
    scores = (a_norm ** gamma_area) * (np.maximum(1e-6, c_norm) ** eta_contrast)
    if scores.sum() <= 0:
        # fallback: uniform allocation
        scores = np.ones_like(scores) / scores.size
    else:
        scores = scores / scores.sum()

    # Allocate integer counts with floor + residual
    alloc = np.floor(k_target * scores).astype(int)
    # Ensure min_per_region if total allows
    if alloc.sum() < len(alloc) * min_per_region and k_target >= len(alloc) * min_per_region:
        alloc[:] = min_per_region
    else:
        for i in range(len(alloc)):
            if alloc[i] < min_per_region and k_target >= len(alloc):
                alloc[i] = min_per_region
    # Adjust to exact total
    diff = k_target - int(alloc.sum())
    if diff != 0:
        # distribute residual by largest fractional parts or randomly
        frac = k_target * scores - np.floor(k_target * scores)
        order = np.argsort(-frac if diff > 0 else frac)  # add to largest frac; remove from smallest
        for j in range(min(abs(diff), len(order))):
            idx = order[j]
            alloc[idx] += 1 if diff > 0 else -1

    # Sample inside each region proportional to Sal
    seeds = []
    for i, cnt in zip(comp_ids, alloc.tolist()):
        if cnt <= 0:
            continue
        comp = (lbl == i)
        S = Sal.copy()
        S[~comp] = 0
        if S.sum() <= 0:
            ys, xs = np.where(comp)
            if ys.size == 0:
                continue
            sel = rng.choice(ys.size, size=min(cnt, ys.size), replace=False)
            seeds += list(zip(ys[sel].tolist(), xs[sel].tolist()))
        else:
            seeds += sample_from_density(S, cnt, forbid=None, rng=rng)
    return seeds


def refine_seeds_with_score(
    seeds: List[Tuple[int,int]],
    grad: np.ndarray,
    edge_band: np.ndarray,
    Sal: np.ndarray,
    SHR_ring: np.ndarray,
    win_r_in_shr: int = 1,
    win_r_else: int = 2,
) -> List[Tuple[int,int]]:
    """Snap each seed to a nearby peak of a combined score: Analysis_Gating.6*grad + Analysis_Gating.8*edge + 1.2*Sal + Analysis_Gating.5*ring"""
    H, W = grad.shape
    score = normalize01(0.6*grad + 0.8*edge_band + 1.2*Sal + 0.5*SHR_ring.astype(np.float32))
    out = []
    for (y, x) in seeds:
        r = win_r_in_shr if Sal[y, x] > 0.5 else win_r_else
        y0, y1 = max(0, y-r), min(H, y+r+1)
        x0, x1 = max(0, x-r), min(W, x+r+1)
        patch = score[y0:y1, x0:x1]
        idx = np.argmax(patch)
        dy, dx = np.unravel_index(idx, patch.shape)
        out.append((y0+dy, x0+dx))
    # Deduplicate
    out = list(dict.fromkeys(out))
    return out


# ----------------------
# SNIC-like segmentation
# ----------------------

class SNICGrower:
    def __init__(
        self,
        gray: np.ndarray,
        Sal: np.ndarray,
        grad: np.ndarray,
        SHR_ring: np.ndarray,
        deep_bg: Optional[np.ndarray] = None,
        compactness: float = 10.0,
        local_compactness_alpha: float = 0.75,
        grad_weight: float = 0.2,
        hl_growth_relief: float = 0.4,
        lambda_sal: float = 0.3,
        connectivity: int = 4,
    ):
        self.gray = gray.astype(np.float32)
        self.Sal = np.clip(Sal.astype(np.float32), 0.0, 1.0)
        self.grad = np.clip(grad.astype(np.float32), 0.0, 1.0)
        self.SHR_ring = SHR_ring.astype(bool) if SHR_ring is not None else np.zeros_like(gray, dtype=bool)
        self.deep_bg = deep_bg.astype(bool) if deep_bg is not None else np.zeros_like(gray, dtype=bool)
        self.compactness = float(compactness)
        self.alpha = float(local_compactness_alpha)
        self.grad_w = float(grad_weight)
        self.relief = float(hl_growth_relief)
        self.lambda_sal = float(lambda_sal)
        self.connectivity = 8 if connectivity == 8 else 4

        self.H, self.W = gray.shape
        self.labels = -np.ones((self.H, self.W), dtype=np.int32)

    def _neighbors(self, y: int, x: int):
        if self.connectivity == 8:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < self.H and 0 <= xx < self.W:
                        yield yy, xx
        else:
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < self.H and 0 <= xx < self.W:
                    yield yy, xx

    def segment(self, seeds: List[Tuple[int,int]]) -> np.ndarray:
        K = len(seeds)
        if K == 0:
            return self.labels  # all -1
        # Approximate S (grid step) for spatial term
        S = float(np.sqrt((self.H*self.W)/max(1, K)))

        # Per-cluster accumulators
        cx = np.array([x for (y,x) in seeds], dtype=np.float64)
        cy = np.array([y for (y,x) in seeds], dtype=np.float64)
        cf = np.array([self.gray[y, x] for (y,x) in seeds], dtype=np.float64)
        csal = np.array([self.Sal[y, x] for (y,x) in seeds], dtype=np.float64)
        count = np.ones(K, dtype=np.int32)

        # Initialize priority queue
        pq = []  # (distance, idx_linear, cluster_id)
        for k, (y, x) in enumerate(seeds):
            self.labels[y, x] = k
            for (yy, xx) in self._neighbors(y, x):
                if self.labels[yy, xx] < 0:
                    d = self._pixel_distance(yy, xx, k, S, cf[k], cx[k], cy[k], csal[k])
                    heapq.heappush(pq, (d, yy*self.W + xx, k))

        # Region growing
        while pq:
            d, idx, k = heapq.heappop(pq)
            y, x = divmod(idx, self.W)
            if self.labels[y, x] >= 0:
                continue
            # Assign
            self.labels[y, x] = k
            # Update cluster stats (incremental mean)
            f = float(self.gray[y, x])
            s = float(self.Sal[y, x])
            count[k] += 1
            w = 1.0 / count[k]
            cf[k] = (1 - w)*cf[k] + w*f
            csal[k] = (1 - w)*csal[k] + w*s
            cx[k] = (1 - w)*cx[k] + w*x
            cy[k] = (1 - w)*cy[k] + w*y

            # Push neighbors
            for (yy, xx) in self._neighbors(y, x):
                if self.labels[yy, xx] < 0:
                    d2 = self._pixel_distance(yy, xx, k, S, cf[k], cx[k], cy[k], csal[k])
                    heapq.heappush(pq, (d2, yy*self.W + xx, k))
        return self.labels

    def _pixel_distance(
        self, y: int, x: int, k: int, S: float,
        cf_k: Optional[float]=None, cx_k: Optional[float]=None, cy_k: Optional[float]=None, csal_k: Optional[float]=None
    ) -> float:
        """Distance between pixel (y,x) and cluster k with adaptive compactness & saliency penalty."""
        f = float(self.gray[y, x])
        # Feature difference (L1 in gray-space)
        df = 0.0 if cf_k is None else abs(f - cf_k)

        # Spatial term with adaptive compactness
        dx = float(x) - (float(cx_k) if cx_k is not None else float(x))
        dy = float(y) - (float(cy_k) if cy_k is not None else float(y))
        local_w = max(float(self.Sal[y, x]), 0.7 if self.SHR_ring[y, x] else 0.0)
        m_eff = self.compactness * max(0.0, 1.0 - self.alpha * local_w)
        ds = (m_eff / (S + 1e-6))**2 * (dx*dx + dy*dy)

        # Edge-stop multiplicative factor; relaxed in SHR ring
        relief = (1.0 - self.relief) if self.SHR_ring[y, x] else 1.0
        gstop = 1.0 + self.grad_w * float(self.grad[y, x]) * relief

        # Deep background barrier
        barrier = 0.0
        if self.deep_bg[y, x]:
            barrier = 1e3  # large penalty to prevent invasion

        # Soft out_mask-of-region penalty: avoid draining high-saliency clusters into low-sal pixels
        sal_p = float(self.Sal[y, x])
        sal_k = float(csal_k if csal_k is not None else self.Sal[y, x])
        sal_pen = self.lambda_sal * max(0.0, sal_k - sal_p)

        return (df + ds + sal_pen) * gstop + barrier


# ----------------------
# Orchestration
# ----------------------

def compute_superpixels_rsg_snic(
    image: np.ndarray,
    coarse_mask: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    # highlight / region-saliency params
    scales: Tuple[int,...] = (3,5,7,9),
    q_high: float = 99.0,
    q_low: float = 95.0,
    area_min_ratio: float = 0.0002,
    contrast_ring_r: int = 3,
    contrast_min: float = 0.12,
    sal_alpha: float = 0.7,
    sal_beta: float = 0.3,
    # density / sampling params
    w_g: float = 0.5, w_h: float = 1.1, w_e: float = 0.8, w_u: float = 0.5,
    hl_ring_w: float = 0.7, ring_r: int = 3,
    gamma_bg: float = 1.3, r_allow: int = 4,
    num_superpixels: int = 1000,
    shr_quota_ratio: float = 0.6,   # fraction of total seeds reserved for SHR regions
    bg_sentinel_per_tile: int = 1,
    density_tile: int = 12,
    # seed refine
    win_r_in_shr: int = 1, win_r_else: int = 2,
    # SNIC params
    compactness: float = 10.0,
    local_compactness_alpha: float = 0.75,
    grad_weight: float = 0.2,
    hl_growth_relief: float = 0.4,
    lambda_sal: float = 0.3,
    connectivity: int = 4,
    random_state: int = 123,
) -> Tuple[np.ndarray, List[Tuple[int,int]], Dict[str, np.ndarray]]:
    """
    Full pipeline. Returns (labels, seeds, debug_dict).
    """
    rng = np.random.RandomState(random_state)

    gray = ensure_float01(image)
    H, W = gray.shape

    # Basic features
    grad = sobel_gradmag(gray)

    # Region-level saliency
    R, SHR, Sal = detect_region_saliency(
        gray, scales=scales, q_high=q_high, q_low=q_low,
        area_min_ratio=area_min_ratio, contrast_ring_r=contrast_ring_r,
        contrast_min=contrast_min, sal_alpha=sal_alpha, sal_beta=sal_beta
    )
    SHR_ring = dilate_ring(SHR, ring_r)

    # Density map
    M, allow_band, deep_bg = build_density_map(
        gray=gray, grad=grad, Sal=Sal, SHR=SHR, uncertainty=uncertainty,
        w_g=w_g, w_h=w_h, w_e=w_e, w_u=w_u, hl_ring_w=hl_ring_w, ring_r=ring_r,
        coarse_fg=coarse_mask, gamma_bg=gamma_bg, r_allow=r_allow
    )
    edge_band = (grad**0.8)  # for seed refinement

    # Seed allocation
    K = int(num_superpixels)
    K_shr = int(max(0, min(1.0, shr_quota_ratio)) * K)
    seeds_shr = allocate_seeds_by_regions(Sal, SHR, gray, K_shr, rng=rng)
    # Remainder from density (forbid deep background)
    K_rem = max(0, K - len(seeds_shr))
    forbid = deep_bg.copy() if deep_bg is not None else None
    seeds_rem = sample_from_density(M, K_rem, forbid=forbid, rng=rng)
    seeds = seeds_shr + seeds_rem

    # Optional: add background sentinel per tile (within allowed band only)
    if coarse_mask is not None and bg_sentinel_per_tile > 0 and density_tile > 0:
        htiles = max(1, H // density_tile)
        wtiles = max(1, W // density_tile)
        for i in range(htiles):
            for j in range(wtiles):
                y0, y1 = i*density_tile, min(H, (i+1)*density_tile)
                x0, x1 = j*density_tile, min(W, (j+1)*density_tile)
                tile = np.s_[y0:y1, x0:x1]
                band = allow_band[tile]
                if band.any():
                    yb, xb = np.where(band)
                    sel = rng.choice(yb.size, size=min(bg_sentinel_per_tile, yb.size), replace=False)
                    for s in sel.tolist():
                        seeds.append((y0 + int(yb[s]), x0 + int(xb[s])))
    # Deduplicate seeds
    seeds = list(dict.fromkeys(seeds))

    # Seed refinement (local snapping)
    seeds = refine_seeds_with_score(
        seeds, grad=grad, edge_band=edge_band, Sal=Sal, SHR_ring=SHR_ring,
        win_r_in_shr=win_r_in_shr, win_r_else=win_r_else
    )

    # SNIC-like segmentation
    grower = SNICGrower(
        gray=gray, Sal=Sal, grad=grad, SHR_ring=SHR_ring, deep_bg=deep_bg,
        compactness=compactness, local_compactness_alpha=local_compactness_alpha,
        grad_weight=grad_weight, hl_growth_relief=hl_growth_relief,
        lambda_sal=lambda_sal, connectivity=connectivity
    )
    labels = grower.segment(seeds)

    debug = dict(
        gray=gray, grad=grad, R=R, Sal=Sal, SHR=SHR.astype(np.uint8),
        SHR_ring=SHR_ring.astype(np.uint8), density=M,
        deep_bg=deep_bg.astype(np.uint8) if deep_bg is not None else np.zeros_like(gray, dtype=np.uint8)
    )
    return labels.astype(np.int32), seeds, debug


# ----------------------
#  给 OCTA 用的简化接口
# ----------------------

def rsg_snic_superpixels_from_octa(
    octa_image: np.ndarray,
    coarse_mask: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    num_superpixels: int = 1500,
    random_state: int = 123,
) -> np.ndarray:
    """
    对 OCTA 图像进行 RSG+SNIC 超像素分割。
    返回 sp_index，shape=(H,W)，标签重编号为 Analysis_Gating..N-1 连续整数。

    参数
    ----
    octa_image : 原始 OCTA 图像（灰度或 RGB）
    coarse_mask : 粗分割前景（例如 U_Net 的非背景区域），bool 或 Analysis_Gating/1
    uncertainty : 不确定性图（可选）
    num_superpixels : 期望超像素数
    random_state : 随机种子
    """
    labels, seeds, debug = compute_superpixels_rsg_snic(
        image=octa_image,
        coarse_mask=coarse_mask,
        uncertainty=uncertainty,
        num_superpixels=num_superpixels,
        random_state=random_state,
    )
    lab = labels.astype(np.int64)
    uniq = np.unique(lab)
    lut = {int(u): i for i, u in enumerate(uniq)}
    for u, i in lut.items():
        lab[lab == u] = i
    return lab


if __name__ == "__main__":
    # 简单自检
    H, W = 128, 160
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), dtype=np.float32)
    img += 0.1 + 0.05*np.random.RandomState(0).randn(H, W).astype(np.float32)
    center1 = (H//3, W//3)
    center2 = (2*H//3, 2*W//3)
    r1, r2 = 16, 12
    mask1 = ((yy-center1[0])**2)/(r1*r1) + ((xx-center1[1])**2)/(r2*r2) <= 1.0
    mask2 = ((yy-center2[0])**2)/((r1+4)*(r1+4)) + ((xx-center2[1])**2)/((r2+6)*(r2+6)) <= 1.0
    img[mask1] += 0.6
    img[mask2] += 0.5
    img = np.clip(img, 0, 1)

    labels, seeds, debug = compute_superpixels_rsg_snic(
        img, num_superpixels=300, shr_quota_ratio=0.6, random_state=42
    )
    print("RSG+SNIC demo labels shape:", labels.shape)
