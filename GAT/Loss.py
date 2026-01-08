import torch
import torch.nn.functional as F

def class_balanced_weights(counts, beta=0.99):
    eff_num = 1.0 - (beta ** counts.clamp(min=1))
    w = (1.0 - beta) / eff_num
    w = w / w.sum() * len(counts)   # 归一化到均值=1
    return w                        # [C]

def soft_ce_with_focal(logits, target_soft, class_weights=None, gamma=1.5, eps=1e-8):
    """
    logits: [K,C], target_soft: [K,C]
    """
    logp = F.log_softmax(logits, dim=-1)
    p    = logp.exp()
    # 软交叉熵
    ce   = -(target_soft * logp).sum(dim=-1)       # [K]
    # focal 调制，p_t = sum_c q_c * p_c
    pt   = (target_soft * p).sum(dim=-1).clamp(min=eps)
    mod  = (1.0 - pt) ** gamma
    loss = mod * ce                                 # [K]
    if class_weights is not None:
        # per-node 类别权重 = sum_c q_c * w_c
        w = (target_soft * class_weights[None, :]).sum(dim=-1)
        loss = loss * w
    return loss.mean()

def symmetric_kl(p, q, eps=1e-8):
    # p,q: [N,C], 均为概率分布
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    return ( (p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1) )

def edge_consistency_loss(logits, edge_index, same_label_mask=None, weight=None):

    p = F.softmax(logits, dim=-1)
    i, j = edge_index
    if same_label_mask is not None:
        i = i[same_label_mask]; j = j[same_label_mask]
        if i.numel() == 0: return logits.new_tensor(0.0)
    skl = symmetric_kl(p[i], p[j])   # [E_same]
    if weight is not None:
        skl = skl * weight.view(-1)  # 例如用低边界梯度的边赋更大权
    return skl.mean()

def broadcast_node_to_pixels(node_probs, sp_map_flat):
    return node_probs.index_select(dim=0, index=sp_map_flat)   # [HW,C]

def multiclass_dice_loss_from_pixels(pix_probs, gt_pix, C, mask=None, eps=1e-6):
    HW = gt_pix.numel()
    if mask is None:
        mask = torch.ones(HW, dtype=torch.bool, device=pix_probs.device)
    # one-hot GT
    oh = F.one_hot(gt_pix, num_classes=C).float()            # [HW,C]
    p  = pix_probs[mask]                                     # [H'M,C]
    g  = oh[mask]
    # Generalized Dice
    w  = 1.0 / (g.sum(dim=0).pow(2) + eps)                   # [C]
    inter = (p * g).sum(dim=0)                               # [C]
    union = (p + g).sum(dim=0)                               # [C]
    dice = (2*inter + eps) / (union + eps)                   # [C]
    gdice = 1.0 - (w * dice).sum() / (w.sum() + eps)
    return gdice

def kd_loss_node(logits_gat, probs_unet_node, T=2.0, mask=None):
    if mask is None: mask = torch.ones(logits_gat.size(0), dtype=torch.bool, device=logits_gat.device)
    if mask.sum() == 0: return logits_gat.new_tensor(0.0)
    p_s = F.log_softmax(logits_gat[mask] / T, dim=-1)        # student
    p_t = (probs_unet_node[mask] / probs_unet_node[mask].sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
    return loss

def total_loss(
    logits,                  # [K,C]
    y_soft,                  # [K,C]
    edge_index,              # [2,E]
    sp_map,                  # [H,W] (numpy→torch.long)
    gt_pix,                  # [H,W]
    class_counts=None,       # [C] (for CB weights)
    same_edge_mask=None,     # [E]
    hard_region_masks=None,  # dict: {'highlight': [H,W] bool, 'boundary': [H,W] bool}
    coarse_node=None,        # [K,C]
    kd_trusted_mask=None,    # [K] bool
    lamb=(1.0, 0.2, 0.5, 0.3),
    focal_gamma=1.5
):
    K, C = logits.shape
    device = logits.device
    class_weights = None
    if class_counts is not None:
        class_weights = class_balanced_weights(class_counts.to(device))
    L1 = soft_ce_with_focal(logits, y_soft, class_weights=class_weights, gamma=focal_gamma)

    L2 = edge_consistency_loss(logits, edge_index.to(device), same_label_mask=same_edge_mask)

    sp_flat = torch.from_numpy(sp_map.reshape(-1)).to(device).long()
    p_node  = F.softmax(logits, dim=-1)                      # [K,C]
    p_pix   = broadcast_node_to_pixels(p_node, sp_flat)      # [HW,C]
    gt_flat = gt_pix.reshape(-1).to(device).long()           # [HW]
    region_mask = None
    if hard_region_masks is not None:
        m = torch.zeros_like(gt_flat, dtype=torch.bool)
        for k in ('highlight','boundary'):
            if k in hard_region_masks and hard_region_masks[k] is not None:
                m |= hard_region_masks[k].reshape(-1).to(device).bool()
        region_mask = m
    L3 = multiclass_dice_loss_from_pixels(p_pix, gt_flat, C, mask=region_mask)

    L4 = torch.tensor(0.0, device=device)
    if (coarse_node is not None) and (kd_trusted_mask is not None):
        L4 = kd_loss_node(logits, coarse_node.to(device), T=2.0, mask=kd_trusted_mask.to(device))

    λ1, λ2, λ3, λ4 = lamb
    return {
        "L1_node_softCE": L1,
        "L2_edge_cons":   L2,
        "L3_dice_hard":   L3,
        "L4_kd":          L4,
        "L_total":        λ1*L1 + λ2*L2 + λ3*L3 + λ4*L4
    }
