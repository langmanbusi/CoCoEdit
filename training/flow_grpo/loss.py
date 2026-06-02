import torch
import torch.nn.functional as F
import ipdb
# === 输入要求 ===
# x0, xt, t_expanded, positive_prediction, implicit_negative_prediction, r, valid_mask 已按你代码得到
# train_sample_batch 需包含 "edit_mask" （1=需编辑），同 x0 的空间尺寸；通道为 1
# 可选：config.train.use_lpips, config.train.lpips_tau_pos, lpips_tau_neg, weights 等超参
# x0, xt, positive_prediction, implicit_negative_prediction: [4, 1024, 64]
# edit_mask: [4, 512, 512]
# t_expanded: [4, 1, 1]
# latents = self._unpack_latents(latents, height, width, self.vae_scale_factor) config.resolution=512, pipeline.vae_scale_factor=8
debug = True

def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

def calculate_loss(x0, xt, t_expanded, positive_prediction, implicit_negative_prediction, r, valid_mask, config, train_sample_batch, loss_terms):
    # adaptive weighting
    x0_prediction = xt - t_expanded * positive_prediction
    with torch.no_grad():
        weight_factor = (
            torch.abs(x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
        dim=tuple(range(1, x0.ndim))
    )
    negative_x0_prediction = (
        xt - t_expanded * implicit_negative_prediction
    )
    with torch.no_grad():
        negative_weight_factor = (
            torch.abs(negative_x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    negative_loss = (
        (negative_x0_prediction - x0) ** 2 / negative_weight_factor
    ).mean(dim=tuple(range(1, x0.ndim)))
    ori_policy_loss = (
        r * positive_loss * valid_mask / config.beta
        + (1.0 - r) * negative_loss * valid_mask / config.beta
    )

    def mean_by_mask(x, mask):
        if mask.sum() == 0:
            return x.sum() * 0
        return x.sum() / mask.sum()

    policy_loss = mean_by_mask(ori_policy_loss * config.train.adv_clip_max, valid_mask)
    loss = policy_loss

    # === 前置：你已有这些张量/变量 ===
    # x0, xt, t_expanded, positive_prediction, implicit_negative_prediction, r, valid_mask, config, train_sample_batch, device
    if config.debug:
        ipdb.set_trace()
    # edit_mask_raw: [B,1,H,W]，1 表示“非编辑区域/背景”，0 表示“编辑区域”
    edit_mask_raw = train_sample_batch["masks_imgsize"].float().unsqueeze(1)
    height, width = edit_mask_raw.shape[-2:]

    x0_prediction = _unpack_latents(x0_prediction, height, width, 8).squeeze(2)
    negative_x0_prediction = _unpack_latents(negative_x0_prediction, height, width, 8).squeeze(2)
    # [4, 16, 1, 64, 64]
    if config.debug:
        ipdb.set_trace()
    edit_mask_raw = torch.nn.functional.interpolate(edit_mask_raw, size=x0_prediction.shape[-2:], mode="nearest").to(x0.dtype).to(x0.device)

    # 1) 掩膜与环带（在 latent 分辨率）
    # 直接作为背景掩膜使用（1 表示非编辑区域/背景）
    bg_mask_raw = edit_mask_raw  # [B,1,H,W] 或 [B,1,D,H,W]，值在 {0,1} 或 [0,1]

    # 编辑区掩膜为其补集
    ed_mask_raw = 1.0 - bg_mask_raw

    if config.debug:
        ipdb.set_trace()

    def morph_erode(mask01, radius=1):
        """
        对 2D 或 3D 掩膜进行腐蚀（0-1 软/硬掩膜均可）。
        radius: 腐蚀半径（像素/voxel），0 表示不腐蚀。
        自动根据 mask 维度选择 2D/3D。
        """
        if radius <= 0:
            return mask01
        k = 2 * radius + 1
        if mask01.ndim == 5:
            # (B,1,D,H,W)
            pooled = F.max_pool3d(1.0 - mask01, kernel_size=k, stride=1, padding=radius)
            eroded = 1.0 - pooled
        else:
            # (B,1,H,W)
            pooled = F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=radius)
            eroded = 1.0 - pooled
        return eroded.clamp(0, 1)

    def build_ring_with_erosion(bg_mask, ed_mask, r_bg=1, r_ed=1, thin_edge=False):
        """
        - 先分别腐蚀背景掩膜与编辑掩膜
        - 环带为两者腐蚀后之间的空隙
        - 如果 thin_edge=True，则生成贴边的薄环带
        返回: bg_mask_erode, ed_mask_erode, ring
        """
        bg_erode = morph_erode(bg_mask, radius=r_bg)
        ed_erode = morph_erode(ed_mask, radius=r_ed)

        # 环带：两者都不占的区域
        union_eroded = torch.clamp(bg_erode + ed_erode, 0, 1)
        ring = (1.0 - union_eroded).clamp(0, 1)

        if thin_edge:
            # 可选：将 ring 压成更薄的贴边条带
            # 方法：对腐蚀后的各掩膜做一次轻微膨胀，取膨胀后与腐蚀后的差作为边缘，再合并
            def morph_dilate(mask01, radius=1):
                if radius <= 0:
                    return mask01
                k = 2 * radius + 1
                if mask01.ndim == 5:
                    dil = F.max_pool3d(mask01, kernel_size=k, stride=1, padding=radius)
                else:
                    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=radius)
                return dil.clamp(0, 1)

            edge_bg = (morph_dilate(bg_erode, radius=1) - bg_erode).clamp(0, 1)
            edge_ed = (morph_dilate(ed_erode, radius=1) - ed_erode).clamp(0, 1)
            ring = torch.clamp(edge_bg + edge_ed, 0, 1)

        return bg_erode, ed_erode, ring

    # 使用：从“编辑区掩膜”ed_mask_raw 派生环带，但现在我们同时对两侧做腐蚀
    r_bg = getattr(config.train, "erode_radius_bg", 2)
    r_ed = getattr(config.train, "erode_radius_ed", 2)
    thin_edge = getattr(config.train, "ring_thin_edge", False)

    bg_mask, ed_mask, ring = build_ring_with_erosion(
        bg_mask=bg_mask_raw,
        ed_mask=ed_mask_raw,
        r_bg=r_bg,
        r_ed=r_ed,
        thin_edge=thin_edge
    )
    if config.debug:
        ipdb.set_trace()

    # 后续权重组合（示例）
    w_ring_pos = getattr(config.train, "ring_weight_pos", 0.2)  # 背景一侧的环带系数
    w_ring_neg = getattr(config.train, "ring_weight_neg", 0.2)  # 编辑区一侧的环带系数

    W_pos_bg = (bg_mask + w_ring_pos * ring).clamp(0, 1)
    W_neg_ed = (ed_mask + w_ring_neg * ring).clamp(0, 1)

    # 如果需要负分支在背景上加弱约束
    w_bg_neg = getattr(config.train, "bg_weight_neg", 0.0)
    W_neg_bg = (w_bg_neg * (bg_mask + w_ring_pos * ring)).clamp(0, 1) if w_bg_neg > 0 else None

    # 2) 从 v 预测还原 x̂0（latent 空间）
    # 3) 在 latent 上的加权 L2 距离（通道加权平均）
    def l2_dist_weighted_latent(a, b, W):
        # a,b: [B,C,...] latent；W: [B,1,...] 或 [B,C,...]
        diff2 = (a - b) ** 2
        if W.shape[1] == 1 and diff2.shape[1] != 1:
            W = W.expand(-1, diff2.shape[1], *([ -1 ] * (diff2.ndim - 2)))
        num = (diff2 * W).flatten(start_dim=1).sum(dim=1)
        den = (W.expand_as(diff2)).flatten(start_dim=1).sum(dim=1).clamp(min=1e-6)
        return num / den  # [B], 距离越大差异越大
    if config.debug:
        ipdb.set_trace()

    # 计算各区域距离
    if config.use_reg_xsrc:
        x_src = train_sample_batch["image_latents"]
        x0 = _unpack_latents(x0, height, width, 8).squeeze(2)
        x_src = _unpack_latents(x_src, height, width, 8).squeeze(2)
        d_bg_pos = l2_dist_weighted_latent(x0_prediction, x0, W_pos_bg)   # 正分支：非编辑区应小
        d_ed_neg = l2_dist_weighted_latent(negative_x0_prediction, x_src, W_neg_ed)   # 负分支：编辑区应大
        d_bg_neg = l2_dist_weighted_latent(negative_x0_prediction, x0, W_neg_bg) if W_neg_bg is not None else None
    else:
        x0 = _unpack_latents(x0, height, width, 8).squeeze(2)
        d_bg_pos = l2_dist_weighted_latent(x0_prediction, x0, W_pos_bg)   # 正分支：非编辑区应小
        d_ed_neg = l2_dist_weighted_latent(negative_x0_prediction, x0, W_neg_ed)   # 负分支：编辑区应大
        d_bg_neg = l2_dist_weighted_latent(negative_x0_prediction, x0, W_neg_bg) if W_neg_bg is not None else None

    # 4) 阈值化铰链（latent L2 的阈值通常比像素 L2 小）
    tau_pos = getattr(config.train, "tau_pos_l2_latent", 1e-3)   # 背景容忍度：~1e-3 到 1e-2
    tau_neg = getattr(config.train, "tau_neg_l2_latent", 3e-2)   # 编辑区要求：~2e-2 到 6e-2
    tau_bg_neg = getattr(config.train, "tau_bg_neg_l2_latent", 3e-2)

    # 正分支：超阈才惩罚（希望 d 小）
    L_bg_pos = torch.relu(d_bg_pos - tau_pos)  # [B]

    # 负分支：未达阈惩罚（希望 d 大）
    L_ed_neg = torch.relu(tau_neg - d_ed_neg)  # [B]

    L_bg_neg = None
    if d_bg_neg is not None:
        L_bg_neg = torch.relu(d_bg_neg - tau_bg_neg)  # [B]

    # 5) 时序权重（后期更强调背景一致）
    # t_scalar = train_sample_batch["timesteps"][:, j_idx].float()
    # T_max = float(getattr(config.train, "t_max", 1000.0))
    # t_frac = (t_scalar / T_max).clamp(0, 1)
    t_frac = t_expanded.view(-1)
    w_t_pos = (1.0 - t_frac).to(x0.dtype)     # 后期更强
    w_t_neg = torch.ones_like(w_t_pos)         # 负分支全时段可用
    w_t_pos = w_t_neg # 先不设时序权重

    # 6) 融合正则到损失
    lam_bg_pos = getattr(config.train, "lambda_bg_pos", 1.0)
    lam_ed_neg = getattr(config.train, "lambda_ed_neg", 0.0)
    lam_bg_neg = getattr(config.train, "lambda_bg_neg", 0.0)

    # with torch.no_grad():
    #     r_hat = r.clone()

    def reduce_with_mask(vec, mask):
        # vec: [B]，mask: [B]（样本级有效性掩膜）
        return (vec * mask).sum() / (mask.sum().clamp(min=1.0))
    if config.debug:
        ipdb.set_trace()

    pos_reg = lam_bg_pos * w_t_pos.to(L_bg_pos.dtype) * L_bg_pos
    neg_reg = lam_ed_neg * w_t_neg.to(L_ed_neg.dtype) * L_ed_neg
    if L_bg_neg is not None and lam_bg_neg > 0:
        neg_bg_reg = lam_bg_neg * w_t_neg.to(L_bg_neg.dtype) * L_bg_neg
        neg_reg = neg_reg + neg_bg_reg

    pos_reg_loss = reduce_with_mask(r * pos_reg, valid_mask)
    neg_reg_loss = reduce_with_mask((1.0 - r) * neg_reg, valid_mask)

    loss_reg = pos_reg_loss + neg_reg_loss
    loss = loss + loss_reg
    if config.debug:
        ipdb.set_trace()

    # 记录指标（便于监控）
    loss_terms["pos_bg_reg"] = pos_reg_loss.detach()
    loss_terms["neg_ed_reg"] = neg_reg_loss.detach()
    if L_bg_neg is not None and lam_bg_neg > 0:
        loss_terms["neg_bg_reg"] = (lam_bg_neg * reduce_with_mask((1.0 - r) * w_t_neg * L_bg_neg, valid_mask)).detach()
    else:
        loss_terms["neg_bg_reg"] = torch.tensor(0.0, device=x0.device)

    # loss_terms["d_bg_pos_mean"] = d_bg_pos.mean().detach()
    # loss_terms["d_ed_neg_mean"] = d_ed_neg.mean().detach()
    # if d_bg_neg is not None:
    #     loss_terms["d_bg_neg_mean"] = d_bg_neg.mean().detach()

    return loss, policy_loss, ori_policy_loss, loss_terms
