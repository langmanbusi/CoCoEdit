import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, List, Union

def _to_tensor_nchw_uint01(x) -> torch.Tensor:
    """
    将输入转换为 NCHW 的 torch.FloatTensor，范围 [0,1]
    允许输入为:
      - torch.Tensor: 支持 NCHW 或 NHWC；范围 [0,1] 或 [0,255]
      - numpy.ndarray: 支持 NHWC
      - List[PIL.Image] 或 List[np.ndarray]
    """
    if isinstance(x, torch.Tensor):
        t = x
        if t.ndim == 4 and t.shape[1] in (1,3):  # NCHW
            pass
        elif t.ndim == 4 and t.shape[-1] in (1,3):  # NHWC -> NCHW
            t = t.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported tensor shape: {t.shape}")
        # 归一化到 [0,1]
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            # 假设已是 float，若值域>1 则按 255 缩放
            if t.max() > 1.0:
                t = t / 255.0
        return t.contiguous()
    elif isinstance(x, np.ndarray):
        arr = x
        if arr.ndim != 4 or arr.shape[-1] not in (1,3):
            raise ValueError(f"Expected NHWC numpy array, got {arr.shape}")
        t = torch.from_numpy(arr).permute(0,3,1,2).contiguous()
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            if t.max() > 1.0:
                t = t / 255.0
        return t
    elif isinstance(x, list):
        # 认为是 list of PIL.Image or np.array
        arrs = []
        for item in x:
            if hasattr(item, "size") and hasattr(item, "mode"):  # PIL Image
                arr = np.array(item)
            else:
                arr = np.array(item)
            if arr.ndim == 2:  # 灰度
                arr = arr[..., None]
            arrs.append(arr)
        arr = np.stack(arrs, axis=0)  # NHWC
        return _to_tensor_nchw_uint01(arr)
    else:
        raise ValueError(f"Unsupported input type: {type(x)}")

def _to_mask_tensor(images: torch.Tensor, masks) -> torch.Tensor:
    """
    将 mask 转为与 images 同形状的单通道 NCHW [0,1] float 张量。
    接受:
      - torch.Tensor (NCHW 或 NHWC)
      - numpy.ndarray (NHWC)
      - list 的 PIL/np
    白色(1或255)表示“不需要编辑的区域” -> 评分区域。
    """
    m = _to_tensor_nchw_uint01(masks)  # 转到 [0,1], NCHW
    if m.shape[1] == 3:
        # 若是三通道，转换为单通道（取最大，保守地把亮的都算白）
        m = m.max(dim=1, keepdim=True).values
    elif m.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Mask channel not 1 or 3: {m.shape}")
    # 双线性到目标尺寸（与 images 对齐）
    if (m.shape[2] != images.shape[2]) or (m.shape[3] != images.shape[3]):
        m = F.interpolate(m, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
    # 二值化（阈值可调，这里用0.5）
    m = (m >= 0.5).float()
    return m

def _masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    在 mask==1 的区域计算逐样本 MSE，x,y 为 NCHW，mask 为 N1HW
    返回形状 [N]
    """
    # broadcast mask 到通道
    mask_c = mask
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask_c = mask.repeat(1, x.shape[1], 1, 1)
    valid = mask.sum(dim=(1,2,3))  # [N]
    diff2 = ((x - y) ** 2) * mask_c
    mse = diff2.sum(dim=(1,2,3)) / (valid + eps)
    # 对于 valid=0 的样本，标记为 nan，后续处理
    mse = torch.where(valid > 0, mse, torch.full_like(mse, float("nan")))
    return mse

def _psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    PSNR = 10 * log10(MAX^2 / MSE)
    对于 mse=0，返回一个大值（用 clamp）
    """
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    return gauss

def _create_ssim_window(window_size: int, channel: int, device, dtype):
    g = _gaussian_window(window_size, 1.5, device, dtype)
    window_1d = g.unsqueeze(1)  # [W,1]
    window_2d = (window_1d @ window_1d.t())  # [W,W]
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim_masked(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, window_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    """
    基于经典 SSIM 实现，加入 mask，仅在 mask==1 区域统计。
    返回逐样本 SSIM，范围约 0-1。
    参考: Wang et al., 2004。实现与 pytorch-ssim 类似，但加入掩码归一化。
    """
    assert x.shape == y.shape and x.ndim == 4
    N, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # 准备窗口与常数
    window = _create_ssim_window(window_size, C, device, dtype)  # 形状应为 (C,1,k,k)
    padding = window_size // 2
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    m = mask
    if m.shape[1] == 1 and C != 1:
        m = m.repeat(1, C, 1, 1)

    def conv(img, kernel, padding, groups=None):
        N_, C_, H_, W_ = img.shape
        if kernel.ndim == 2:
            kH, kW = kernel.shape
            weight = torch.as_tensor(kernel, device=img.device, dtype=img.dtype).view(1, 1, kH, kW)
            weight = weight.repeat(C_, 1, 1, 1)
        elif kernel.ndim == 4:
            weight = torch.as_tensor(kernel, device=img.device, dtype=img.dtype)
            if weight.shape[0] == 1 and C_ > 1:
                weight = weight.repeat(C_, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported kernel shape: {kernel.shape}")
        if groups is None:
            groups = C_
        if weight.dtype != img.dtype:
            weight = weight.to(dtype=img.dtype)
        return F.conv2d(img, weight, padding=padding, groups=groups)

    mask_area = conv(m, torch.ones_like(window), padding=padding) + eps
    mu_x = conv(x * m, window, padding=padding) / mask_area
    mu_y = conv(y * m, window, padding=padding) / mask_area
    sigma_x = conv((x * x) * m, window, padding=padding) / mask_area - mu_x.pow(2)
    sigma_y = conv((y * y) * m, window, padding=padding) / mask_area - mu_y.pow(2)
    sigma_xy = conv((x * y) * m, window, padding=padding) / mask_area - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2))
    ssim_map_mean_c = ssim_map.mean(dim=1, keepdim=True)
    mask_single = mask if mask.shape[1] == 1 else mask[:, :1]
    valid = mask_single.sum(dim=(1,2,3))
    ssim_num = (ssim_map_mean_c * mask_single).sum(dim=(1,2,3))
    ssim = ssim_num / (valid + eps)
    ssim = torch.where(valid > 0, ssim, torch.full_like(ssim, float("nan")))
    return ssim.clamp(0.0, 1.0)

class MaskedPSNRSSIMScorer(torch.nn.Module):
    """
    输入: images, ref_images, masks
    输出: (scores, extras)
      - scores: [N] 的 0-1 分数
      - extras: 字典，包含 PSNR(dB), SSIM(0-1), PSNR_norm(0-1)
    """
    def __init__(self, device: Union[str, torch.device] = "cuda",
                 psnr_max_db: float = 40.0,
                 ssim_weight: float = 0.7):
        super().__init__()
        self.device = torch.device(device)
        self.psnr_max_db = float(psnr_max_db)
        self.ssim_weight = float(ssim_weight)

    def forward(self, images, ref_images, masks) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = _to_tensor_nchw_uint01(images).to(self.device)
        y = _to_tensor_nchw_uint01(ref_images).to(self.device)
        m = _to_mask_tensor(x, masks).to(self.device)

        if x.shape != y.shape:
            raise ValueError(f"images and ref_images shape mismatch: {x.shape} vs {y.shape}")

        mse = _masked_mse(x, y, m)  # [N], 可能包含 nan
        psnr = _psnr_from_mse(mse)  # [N]
        ssim = _ssim_masked(x, y, m)  # [N]

        # 将 nan（无有效区域）改为 0，并标记
        invalid = torch.isnan(psnr) | torch.isnan(ssim)
        psnr = torch.where(torch.isnan(psnr), torch.zeros_like(psnr), psnr)
        ssim = torch.where(torch.isnan(ssim), torch.zeros_like(ssim), ssim)

        # 归一化 PSNR -> [0,1]，上限截断在 psnr_max_db
        psnr_norm = (psnr / self.psnr_max_db).clamp(0.0, 1.0)

        w = self.ssim_weight
        score = w * ssim + (1.0 - w) * psnr_norm  # [N]

        extras = {
            "PSNR_masked": psnr,        # dB
            "SSIM_masked": ssim,        # 0-1
            "PSNR_norm": psnr_norm,     # 0-1
            "no_valid_area": invalid,   # [N] bool
        }
        return score#, extras


class MaskedSSIMCharbonnierScorer(torch.nn.Module):
    """
    输入: images, ref_images, masks
    输出: (scores, extras)
      - scores: [N] 的 0-1 分数
      - extras: 字典，包含 SSIM(0-1), Charbonnier(距离), Charb_sim(0-1)
    说明:
      - 在未编辑区域(1-mask)上评估一致性。
      - 最终分数 = ssim_weight * SSIM + (1-ssim_weight) * Charb_sim
    """
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        ssim_weight: float = 0.35,
        psnr_weight: float = 0.3,
        psnr_max_db: float = 40.0,
        charbonnier_eps: float = 1e-3,
        char_scale: float = 5.0,  # 将Charbonnier距离缩放到[0,1]相似度的经验尺度
    ):
        super().__init__()
        self.device = torch.device(device)
        self.ssim_weight = float(ssim_weight)
        self.psnr_weight = float(psnr_weight)
        self.psnr_max_db = float(psnr_max_db)
        self.charbonnier_eps = float(charbonnier_eps)
        self.char_scale = float(char_scale)

    def forward(self, images, ref_images, masks) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = _to_tensor_nchw_uint01(images).to(self.device)
        y = _to_tensor_nchw_uint01(ref_images).to(self.device)
        m = _to_mask_tensor(x, masks).to(self.device)  # 1表示编辑区域；未编辑区域为 (1-m)

        if x.shape != y.shape:
            raise ValueError(f"images and ref_images shape mismatch: {x.shape} vs {y.shape}")

        # psnr
        mse = _masked_mse(x, y, m)  # [N], 可能包含 nan
        psnr = _psnr_from_mse(mse)  # [N]
        invalid = torch.isnan(psnr) 
        psnr = torch.where(torch.isnan(psnr), torch.zeros_like(psnr), psnr)
        # 归一化 PSNR -> [0,1]，上限截断在 psnr_max_db
        psnr_norm = (psnr / self.psnr_max_db).clamp(0.0, 1.0)

        # SSIM（在未编辑区域上）
        ssim = _ssim_masked(x, y, m)  # [N], 期望返回0-1

        # Charbonnier 距离（在未编辑区域上）
        # diff: [N,C,H,W], bg: [N,1,H,W] 或 [N,H,W]（_to_mask_tensor应保证可广播）
        diff = x - y
        charb_map = torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps)
        # 按未编辑区域平均（避免不同bg面积引入偏置）
        denom = m.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-8)  # 若bg全0，会在后面标记invalid
        # 对通道取平均，再对空间做掩码平均
        # 先对空间：乘mask后除面积，再对通道均值
        charb_spatial = (charb_map * m).sum(dim=(-1, -2), keepdim=True) / (m.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8))
        charb = charb_spatial.mean(dim=1).squeeze(-1).squeeze(-1)  # [N]

        # 将 Charbonnier 距离转为相似度（0-1），距离越小分数越高
        charb_sim = (1.0 - self.char_scale * charb).clamp(0.0, 1.0)

        # 处理无有效未编辑区域的样本（如mask全1）
        invalid = (m.sum(dim=(-1, -2, -3)) == 0)  # [N] bool
        ssim = torch.where(torch.isnan(ssim), torch.zeros_like(ssim), ssim)
        charb = torch.where(torch.isnan(charb), torch.zeros_like(charb), charb)
        charb_sim = torch.where(torch.isnan(charb_sim), torch.zeros_like(charb_sim), charb_sim)

        w1 = self.ssim_weight
        w2 = self.psnr_weight
        score = w1 * ssim + w2 * psnr_norm + (1.0 - w1 - w2) * charb_sim  # [N], 0-1

        # extras = {
        #     "SSIM_masked": ssim,          # 0-1
        #     "Charbonnier_bg": charb,      # 距离，越小越好
        #     "Charb_sim": charb_sim,       # 0-1，相似度
        #     # "no_valid_area": invalid,     # [N] bool
        # }
        return score #, extras

