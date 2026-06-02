import os
import glob
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_mask_as_tensor(path, force_binary=True):
    """
    读取单通道掩膜为张量 [1,1,H,W]，数值范围 0~1。
    force_binary=True 时会将 >0 的像素置为 1。
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)
    # 归一化到 0~1；如果是 0/255，也会得到 0 和 1
    if arr.max() > 1.0:
        arr = arr / 255.0
    if force_binary:
        arr = (arr > 0.5).astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t

def save_overlay(output_path, base_mask, bg_erode, ed_erode, ring):
    """
    生成一个拼图图像，展示：
    - 原始掩膜 base_mask
    - 背景腐蚀 bg_erode
    - 编辑腐蚀 ed_erode
    - 环带 ring
    """
    base = base_mask.squeeze().cpu().numpy()
    bg_e = bg_erode.squeeze().cpu().numpy()
    ed_e = ed_erode.squeeze().cpu().numpy()
    ring_np = ring.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("raw bg mask (1=background)")
    axes[1].imshow(bg_e, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("bg eroded")
    axes[2].imshow(ed_e, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("ed eroded")
    # 环带用彩色便于观察
    axes[3].imshow(ring_np, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("ring")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def morph_erode(mask01, radius=1):
    """
    对 2D 掩膜进行腐蚀（0-1 软/硬掩膜均可）。
    mask01: [B,1,H,W]
    """
    if radius <= 0:
        return mask01
    k = 2 * radius + 1
    pooled = F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=radius)
    eroded = 1.0 - pooled
    return eroded.clamp(0, 1)

def morph_dilate(mask01, radius=1):
    if radius <= 0:
        return mask01
    k = 2 * radius + 1
    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=radius)
    return dil.clamp(0, 1)

def build_ring_with_erosion(bg_mask, ed_mask, r_bg=1, r_ed=1, thin_edge=False):
    """
    输入:
      bg_mask: [B,1,H,W], 1=背景(非编辑区)
      ed_mask: [B,1,H,W], 1=编辑区
    步骤:
      1) 分别腐蚀
      2) 环带 = 1 - clamp(bg_erode + ed_erode, 0, 1) 为空隙
      3) 若 thin_edge=True，则改用贴边薄环
    返回: bg_erode, ed_erode, ring
    """
    bg_erode = morph_erode(bg_mask, radius=r_bg)
    ed_erode = morph_erode(ed_mask, radius=r_ed)

    union_eroded = torch.clamp(bg_erode + ed_erode, 0, 1)
    ring = (1.0 - union_eroded).clamp(0, 1)

    if thin_edge:
        edge_bg = (morph_dilate(bg_erode, radius=1) - bg_erode).clamp(0, 1)
        edge_ed = (morph_dilate(ed_erode, radius=1) - ed_erode).clamp(0, 1)
        ring = torch.clamp(edge_bg + edge_ed, 0, 1)

    return bg_erode, ed_erode, ring

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="输入掩膜文件夹（单通道图像）")
    parser.add_argument("--out_dir", type=str, required=True, help="输出可视化文件夹")
    parser.add_argument("--glob", type=str, default="*mask.png", help="通配符，如 *.png, *.jpg")
    parser.add_argument("--r_bg", type=int, default=200, help="背景掩膜腐蚀半径")
    parser.add_argument("--r_ed", type=int, default=20, help="编辑掩膜腐蚀半径")
    parser.add_argument("--thin_edge", action="store_true", help="是否使用贴边薄环")
    parser.add_argument("--save_numpy", action="store_true", help="是否另存为 npy（bg_erode/ed_erode/ring）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not paths:
        print("No mask files found.")
        return

    for p in paths:
        try:
            mask_bg = load_mask_as_tensor(p, force_binary=True)  # 1=背景
            mask_ed = 1.0 - mask_bg

            import time
            t1 = time.time()
            bg_e, ed_e, ring = build_ring_with_erosion(
                mask_bg, mask_ed, r_bg=args.r_bg, r_ed=args.r_ed, thin_edge=args.thin_edge
            )
            print(time.time() - t1)
            # 可视化保存
            fname = os.path.splitext(os.path.basename(p))[0]
            out_png = os.path.join(args.out_dir, f"{fname}_viz.png")
            save_overlay(out_png, mask_bg, bg_e, ed_e, ring)
            print(f"Saved: {out_png}")

            if args.save_numpy:
                npy_dir = os.path.join(args.out_dir, "npy")
                os.makedirs(npy_dir, exist_ok=True)
                np.save(os.path.join(npy_dir, f"{fname}_bg_erode.npy"), bg_e.squeeze().cpu().numpy())
                np.save(os.path.join(npy_dir, f"{fname}_ed_erode.npy"), ed_e.squeeze().cpu().numpy())
                np.save(os.path.join(npy_dir, f"{fname}_ring.npy"), ring.squeeze().cpu().numpy())

        except Exception as e:
            print(f"Failed on {p}: {e}")

if __name__ == "__main__":
    main()

# python loss_vis.py --in_dir '/home/notebook/data/group/wyh/datasets/ImgEdit/Singleturn_images_samples_9_9_10_0_36k_forvis/results_remove_laion_part1/00000_00021_000211882' --out_dir './mask_vis'