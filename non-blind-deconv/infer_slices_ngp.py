import argparse
import os
from typing import List

import numpy as np
import torch
import tifffile
from scipy.signal import fftconvolve
from tqdm import tqdm
import matplotlib.pyplot as plt  # [新增] 导入 matplotlib

from models.instantngp import InstantNGPTorchModel


def build_plane_normalized_coords(z_index: int, height: int, width: int, full_dims: tuple, device: torch.device) -> torch.Tensor:
    """
    Build normalized coordinates for a single z-plane.
    Returns [height*width, 3] in (x, y, z) order, values in [0, 1].
    """
    vz, vy, vx = full_dims

    ys_n = torch.arange(0, height, device=device, dtype=torch.float32) / (vy - 1)
    xs_n = torch.arange(0, width,  device=device, dtype=torch.float32) / (vx - 1)
    z_n  = float(z_index) / (vz - 1)

    grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    grid_z = torch.full((height, width), z_n, device=device)

    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
    return coords


def predict_plane(coords_flat: torch.Tensor, model: InstantNGPTorchModel, batch_size: int = 1_000_000):
    """Run model inference in batches to avoid OOM. Returns [N] float32 tensor."""
    predictions = []
    for i in range(0, coords_flat.shape[0], batch_size):
        batch_coords = coords_flat[i:i + batch_size]
        pred, _ = model(batch_coords, variance=None)
        predictions.append(pred)
    return torch.cat(predictions, dim=0)


def choose_slices(dz: int, num_slices: int, seed: int, mode: str) -> List[int]:
    if mode == "fixed":
        return [1050]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(dz, size=min(num_slices, dz), replace=False).tolist())
    elif mode == "range":
        step = max(1, dz // num_slices)
        return list(range(0, dz, step))[:num_slices]
    else:
        raise ValueError(f"Unknown slice_mode: {mode}")

# [新增] 用于图像归一化的辅助函数
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Min-Max 归一化，将图像像素值拉伸到 [0, 1] 之间以便可视化"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img


def main():
    parser = argparse.ArgumentParser(description="Infer z-slices from a trained Instant-NGP network and save as float32 TIF")
    parser.add_argument("--volume_tif",  type=str, default="/workspace/temp/workspace/non-blind-deconv/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--checkpoint",  type=str, default="/workspace/temp/workspace/checkpoints/lsm_deconv_gmm_reg.pth")
    parser.add_argument("--psf_path",    type=str, default="/workspace/temp/workspace/psf_t0_v0.tif")
    parser.add_argument("--out_dir",     type=str, default="/workspace/temp/workspace/inference/")
    parser.add_argument("--num_slices",  type=int, default=1)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--batch_size",  type=int, default=1_000_000)
    parser.add_argument("--slice_mode",  type=str, default="fixed", choices=["random", "fixed", "range"])
    parser.add_argument("--z_pad",       type=int, default=20,
                        help="Slices before/after target slice to include for 3-D PSF convolution")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # ── reference volume (shape only) ───────────────────────────────────────
    volume = tifffile.imread(args.volume_tif)
    dz, dy, dx = volume.shape
    del volume
    print(f"Reference volume shape: (z={dz}, y={dy}, x={dx})")

    # ── PSF ─────────────────────────────────────────────────────────────────
    print(f"Loading PSF from: {args.psf_path}")
    psf = tifffile.imread(args.psf_path).astype(np.float32)
    psf /= psf.sum()
    print(f"PSF shape: {psf.shape}")

    # ── checkpoint ──────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder_config = ckpt.get('encoder_config', None)
    decoder_config = ckpt.get('decoder_config', None)
    if encoder_config is not None:
        print(f"Loaded encoder config: {encoder_config}")
        print(f"Loaded decoder config: {decoder_config}")

    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        learn_variance=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded.")

    # ── inference ────────────────────────────────────────────────────────────
    z_indices      = choose_slices(dz, args.num_slices, args.seed, args.slice_mode)
    checkpoint_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]

    with torch.no_grad():
        for z_idx in z_indices:
            z_start          = max(0, z_idx - args.z_pad)
            z_end            = min(dz, z_idx + args.z_pad + 1)
            num_slices_chunk = z_end - z_start

            print(f"\nTarget slice z={z_idx} — inferring z=[{z_start}, {z_end})")

            # ── predict volume chunk (clear) ─────────────────────────────────
            clear_chunk = np.zeros((num_slices_chunk, dy, dx), dtype=np.float32)

            for i, z in enumerate(tqdm(range(z_start, z_end), desc="  z-slices", leave=False)):
                coords = build_plane_normalized_coords(z, dy, dx, (dz, dy, dx), device)
                pred   = predict_plane(coords, model, batch_size=args.batch_size)
                clear_chunk[i] = pred.view(dy, dx).cpu().numpy()

            clear_chunk = np.clip(clear_chunk, 0.0, 1.0)

            # ── extract & save clear slice ───────────────────────────────────
            slice_in_chunk = z_idx - z_start
            clear_slice    = clear_chunk[slice_in_chunk]

            clear_out  = (clear_slice * 65535.0).clip(0, 65535).astype(np.float32)
            clear_path = os.path.join(args.out_dir, f"{checkpoint_tag}_clear_z{z_idx:05d}.tif")
            tifffile.imwrite(clear_path, clear_out)
            print(f"  Saved clear  : {clear_path}  range=[{clear_out.min():.1f}, {clear_out.max():.1f}]")

            # ── convolve with PSF and extract blurred slice ──────────────────
            blurred_chunk = fftconvolve(clear_chunk, psf, mode='same')
            blurred_slice = blurred_chunk[slice_in_chunk]

            blurred_out  = (blurred_slice * 65535.0).clip(0, 65535).astype(np.float32)
            blurred_path = os.path.join(args.out_dir, f"{checkpoint_tag}_blurred_z{z_idx:05d}.tif")
            tifffile.imwrite(blurred_path, blurred_out)
            print(f"  Saved blurred: {blurred_path}  range=[{blurred_out.min():.1f}, {blurred_out.max():.1f}]")

            # ── [新增] 归一化并保存为并排比较的 PDF ────────────────────────────
            print("  Generating comparison PDF...")
            clear_norm = normalize_image(clear_slice)
            blurred_norm = normalize_image(blurred_slice)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 左侧：Clear (Deconvolved)
            axes[0].imshow(clear_norm, cmap='gray')
            axes[0].set_title(f"Clear Slice (z={z_idx})")
            axes[0].axis('off')

            # 右侧：Blurred (Convolved with PSF)
            axes[1].imshow(blurred_norm, cmap='gray')
            axes[1].set_title(f"Blurred Slice (z={z_idx})")
            axes[1].axis('off')

            plt.tight_layout()
            pdf_path = os.path.join(args.out_dir, f"{checkpoint_tag}_comparison_z{z_idx:05d}.pdf")
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close(fig)  # 关闭图像防止内存泄漏
            print(f"  Saved comparison PDF: {pdf_path}")


if __name__ == "__main__":
    main()