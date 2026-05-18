"""
Infer the fitted image from a trained NglodModel, compare against GT and an
optional baseline.

Outputs
-------
  <out_dir>/
    inferred.png      – network prediction (uint8)
    comparison.pdf    – side-by-side: [GT | Inferred | Error] (+ baseline row)
    metrics.txt       – PSNR / SSIM / MSE numbers

PSNR is computed in float32 [0, 1] space.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from models.nglod import NglodModel


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_gray_norm(path: str) -> np.ndarray:
    """Load an image as float32 grayscale normalised to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def align_to_reference(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Align src to ref via phase correlation (sub-pixel translation only)."""
    src_f32 = (src * 255).astype(np.float32)
    ref_f32 = (ref * 255).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(src_f32, ref_f32)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    H, W = src.shape
    shifted = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return shifted, (dx, dy)


def build_2d_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Normalised (x, y, z=0.5) grid, shape (H*W, 3).
    Matches the training convention in nglod_adaptive.py:
      coords_for_model = [x_n, y_n, 0.5]
    """
    ys_n = torch.arange(height, device=device, dtype=torch.float32) / max(height - 1, 1)
    xs_n = torch.arange(width,  device=device, dtype=torch.float32) / max(width  - 1, 1)
    gy, gx = torch.meshgrid(ys_n, xs_n, indexing="ij")
    gz = torch.full((height, width), 0.5, device=device)
    return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)


@torch.no_grad()
def infer_image(model: NglodModel, height: int, width: int,
                device: torch.device, batch_size: int = 500_000) -> np.ndarray:
    """Run inference over the full image grid; return float32 [0, 1] array."""
    coords = build_2d_coords(height, width, device)
    preds = []
    for i in range(0, coords.shape[0], batch_size):
        pred, _ = model(coords[i : i + batch_size])
        preds.append(pred.cpu())
    img = torch.cat(preds, dim=0).numpy().reshape(height, width).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def error_map(pred: np.ndarray, gt: np.ndarray, cmap: str = "hot",
              vmax: float | None = None) -> np.ndarray:
    """Absolute error as an RGB uint8 image."""
    err = np.abs(pred - gt)
    print(f"  Error map stats: max={err.max():.6f}, mean={err.mean():.6f}, min={err.min():.6f}")
    scale = vmax if vmax is not None else max(float(err.max()), 1e-8)
    err_norm = np.clip(err / scale, 0.0, 1.0)
    rgb = (plt.get_cmap(cmap)(err_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_comparison(gt: np.ndarray, inferred: np.ndarray,
                    baseline: np.ndarray | None,
                    psnr_inferred: float, ssim_inferred: float, mse_inferred: float,
                    psnr_baseline: float | None, ssim_baseline: float | None,
                    mse_baseline: float | None,
                    out_path: str):
    """
    Without baseline : 1 row  [GT | Inferred | Error(Inferred vs GT)]
    With baseline    : 2 rows  row0: [GT | Inferred | Error(Inferred vs GT)]
                               row1: [GT | Baseline | Error(Baseline vs GT)]
    """
    def to_rgb(arr):
        return np.stack([arr, arr, arr], axis=-1)

    H, W  = gt.shape
    col_w = 5.0
    fig_w = col_w * 3

    if baseline is not None:
        shared_vmax = max(
            float(np.abs(inferred - gt).max()),
            float(np.abs(baseline - gt).max()),
            1e-8,
        )
        rows = [
            [to_rgb(gt), to_rgb(inferred), error_map(inferred, gt, vmax=shared_vmax)],
            [to_rgb(gt), to_rgb(baseline), error_map(baseline, gt, vmax=shared_vmax)],
        ]
        titles = [
            ["GT",
             f"Inferred  PSNR={psnr_inferred:.2f} dB  SSIM={ssim_inferred:.4f}  MSE={mse_inferred:.2e}",
             "Error (Inferred vs GT)"],
            ["GT",
             f"Baseline  PSNR={psnr_baseline:.2f} dB  SSIM={ssim_baseline:.4f}  MSE={mse_baseline:.2e}",
             "Error (Baseline vs GT)"],
        ]
        row_h = col_w / (W / H) + 0.3
        fig, axes = plt.subplots(2, 3, figsize=(fig_w, row_h * 2),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.15})
        for r, (panels, row_titles) in enumerate(zip(rows, titles)):
            for c, (panel, title) in enumerate(zip(panels, row_titles)):
                axes[r, c].imshow(panel)
                axes[r, c].set_title(title, fontsize=8, pad=4)
                axes[r, c].axis("off")
    else:
        err_inf = error_map(inferred, gt)
        panels = [to_rgb(gt), to_rgb(inferred), err_inf]
        titles = [
            "GT",
            f"Inferred  PSNR={psnr_inferred:.2f} dB  SSIM={ssim_inferred:.4f}  MSE={mse_inferred:.2e}",
            "Error (Inferred vs GT)",
        ]
        fig_h = col_w / (W / H) + 0.3
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                                 gridspec_kw={"wspace": 0.05})
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=8, pad=4)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer INR (NglodModel), compare vs GT and optional baseline."
    )
    parser.add_argument("--checkpoint",     type=str, default="../checkpoints/im05_ker04_fft_circular.pth")
    parser.add_argument("--gt_image",       type=str, default="/workspace/nonblind/Deblur-INR/results/im05.png")
    parser.add_argument("--baseline_image", type=str, default="/workspace/nonblind/Deblur-INR/results/im05_ker04_fft_circular_x.png")
    parser.add_argument("--out_dir",        type=str, default="../inference_nglod")
    parser.add_argument("--sdfnet_root",    type=str,
                        default="/workspace/nonblind/workspace/non-blind-deconv/models/sdf-net")
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--batch_size",     type=int, default=500_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load GT ──────────────────────────────────────────────────────────────
    print(f"Loading GT image: {args.gt_image}")
    gt = load_gray_norm(args.gt_image)
    H, W = gt.shape
    print(f"  GT shape: {H} x {W}")

    # ── Load baseline (optional) ──────────────────────────────────────────────
    baseline = None
    if args.baseline_image:
        print(f"Loading baseline image: {args.baseline_image}")
        baseline_raw = load_gray_norm(args.baseline_image)
        if baseline_raw.shape != gt.shape:
            print(f"  Resizing baseline {baseline_raw.shape} → {gt.shape}")
            baseline_raw = cv2.resize(baseline_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        baseline_raw, (dx, dy) = align_to_reference(baseline_raw, gt)
        print(f"  Aligned baseline: shift = ({dx:.2f}, {dy:.2f}) px")
        baseline = baseline_raw

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt['nglod_config']
    print(f"  nglod_config: {cfg}")

    model = NglodModel(
        n_input_dims=3,
        num_lods=cfg['num_lods'],
        base_lod=cfg['base_lod'],
        feature_dim=cfg['feature_dim'],
        feature_size=cfg['feature_size'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        sdfnet_root=args.sdfnet_root,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.set_max_level(cfg['num_lods'])
    print("  Model loaded.")

    # ── Infer ────────────────────────────────────────────────────────────────
    print(f"Inferring image ({H}x{W})…")
    inferred = infer_image(model, H, W, device, batch_size=args.batch_size)
    print(f"  Inferred range: [{inferred.min():.4f}, {inferred.max():.4f}]")
    inferred, (dx, dy) = align_to_reference(inferred, gt)
    print(f"  Aligned inferred: shift = ({dx:.2f}, {dy:.2f}) px")

    inferred_path = os.path.join(args.out_dir, "inferred.png")
    cv2.imwrite(inferred_path, (inferred * 255).clip(0, 255).astype(np.uint8))
    print(f"  Saved inferred image: {inferred_path}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    gt_01       = gt.clip(0, 1)
    inferred_01 = inferred.clip(0, 1)

    psnr_inferred = peak_signal_noise_ratio(gt_01, inferred_01, data_range=1.0)
    ssim_inferred = structural_similarity(gt_01, inferred_01, data_range=1.0)
    mse_inferred  = mean_squared_error(gt_01, inferred_01)
    print(f"\nInferred vs GT:  PSNR={psnr_inferred:.4f} dB  SSIM={ssim_inferred:.6f}  MSE={mse_inferred:.6e}")

    psnr_baseline = ssim_baseline = mse_baseline = None
    if baseline is not None:
        baseline_01   = baseline.clip(0, 1)
        psnr_baseline = peak_signal_noise_ratio(gt_01, baseline_01, data_range=1.0)
        ssim_baseline = structural_similarity(gt_01, baseline_01, data_range=1.0)
        mse_baseline  = mean_squared_error(gt_01, baseline_01)
        print(f"Baseline vs GT:  PSNR={psnr_baseline:.4f} dB  SSIM={ssim_baseline:.6f}  MSE={mse_baseline:.6e}")

    txt_path = os.path.join(args.out_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Inferred vs GT : PSNR={psnr_inferred:.6f} dB  SSIM={ssim_inferred:.6f}  MSE={mse_inferred:.6e}\n")
        if psnr_baseline is not None:
            f.write(f"Baseline vs GT : PSNR={psnr_baseline:.6f} dB  SSIM={ssim_baseline:.6f}  MSE={mse_baseline:.6e}\n")
    print(f"Saved metrics: {txt_path}")

    # ── Comparison figure ─────────────────────────────────────────────────────
    comp_path = os.path.join(args.out_dir, "comparison.pdf")
    save_comparison(
        gt=gt_01, inferred=inferred_01, baseline=baseline_01 if baseline is not None else None,
        psnr_inferred=psnr_inferred, ssim_inferred=ssim_inferred, mse_inferred=mse_inferred,
        psnr_baseline=psnr_baseline, ssim_baseline=ssim_baseline, mse_baseline=mse_baseline,
        out_path=comp_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
