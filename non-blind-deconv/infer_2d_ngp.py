"""
infer_compare.py
----------------
Infer the INR-fitted image, compare against GT and an optional baseline.

Outputs
-------
  <out_dir>/
    inferred.png            – network prediction (uint8)
    comparison.png          – side-by-side: [GT | Inferred | error0] (+ baseline columns)
    psnr_results.txt        – PSNR numbers

PSNR is computed in float32 [0, 1] space.
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.instantngp import InstantNGPTorchModel


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
        img = img / 255.0          # assume uint8 range
    return img



def build_2d_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Normalised (x, y) coordinate grid in [0, 1], shape (H*W, 2)."""
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width,  device=device, dtype=torch.float32)
    ys_n = ys / max(height - 1, 1)
    xs_n = xs / max(width  - 1, 1)
    gy, gx = torch.meshgrid(ys_n, xs_n, indexing="ij")
    return torch.stack([gx, gy], dim=-1).view(-1, 2)


@torch.no_grad()
def infer_image(model: InstantNGPTorchModel, height: int, width: int,
                device: torch.device, batch_size: int = 500_000) -> np.ndarray:
    """Run inference over the full image grid; return float32 [0,1] array."""
    coords = build_2d_coords(height, width, device)
    preds = []
    for i in range(0, coords.shape[0], batch_size):
        chunk = coords[i : i + batch_size]
        pred, _ = model(chunk, variance=None)
        preds.append(pred.cpu())
    pred_flat = torch.cat(preds, dim=0).numpy()
    img = pred_flat.reshape(height, width).astype(np.float32)
    # Normalise to [0, 1]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img


def error_map(pred: np.ndarray, gt: np.ndarray, cmap: str = "hot",
              vmax: float | None = None) -> np.ndarray:
    """Absolute error as an RGB uint8 image.

    Pass vmax to use a shared scale across multiple error maps.
    """
    err = np.abs(pred - gt)
    scale = vmax if vmax is not None else max(float(err.max()), 1e-8)
    err_norm = np.clip(err / scale, 0.0, 1.0)
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(err_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_comparison(gt: np.ndarray, inferred: np.ndarray,
                    baseline: np.ndarray | None,
                    psnr_inferred: float, ssim_inferred: float, mse_inferred: float,
                    psnr_baseline: float | None, ssim_baseline: float | None, mse_baseline: float | None,
                    out_path: str):
    """
    Without baseline : 1 row  [GT | Inferred | Error(Inferred vs GT)]
    With baseline    : 2 rows  row0: [GT | Inferred | Error(Inferred vs GT)]
                               row1: [GT | Baseline | Error(Baseline vs GT)]
    """
    def to_rgb(arr: np.ndarray) -> np.ndarray:
        return np.stack([arr, arr, arr], axis=-1)

    gt_rgb  = to_rgb(gt)
    inf_rgb = to_rgb(inferred)

    if baseline is not None:
        shared_vmax = max(
            float(np.abs(inferred - gt).max()),
            float(np.abs(baseline - gt).max()),
            1e-8,
        )
        err_inf = error_map(inferred, gt, vmax=shared_vmax)
        err_bas = error_map(baseline, gt, vmax=shared_vmax)
        bas_rgb  = to_rgb(baseline)
        rows = [
            [gt_rgb,  inf_rgb, err_inf],
            [gt_rgb,  bas_rgb, err_bas],
        ]
        titles = [
            ["GT",
             f"Inferred  PSNR={psnr_inferred:.2f} dB  SSIM={ssim_inferred:.4f}  MSE={mse_inferred:.2e}",
             "Error (Inferred vs GT)"],
            ["GT",
             f"Baseline  PSNR={psnr_baseline:.2f} dB  SSIM={ssim_baseline:.4f}  MSE={mse_baseline:.2e}",
             "Error (Baseline vs GT)"],
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150,
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.08})
        for r, (row_panels, row_titles) in enumerate(zip(rows, titles)):
            for c, (panel, title) in enumerate(zip(row_panels, row_titles)):
                axes[r, c].imshow(panel)
                axes[r, c].set_title(title, fontsize=8, pad=4)
                axes[r, c].axis("off")
    else:
        err_inf = error_map(inferred, gt)
        panels = [gt_rgb, inf_rgb, err_inf]
        titles = [
            "GT",
            f"Inferred  PSNR={psnr_inferred:.2f} dB  SSIM={ssim_inferred:.4f}  MSE={mse_inferred:.2e}",
            "Error (Inferred vs GT)",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150,
                                 gridspec_kw={"wspace": 0.05})
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=8, pad=4)
            ax.axis("off")

    fig.suptitle("Comparison", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer INR, compare vs GT (and optional baseline), output PSNR + visual."
    )
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--gt_image",     type=str, required=True)
    parser.add_argument("--baseline_image", type=str, default=None)
    parser.add_argument("--out_dir",      type=str, default="../inference_2d")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--batch_size",   type=int, default=500_000)
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
        # Resize to GT resolution if needed
        if baseline_raw.shape != gt.shape:
            print(f"  Resizing baseline {baseline_raw.shape} → {gt.shape}")
            baseline_raw = cv2.resize(baseline_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        baseline = baseline_raw

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder_config = ckpt.get("encoder_config")
    decoder_config = ckpt.get("decoder_config")
    if encoder_config is None:
        print("  Warning: no encoder_config in checkpoint – using defaults.")

    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  Model loaded.")

    # ── Infer ────────────────────────────────────────────────────────────────
    print(f"Inferring image ({H}x{W})…")
    inferred = infer_image(model, H, W, device, batch_size=args.batch_size)
    print(f"  Inferred range: [{inferred.min():.4f}, {inferred.max():.4f}]")

    # Save raw inferred image
    inferred_uint8 = (inferred * 255).clip(0, 255).astype(np.uint8)
    inferred_path = os.path.join(args.out_dir, "inferred.png")
    cv2.imwrite(inferred_path, inferred_uint8)
    print(f"  Saved inferred image: {inferred_path}")

    # ── PSNR ────────────────────────────────────────────────────────────────
    # All images are already in [0, 1]; explicitly re-clip for safety.
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

    # Save metrics
    txt_path = os.path.join(args.out_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Inferred vs GT : PSNR={psnr_inferred:.6f} dB  SSIM={ssim_inferred:.6f}  MSE={mse_inferred:.6e}\n")
        if psnr_baseline is not None:
            f.write(f"Baseline vs GT : PSNR={psnr_baseline:.6f} dB  SSIM={ssim_baseline:.6f}  MSE={mse_baseline:.6e}\n")
    print(f"Saved metrics: {txt_path}")

    # ── Comparison figure ─────────────────────────────────────────────────────
    comp_path = os.path.join(args.out_dir, "comparison.png")
    save_comparison(
        gt=gt_01,
        inferred=inferred_01,
        baseline=baseline_01 if baseline is not None else None,
        psnr_inferred=psnr_inferred, ssim_inferred=ssim_inferred, mse_inferred=mse_inferred,
        psnr_baseline=psnr_baseline, ssim_baseline=ssim_baseline, mse_baseline=mse_baseline,
        out_path=comp_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()