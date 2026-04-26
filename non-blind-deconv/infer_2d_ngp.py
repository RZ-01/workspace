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


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB. Both arrays must be float32 in [0, 1]."""
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


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


def error_map(pred: np.ndarray, gt: np.ndarray, cmap: str = "hot") -> np.ndarray:
    """Absolute error as an RGB uint8 image."""
    err = np.abs(pred - gt)            # [0, 1]
    err_norm = err / max(err.max(), 1e-8)
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(err_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_comparison(gt: np.ndarray, inferred: np.ndarray,
                    baseline: np.ndarray | None,
                    psnr_inferred: float, psnr_baseline: float | None,
                    out_path: str):
    """
    Side-by-side figure.

    Without baseline : [GT | Inferred | Error(Inferred vs GT)]
    With baseline    : [GT | Inferred | Error(Inferred) | Baseline | Error(Baseline)]
    """
    def to_rgb(arr: np.ndarray) -> np.ndarray:
        return np.stack([arr, arr, arr], axis=-1)

    gt_rgb  = to_rgb(gt)
    inf_rgb = to_rgb(inferred)
    err_inf = error_map(inferred, gt)

    if baseline is not None:
        err_bas = error_map(baseline, gt)
        bas_rgb = to_rgb(baseline)
        panels = [gt_rgb, inf_rgb, err_inf, bas_rgb, err_bas]
        titles = [
            "GT",
            f"Inferred  PSNR={psnr_inferred:.2f} dB",
            "Error (Inferred vs GT)",
            f"Baseline  PSNR={psnr_baseline:.2f} dB",
            "Error (Baseline vs GT)",
        ]
    else:
        panels = [gt_rgb, inf_rgb, err_inf]
        titles = [
            "GT",
            f"Inferred  PSNR={psnr_inferred:.2f} dB",
            "Error (Inferred vs GT)",
        ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150,
                             gridspec_kw={"wspace": 0.05})
    if n == 1:
        axes = [axes]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")

    # Shared colorbar for error maps (always the last one or two panels)
    # Simple: add a label instead
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

    psnr_inferred = psnr(inferred_01, gt_01)
    print(f"\nPSNR (Inferred vs GT): {psnr_inferred:.4f} dB")

    psnr_baseline = None
    if baseline is not None:
        baseline_01  = baseline.clip(0, 1)
        psnr_baseline = psnr(baseline_01, gt_01)
        print(f"PSNR (Baseline vs GT): {psnr_baseline:.4f} dB")

    # Save PSNR text
    txt_path = os.path.join(args.out_dir, "psnr_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Inferred vs GT : {psnr_inferred:.6f} dB\n")
        if psnr_baseline is not None:
            f.write(f"Baseline vs GT : {psnr_baseline:.6f} dB\n")
    print(f"Saved PSNR: {txt_path}")

    # ── Comparison figure ─────────────────────────────────────────────────────
    comp_path = os.path.join(args.out_dir, "comparison.png")
    save_comparison(
        gt=gt_01,
        inferred=inferred_01,
        baseline=baseline_01 if baseline is not None else None,
        psnr_inferred=psnr_inferred,
        psnr_baseline=psnr_baseline,
        out_path=comp_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()