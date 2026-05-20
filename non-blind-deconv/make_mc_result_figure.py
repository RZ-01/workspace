"""
Create a side-by-side result figure for MC sample-count experiments.

Columns:
  GT | Blur image | Discrete conv supervision | MC 200 | MC 800 | MC 2000 | MC 4096

The "Discrete conv supervision" column is the INR checkpoint trained by
instant_ngp_image_discrete.py, i.e. direct discrete PSF summation supervision
without Monte Carlo sampling.
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.instantngp import InstantNGPTorchModel


def load_gray_norm(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    else:
        img = img.astype(np.float32)
        max_val = float(img.max()) if img.size else 1.0
        if max_val > 1.0:
            img = img / max_val
    return np.clip(img, 0.0, 1.0)


def load_psf(path: str, flip: bool) -> np.ndarray:
    if path.endswith(".npy"):
        psf = np.load(path).astype(np.float32)
    else:
        psf = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if psf is None:
            raise ValueError(f"Cannot read PSF: {path}")
        if psf.ndim == 3:
            psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
        psf = psf.astype(np.float32)
    psf_sum = float(psf.sum())
    if psf_sum <= 0.0:
        raise ValueError("PSF sum must be positive.")
    psf = psf / psf_sum
    if flip:
        psf = np.flip(psf, axis=(0, 1)).copy()
    return psf


def build_2d_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=torch.float32) / max(height - 1, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32) / max(width - 1, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).view(-1, 2)


@torch.no_grad()
def infer_checkpoint(checkpoint_path: str, height: int, width: int, device: torch.device,
                     batch_size: int) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = InstantNGPTorchModel(
        encoder_config=ckpt.get("encoder_config"),
        decoder_config=ckpt.get("decoder_config"),
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    coords = build_2d_coords(height, width, device)
    preds = []
    for start in range(0, coords.shape[0], batch_size):
        pred, _ = model(coords[start:start + batch_size], variance=None)
        preds.append(pred.detach().cpu())
    img = torch.cat(preds, dim=0).numpy().reshape(height, width).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def align_to_reference(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_f32 = (src * 255.0).astype(np.float32)
    ref_f32 = (ref * 255.0).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(src_f32, ref_f32)
    h, w = ref.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(src, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def convolve_reflect(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    pad_h = psf.shape[0] // 2
    pad_w = psf.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    conv = cv2.filter2D(padded, ddepth=-1, kernel=psf, borderType=cv2.BORDER_CONSTANT)
    return np.clip(conv[pad_h:pad_h + image.shape[0], pad_w:pad_w + image.shape[1]], 0.0, 1.0)


def score_title(label: str, image: np.ndarray, gt: np.ndarray, show_metrics: bool) -> str:
    if not show_metrics:
        return label
    psnr = peak_signal_noise_ratio(gt, image, data_range=1.0)
    ssim = structural_similarity(gt, image, data_range=1.0)
    return f"{label}\nPSNR {psnr:.2f} / SSIM {ssim:.4f}"


def save_figure(panels: list[tuple[str, np.ndarray]], gt: np.ndarray, out_path: str,
                show_metrics: bool, dpi: int) -> None:
    h, w = gt.shape
    n_cols = len(panels)
    panel_w = 2.8
    fig_h = panel_w * h / w + (0.45 if show_metrics else 0.25)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, fig_h), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    for ax, (label, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        title = label if label in {"GT", "Blur"} else score_title(label, image, gt, show_metrics)
        ax.set_title(title, fontsize=9, pad=5)
        ax.axis("off")

    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.88, wspace=0.035)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_labelled_paths(items: list[str], default_prefix: str) -> list[tuple[str, str]]:
    parsed = []
    for idx, item in enumerate(items):
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label, path = f"{default_prefix} {idx + 1}", item
        parsed.append((label, path))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Make GT/blur/discrete-supervision/MC result comparison figure.")
    parser.add_argument("--gt_image", required=True)
    parser.add_argument("--blur_image", required=True)
    parser.add_argument("--direct_checkpoint", default=None,
                        help="Checkpoint from instant_ngp_image_discrete.py or another deterministic PSF-conv run.")
    parser.add_argument("--direct_label", default="Discrete",
                        help="Panel title for --direct_checkpoint result.")
    parser.add_argument("--psf_path", default=None,
                        help="If --direct_checkpoint is omitted, convolve GT with this PSF for the direct-conv column.")
    parser.add_argument("--flip_psf", action="store_true",
                        help="Flip PSF before direct convolution, matching the training scripts' PSF convention.")
    parser.add_argument("--mc_checkpoints", nargs="+", required=True,
                        help="MC checkpoints, optionally labelled as 'MC 200=/path/ckpt.pth'.")
    parser.add_argument("--out_path", default="../mc_result_figure.pdf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=500_000)
    parser.add_argument("--align", action="store_true",
                        help="Phase-align inferred images to GT before plotting/scoring.")
    parser.add_argument("--show_metrics", action="store_true",
                        help="Add PSNR/SSIM under result titles.")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt = load_gray_norm(args.gt_image)
    blur = load_gray_norm(args.blur_image)
    h, w = gt.shape
    if blur.shape != gt.shape:
        blur = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)

    panels: list[tuple[str, np.ndarray]] = [
        ("GT", gt),
        ("Blur", blur),
    ]

    if args.direct_checkpoint is not None:
        direct = infer_checkpoint(args.direct_checkpoint, h, w, device, args.batch_size)
        if args.align:
            direct = align_to_reference(direct, gt)
        panels.append((args.direct_label, direct))
    else:
        if args.psf_path is None:
            raise ValueError("Provide either --direct_checkpoint or --psf_path for the PSF direct-conv column.")
        psf = load_psf(args.psf_path, flip=args.flip_psf)
        panels.append((args.direct_label, convolve_reflect(gt, psf)))

    for label, checkpoint_path in parse_labelled_paths(args.mc_checkpoints, default_prefix="MC"):
        pred = infer_checkpoint(checkpoint_path, h, w, device, args.batch_size)
        if args.align:
            pred = align_to_reference(pred, gt)
        panels.append((label, pred))

    save_figure(panels, gt, args.out_path, show_metrics=args.show_metrics, dpi=args.dpi)
    print(f"Saved result figure: {args.out_path}")


if __name__ == "__main__":
    main()
