import argparse

import cv2
import numpy as np
import torch


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    image = image.astype(np.float32)
    max_val = float(image.max()) if image.size else 1.0
    return image / max_val if max_val > 1.0 else image


def load_image(path: str | None, size: int, device: torch.device) -> torch.Tensor:
    if path is not None:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = normalize_image(image)
    else:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = (
            0.35
            + 0.25 * np.sin(2.0 * np.pi * xx / 17.0)
            + 0.20 * np.cos(2.0 * np.pi * yy / 29.0)
            + 0.20 * ((xx - size * 0.55) ** 2 + (yy - size * 0.45) ** 2 < (size * 0.18) ** 2)
        )
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(image).to(device=device, dtype=torch.float32)


def load_psf(path: str | None, ksize: int, sigma: float, flip: bool, device: torch.device) -> torch.Tensor:
    if path is not None:
        if path.endswith(".npy"):
            psf = np.load(path).astype(np.float32)
        else:
            psf = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if psf is None:
                raise ValueError(f"Failed to read PSF: {path}")
            psf = normalize_image(psf)
    else:
        ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
        yy, xx = np.meshgrid(ax, ax, indexing="ij")
        psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)
    if psf.ndim == 3:
        psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = psf.astype(np.float32)
    psf_sum = float(psf.sum())
    if psf_sum <= 0.0:
        raise ValueError("PSF sum must be positive.")
    if flip:
        psf = np.flip(psf, axis=(0, 1)).copy()
    return torch.from_numpy(psf / psf_sum).to(device=device, dtype=torch.float32)


def reflect_index(x: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(x)
    period = 2 * (size - 1)
    x = torch.remainder(x, period)
    return torch.minimum(x, period - x)


def apply_boundary(y: torch.Tensor, x: torch.Tensor, h: int, w: int, boundary: str) -> tuple[torch.Tensor, torch.Tensor]:
    if boundary == "reflect":
        return reflect_index(y, h), reflect_index(x, w)
    return y.clamp(0, h - 1), x.clamp(0, w - 1)


def psf_offsets(psf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=psf.device),
        torch.arange(w, device=psf.device),
        indexing="ij",
    )
    dy = yy.reshape(-1).float() - (h - 1) / 2.0
    dx = xx.reshape(-1).float() - (w - 1) / 2.0
    weights = psf.reshape(-1)
    return dy, dx, weights


def direct_discrete_conv_at(
    image: torch.Tensor,
    psf: torch.Tensor,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    h, w = image.shape
    dy, dx, weights = psf_offsets(psf)
    src_y = target_y[:, None].float() + dy[None, :]
    src_x = target_x[:, None].float() + dx[None, :]
    src_y, src_x = apply_boundary(src_y, src_x, h, w, boundary)
    values = image[src_y.round().long(), src_x.round().long()]
    return (values * weights[None, :]).sum(dim=1)


def bilinear_sample(image: torch.Tensor, y: torch.Tensor, x: torch.Tensor, boundary: str) -> torch.Tensor:
    h, w = image.shape
    y, x = apply_boundary(y, x, h, w, boundary)
    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()
    y1 = torch.clamp(y0 + 1, max=h - 1)
    x1 = torch.clamp(x0 + 1, max=w - 1)
    wy = y - y0.float()
    wx = x - x0.float()
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    return (
        v00 * (1.0 - wy) * (1.0 - wx)
        + v01 * (1.0 - wy) * wx
        + v10 * wy * (1.0 - wx)
        + v11 * wy * wx
    )


def mc_conv_at(
    image: torch.Tensor,
    psf: torch.Tensor,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    num_samples: int,
    boundary: str,
    jitter: bool,
) -> torch.Tensor:
    h_psf, w_psf = psf.shape
    probs = psf.reshape(-1) / psf.sum()
    idx = torch.multinomial(probs, target_y.numel() * num_samples, replacement=True)
    yy = (idx // w_psf).float() - (h_psf - 1) / 2.0
    xx = (idx % w_psf).float() - (w_psf - 1) / 2.0
    yy = yy.view(target_y.numel(), num_samples)
    xx = xx.view(target_y.numel(), num_samples)
    if jitter:
        yy = yy + torch.rand_like(yy) - 0.5
        xx = xx + torch.rand_like(xx) - 0.5
        values = bilinear_sample(image, target_y[:, None].float() + yy, target_x[:, None].float() + xx, boundary)
    else:
        src_y, src_x = apply_boundary(target_y[:, None].float() + yy, target_x[:, None].float() + xx, *image.shape, boundary)
        values = image[src_y.round().long(), src_x.round().long()]
    return values.mean(dim=1)


def metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    err = estimate - reference
    return {
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "bias": float(err.mean().item()),
        "max_abs": float(err.abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MC PSF sampling against direct discrete convolution.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--psf_path", type=str, default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--psf_size", type=int, default=21)
    parser.add_argument("--psf_sigma", type=float, default=3.0)
    parser.add_argument("--flip_psf", action="store_true",
                        help="Flip the loaded PSF to match the training scripts' scipy.ndimage.convolve convention.")
    parser.add_argument("--num_pixels", type=int, default=8192)
    parser.add_argument("--num_trials", type=int, default=8)
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--boundary", choices=("reflect", "clamp"), default="reflect")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    image = load_image(args.image_path, args.size, device)
    psf = load_psf(args.psf_path, args.psf_size, args.psf_sigma, args.flip_psf, device)
    h, w = image.shape
    num_pixels = min(args.num_pixels, h * w)
    flat_idx = torch.randperm(h * w, device=device)[:num_pixels]
    target_y = flat_idx // w
    target_x = flat_idx % w
    reference = direct_discrete_conv_at(image, psf, target_y, target_x, args.boundary)

    print(f"device={device} image={tuple(image.shape)} psf={tuple(psf.shape)} boundary={args.boundary}")
    print("Reference: direct deterministic discrete PSF sum")
    print()
    print(f"{'samples':>8}  {'mode':>9}  {'mae':>11}  {'rmse':>11}  {'bias':>11}  {'max_abs':>11}")
    for num_samples in args.samples:
        for jitter in (False, True):
            trial_values = []
            for _ in range(args.num_trials):
                estimate = mc_conv_at(image, psf, target_y, target_x, num_samples, args.boundary, jitter)
                trial_values.append(metrics(estimate, reference))
            avg = {
                key: sum(item[key] for item in trial_values) / len(trial_values)
                for key in trial_values[0]
            }
            mode = "jitter" if jitter else "discrete"
            print(
                f"{num_samples:8d}  {mode:>9}  "
                f"{avg['mae']:11.6g}  {avg['rmse']:11.6g}  {avg['bias']:11.6g}  {avg['max_abs']:11.6g}"
            )

    print()
    print("Expected: discrete MC error should shrink roughly like 1/sqrt(samples).")
    print("If jitter keeps a nonzero floor at high samples, it is estimating a different forward model.")


if __name__ == "__main__":
    main()
