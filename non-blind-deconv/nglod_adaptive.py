import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models.nglod import NglodModel

class ImageDataset(Dataset):
    def __init__(self, norm_image_tensor, num_pixels_per_step, num_batches):
        self.norm_image_tensor = norm_image_tensor
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.image_shape = norm_image_tensor.shape

        if len(self.image_shape) != 2:
            raise ValueError(f"Only 2D images are supported, got shape={self.image_shape}")

        h, w = self.image_shape
        self.inv_shape = torch.tensor(
            [1.0 / max(h - 1, 1), 1.0 / max(w - 1, 1)],
            dtype=torch.float32,
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        h, w = self.image_shape
        y_idx_t = torch.randint(0, h, (self.num_pixels_per_step,))
        x_idx_t = torch.randint(0, w, (self.num_pixels_per_step,))
        target_coords = torch.stack([y_idx_t, x_idx_t], dim=1)

        y_idx = target_coords[:, 0].numpy()
        x_idx = target_coords[:, 1].numpy()
        values = self.norm_image_tensor[y_idx, x_idx]
        target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        target_coords_normalized = target_coords.float() * self.inv_shape

        return {
            'target_coords': target_coords_normalized,   # (P, 2)
            'target_values': target_values,              # (P,)
        }


def generate_offsets_on_gpu(
    n: int,
    discrete_psf: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample n PSF offsets on the GPU.

    Steps:
      1. Sample a histogram bin via multinomial (integer pixel offset).
      2. Add uniform jitter within that pixel: U(-0.5, +0.5) per axis.

    Returns offsets of shape (n, 2), float32, in pixel units centred at PSF origin.
    """
    psf_flat = discrete_psf.flatten()
    psf_flat = psf_flat / psf_flat.sum()
    idx = torch.multinomial(psf_flat, n, replacement=True)
    h, w = discrete_psf.shape
    y_idx = idx // w
    x_idx = idx % w
    jitter = torch.rand((n, 2), device=device) - 0.5
    return torch.stack([
        y_idx.float() - (h - 1) / 2.0 + jitter[:, 0],
        x_idx.float() - (w - 1) / 2.0 + jitter[:, 1],
    ], dim=1)


def psf_uniform_sampling_step(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
    discrete_psf: torch.Tensor,
    inv_shape: torch.Tensor,
    stochastic_alpha: float = 0.0,
) -> dict:
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)

    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    sampling_budget = num_pixels * num_mc_samples

    sampled_offsets = generate_offsets_on_gpu(
        sampling_budget, discrete_psf=discrete_psf, device=device,
    )  # (N, 2), pixel units

    inv_shape_gpu = inv_shape.to(device, non_blocking=True)
    sampled_offsets = (sampled_offsets * inv_shape_gpu).view(num_pixels, num_mc_samples, n_dims)

    source_coords = target_coords.unsqueeze(1) + sampled_offsets
    source_coords = torch.clamp(source_coords, 0.0, 1.0)
    source_coords_flat = source_coords.view(-1, n_dims)

    # NglodModel requires 3-D input; pad with a fixed z=0.5 (image lives at mid-plane)
    coords_for_model = torch.stack([
        source_coords_flat[:, 1],
        source_coords_flat[:, 0],
        torch.full((source_coords_flat.shape[0],), 0.5, device=device),
    ], dim=-1).float()
    coords_for_model.requires_grad_(False)

    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float())

    return {
        "reconstruction_loss": data_loss,
        "total_loss":          data_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/nonblind/Deblur-INR/datasets/lai/im05_ker04_fft_circular.png")
    parser.add_argument("--psf_path", type=str, default="/workspace/nonblind/Deblur-INR/results/ker04_truth.png",
                        help="Path to discrete 2D PSF file (image or .npy).")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/im05_ker04_fft_circular.pth")
    parser.add_argument("--logdir", type=str, default="../runs/im05_ker04_fft_circular")
    parser.add_argument("--num_mc_samples", type=int, default=100)
    parser.add_argument("--progressive_steps", type=int, default=2000)
    parser.add_argument("--num_pixels_per_step", type=int, default=10000,
                        help="Number of pixels sampled per training step (default: 4096).")

    # Stochastic preconditioning
    parser.add_argument("--sp_alpha_init", type=float, default=0.03)
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33)

    # NglodModel config
    # base_lod=0 + num_lods=9 → finest 2D resolution = 2^(8+0) = 256 × 256
    parser.add_argument("--num_lods",     type=int, default=9,   help="Total number of octree LODs")
    parser.add_argument("--base_lod",     type=int, default=0,   help="Coarsest active LOD index")
    parser.add_argument("--feature_dim",  type=int, default=16,  help="Feature dim per octree node")
    parser.add_argument("--feature_size", type=int, default=4,   help="Base spatial resolution of feature grid")
    parser.add_argument("--hidden_dim",   type=int, default=128, help="MLP hidden width")
    parser.add_argument("--num_layers",   type=int, default=2,   help="MLP hidden layers")
    parser.add_argument("--sdfnet_root",  type=str,
                        default="/workspace/nonblind/workspace/non-blind-deconv/models/sdf-net",
                        help="Path to sdf-net directory")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {args.image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Only 2D images are supported, got shape={image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        image_norm = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image_norm = image.astype(np.float32)
        max_val = float(np.max(image_norm)) if image_norm.size > 0 else 1.0
        if max_val > 1.0:
            image_norm = image_norm / max_val

    print(f"Image loaded: shape={image_norm.shape}")
    num_pixels_per_step = args.num_pixels_per_step

    model = NglodModel(
        n_input_dims=3,
        num_lods=args.num_lods,
        base_lod=args.base_lod,
        feature_dim=args.feature_dim,
        feature_size=args.feature_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sdfnet_root=args.sdfnet_root,
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Load discrete PSF ─────────────────────────────────────────────────
    if args.psf_path.endswith(".npy"):
        discrete_psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        discrete_psf_np = cv2.imread(args.psf_path, cv2.IMREAD_UNCHANGED)
        if discrete_psf_np is None:
            raise ValueError(f"Unreadable PSF: {args.psf_path}")
        if discrete_psf_np.ndim == 3:
            discrete_psf_np = cv2.cvtColor(discrete_psf_np, cv2.COLOR_BGR2GRAY)
        discrete_psf_np = discrete_psf_np.astype(np.float32)
    if discrete_psf_np.ndim != 2:
        raise ValueError(f"PSF must be 2D, got shape={discrete_psf_np.shape}")
    psf_sum = float(discrete_psf_np.sum())
    if psf_sum <= 0:
        raise ValueError("PSF sum must be positive.")
    discrete_psf_np /= psf_sum
    discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
    # scipy.ndimage.convolve flips the kernel; flip here so the forward model matches
    discrete_psf = torch.flip(discrete_psf, [0, 1])
    print(f"PSF loaded: shape={discrete_psf.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)

    dataset = ImageDataset(
        norm_image_tensor=image_norm,
        num_pixels_per_step=num_pixels_per_step,
        num_batches=args.steps,
    )
    inv_shape = dataset.inv_shape

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)

    n_levels = args.num_lods
    initial_levels = 1  # start from coarsest LOD and ramp up
    steps_per_level = args.progressive_steps // max(n_levels - initial_levels, 1)
    sp_decay_steps = int(args.steps * args.sp_decay_fraction)
    last_set_level = -1

    pbar = tqdm(total=args.steps, desc="Training", dynamic_ncols=True)

    for step in range(args.steps):
        current_level = min(initial_levels + (step // steps_per_level), n_levels)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level

        batch = next(data_iter)
        batch = {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        if step < sp_decay_steps:
            progress = step / sp_decay_steps
            current_alpha = args.sp_alpha_init * np.exp(-5.0 * progress)
        else:
            current_alpha = 0.0

        loss_dict = psf_uniform_sampling_step(
            model=model,
            batch=batch,
            num_mc_samples=args.num_mc_samples,
            device=device,
            discrete_psf=discrete_psf,
            inv_shape=inv_shape,
            stochastic_alpha=current_alpha,
        )

        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item(), step)

        loss = loss_dict["total_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)

        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss_dict["reconstruction_loss"].item():.6f}'})

    pbar.close()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'nglod_config': {
            'num_lods':     args.num_lods,
            'base_lod':     args.base_lod,
            'feature_dim':  args.feature_dim,
            'feature_size': args.feature_size,
            'hidden_dim':   args.hidden_dim,
            'num_layers':   args.num_layers,
        },
    }, args.save_path)
    print(f"\nSaved model to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()
