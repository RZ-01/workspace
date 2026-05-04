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

from models.instantngp import InstantNGPTorchModel

def gradient_prior_loss(model, H, W, patch_size, device, p=0.66, stochastic_alpha=0.0):
    y0 = torch.randint(0, H - patch_size + 1, (1,)).item()
    x0 = torch.randint(0, W - patch_size + 1, (1,)).item()
    ys = torch.arange(y0, y0 + patch_size, device=device).float() / max(H - 1, 1)
    xs = torch.arange(x0, x0 + patch_size, device=device).float() / max(W - 1, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx, gy], dim=-1).view(-1, 2)
    pred, _ = model(coords, variance=None, stochastic_alpha=stochastic_alpha)
    pred = pred.view(patch_size, patch_size)
    dx = pred[:, 1:] - pred[:, :-1]
    dy = pred[1:, :] - pred[:-1, :]
    eps = 1e-6
    return (dx.abs() + eps).pow(p).mean() + (dy.abs() + eps).pow(p).mean()

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

    coords_for_model = torch.stack([
        source_coords_flat[:, 1],
        source_coords_flat[:, 0],
    ], dim=-1).float()
    coords_for_model.requires_grad_(False)

    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float()) * 100

    return {
        "reconstruction_loss": data_loss,
        "total_loss":          data_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/temp/W_DIP/datasets/micro_bad/blur/im0_k0.png")
    parser.add_argument("--psf_path", type=str, default="/workspace/temp/W_DIP/datasets/micro_bad/gt/k0.png",
                        help="Path to discrete 2D PSF file (image or .npy).")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/micro_discrete.pth")
    parser.add_argument("--logdir", type=str, default="../runs/micro_discrete")
    parser.add_argument("--num_mc_samples", type=int, default=300)
    parser.add_argument("--progressive_steps", type=int, default=300)
    parser.add_argument("--num_pixels_per_step", type=int, default=180000,
                        help="Number of pixels sampled per training step (default: 4096).")

    # Stochastic preconditioning
    parser.add_argument("--sp_alpha_init", type=float, default=0.0)
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33)

    # Encoder config
    parser.add_argument("--num_levels", type=int, default=16)
    parser.add_argument("--level_dim", type=int, default=2)
    parser.add_argument("--base_resolution", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=21)
    parser.add_argument("--desired_resolution", type=int, default=512)

    # Decoder config
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

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
    n_dims = 2

    encoder_config = {
        "otype": "HashGrid",
        "n_levels": args.num_levels,
        "n_features_per_level": args.level_dim,
        "log2_hashmap_size": args.log2_hashmap_size,
        "base_resolution": args.base_resolution,
        "per_level_scale": np.exp(
            (np.log(args.desired_resolution) - np.log(args.base_resolution)) / (args.num_levels - 1)
        ),
    }
    decoder_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": args.hidden_dim,
        "n_hidden_layers": args.num_layers,
    }

    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=n_dims,
        learn_variance=False,
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

    n_levels = args.num_levels
    initial_levels = 4
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
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }, args.save_path)
    print(f"\nSaved model to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()
