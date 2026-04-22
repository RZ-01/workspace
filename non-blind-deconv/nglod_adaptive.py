import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.nglod import NglodModel
from sample_strategy.random_sample import sample_random_coords


class VolumeDataset(Dataset):
    def __init__(self, norm_volume_tensor, num_mc_samples, num_pixels_per_step, num_batches, discrete_psf):
        self.norm_volume_tensor = norm_volume_tensor
        self.num_mc_samples = num_mc_samples
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.volume_shape = norm_volume_tensor.shape
        self.discrete_psf = discrete_psf.cpu()

        vz, vy, vx = self.volume_shape
        self.inv_shape = torch.tensor([1.0 / (vz - 1), 1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)

    def __len__(self):
        return self.num_batches

    def _generate_offsets(self):
        sampling_budget = self.num_mc_samples * self.num_pixels_per_step
        psf_shape = self.discrete_psf.shape

        psf_flat = self.discrete_psf.flatten()
        psf_flat = psf_flat / psf_flat.sum()
        sampled_indices = torch.multinomial(psf_flat, sampling_budget, replacement=True)

        d, h, w = psf_shape
        z_idx = sampled_indices // (h * w)
        y_idx = (sampled_indices % (h * w)) // w
        x_idx = sampled_indices % w
        offsets = torch.stack([
            z_idx.float() - (d - 1) / 2.0,
            y_idx.float() - (h - 1) / 2.0,
            x_idx.float() - (w - 1) / 2.0,
        ], dim=1)

        return offsets

    def __getitem__(self, idx):
        target_coords = sample_random_coords(self.num_pixels_per_step, self.volume_shape)

        z_idx = target_coords[:, 0].numpy()
        y_idx = target_coords[:, 1].numpy()
        x_idx = target_coords[:, 2].numpy()
        values = self.norm_volume_tensor[z_idx, y_idx, x_idx]
        target_values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)) / 65535.0

        sampled_offsets = self._generate_offsets()
        target_coords_normalized = target_coords.float() * self.inv_shape
        sampled_offsets_normalized = sampled_offsets.float() * self.inv_shape

        return {
            'target_coords': target_coords_normalized,
            'target_values': target_values,
            'sampled_offsets': sampled_offsets_normalized,
        }


def psf_uniform_sampling_step(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
) -> dict:
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)
    sampled_offsets = batch['sampled_offsets'].to(device, non_blocking=True)

    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    sampled_offsets = sampled_offsets.view(num_pixels, num_mc_samples, n_dims)

    source_coords = target_coords.unsqueeze(1) + sampled_offsets
    source_coords = torch.clamp(source_coords, 0.0, 1.0)
    source_coords_flat = source_coords.view(-1, n_dims)

    # reorder z,y,x → x,y,z for model input
    coords_for_model = torch.stack([
        source_coords_flat[:, 2],
        source_coords_flat[:, 1],
        source_coords_flat[:, 0],
    ], dim=-1).float()
    coords_for_model.requires_grad_(False)

    pred_flat, _ = model(coords_for_model)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    loss = F.mse_loss(simulated_values.float(), target_values.float()) * 100

    return {"reconstruction_loss": loss, "total_loss": loss}


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--volume_tif", type=str, default="/workspace/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--psf_path",   type=str, default="/workspace/psf_t0_v0.tif")
    parser.add_argument("--save_path",  type=str, default="../checkpoints/nglod_mouse_heart_6kprogressive.pth")
    parser.add_argument("--logdir",     type=str, default="../runs/nglod_mouse_heart_6kprogressive")

    # Training
    parser.add_argument("--steps",               type=int,   default=10000)
    parser.add_argument("--lr",                  type=float, default=1e-2)
    parser.add_argument("--num_mc_samples",      type=int,   default=100,     help="PSF samples per pixel")
    parser.add_argument("--num_pixels_per_step", type=int,   default=30000, help="Pixels sampled per step")

    # NglodModel architecture
    parser.add_argument("--num_lods",     type=int, default=5,   help="Total number of octree LODs")
    parser.add_argument("--base_lod",     type=int, default=3,   help="Coarsest active LOD index")
    parser.add_argument("--feature_dim",  type=int, default=16,  help="Feature dim per octree node")
    parser.add_argument("--feature_size", type=int, default=4,   help="Base spatial resolution of feature grid")
    parser.add_argument("--hidden_dim",   type=int, default=128, help="MLP hidden width")
    parser.add_argument("--num_layers",   type=int, default=1,   help="MLP hidden layers")
    parser.add_argument("--sdfnet_root",  type=str, default="/workspace/non-blind-deconv/models/sdf-net", help="Path to sdf-net directory (default: ../sdf-net)")

    # Progressive LOD unlock
    parser.add_argument("--progressive_steps", type=int, default=6000,
                        help="Total steps over which all LODs are unlocked")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── load volume (memory-mapped to avoid OOM) ────────────────────────────
    try:
        vol_norm = tifffile.memmap(args.volume_tif, mode='r')
        print(f"Volume memory-mapped: shape={vol_norm.shape}, dtype={vol_norm.dtype}")
    except (ValueError, NotImplementedError):
        print("WARNING: tifffile.memmap not supported (likely compressed). Falling back to full load.")
        vol_norm = tifffile.imread(args.volume_tif)
        print(f"Volume loaded: shape={vol_norm.shape}, dtype={vol_norm.dtype}")

    n_dims = len(vol_norm.shape)
    if n_dims != 3:
        raise ValueError(f"NglodModel requires 3-D input, got shape {vol_norm.shape}")

    # ── load PSF ────────────────────────────────────────────────────────────
    if args.psf_path.endswith((".tif", ".tiff")):
        discrete_psf_np = tifffile.imread(args.psf_path).astype(np.float32)
    elif args.psf_path.endswith(".npy"):
        discrete_psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PSF format: {args.psf_path}")

    discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
    print(f"Discrete PSF shape: {discrete_psf.shape}, range=[{discrete_psf.min():.4f}, {discrete_psf.max():.4f}]")

    if len(discrete_psf.shape) != n_dims:
        raise ValueError(f"PSF is {len(discrete_psf.shape)}-D but volume is {n_dims}-D")

    # ── model ───────────────────────────────────────────────────────────────
    model = NglodModel(
        n_input_dims=n_dims,
        num_lods=args.num_lods,
        base_lod=args.base_lod,
        feature_dim=args.feature_dim,
        feature_size=args.feature_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sdfnet_root=args.sdfnet_root,
    ).to(device)
    model.train()

    feature_params = sum(p.numel() for p in model.encoder.parameters())
    mlp_params     = sum(p.numel() for p in model.decoder.parameters())
    total_params   = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:   {total_params:,}")
    print(f"  Feature grid:     {feature_params:,}")
    print(f"  MLP decoder:      {mlp_params:,}")

    # ── optimiser & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler    = torch.amp.GradScaler()
    writer    = SummaryWriter(log_dir=args.logdir)

    # ── dataset / dataloader ────────────────────────────────────────────────
    dataset = VolumeDataset(
        norm_volume_tensor=vol_norm,
        num_mc_samples=args.num_mc_samples,
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
        discrete_psf=discrete_psf,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    data_iter  = iter(dataloader)

    # ── progressive LOD unlock ──────────────────────────────────────────────
    initial_lods   = args.base_lod          # start at the coarsest LOD
    total_lods     = args.num_lods
    lods_to_unlock = total_lods - initial_lods
    steps_per_lod  = max(1, args.progressive_steps // lods_to_unlock)
    print(f"Progressive training: {initial_lods} → {total_lods} LODs, "
          f"+1 every {steps_per_lod} steps over {args.progressive_steps} steps")

    last_set_level = -1

    pbar = tqdm(
        total=args.steps,
        desc="Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )

    for step in range(args.steps):
        # unlock one LOD at a time
        current_level = min(initial_lods + (step // steps_per_lod), total_lods)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level
            if step > 0:
                tag = "(max)" if current_level >= total_lods else ""
                print(f"\nStep {step}: Unlocked LOD {current_level}/{total_lods} {tag}")

        batch = next(data_iter)
        batch = {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        loss_dict = psf_uniform_sampling_step(
            model=model,
            batch=batch,
            num_mc_samples=args.num_mc_samples,
            device=device,
        )

        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item(), step)

        scaler.scale(loss_dict["total_loss"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        scaler.step(optimizer)
        scaler.update()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)
        writer.add_scalar("train/GradClipped", float(grad_norm > 50.0), step)

        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss_dict["reconstruction_loss"].item():.6f}',
                          'lr':   f'{current_lr:.2e}'})

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