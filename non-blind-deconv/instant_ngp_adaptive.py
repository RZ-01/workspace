import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models.instantngp import InstantNGPTorchModel
from sample_strategy.random_sample import sample_random_coords
from gmm_psf import GMMPsf


class VolumeDataset(Dataset):
    def __init__(
        self,
        norm_volume_tensor,
        num_mc_samples,
        num_pixels_per_step,
        num_batches,
        discrete_psf,
        gmm_psf: GMMPsf = None,
    ):
        self.norm_volume_tensor = norm_volume_tensor
        self.num_mc_samples = num_mc_samples
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.volume_shape = norm_volume_tensor.shape
        self.discrete_psf = discrete_psf.cpu()
        self.gmm_psf = gmm_psf          # None → discrete mode; set → GMM mode

        vz, vy, vx = self.volume_shape
        self.inv_shape = torch.tensor([1.0 / (vz - 1), 1.0 / (vy - 1), 1.0 / (vx - 1)], dtype=torch.float32)

    def __len__(self):
        return self.num_batches

    def _generate_offsets(self):
        sampling_budget = self.num_mc_samples * self.num_pixels_per_step

        if self.gmm_psf is not None:
            # ── GMM branch: continuous (sub-pixel) offsets ──────────────────
            # GMMPsf.sample() returns (N, n_dims) float32 pixel offsets,
            # already centred at the PSF origin, matching the discrete branch.
            offsets = self.gmm_psf.sample(sampling_budget)   # (N, n_dims)
        else:
            # ── Discrete branch: integer offsets via multinomial ─────────────
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

        # Convert to numpy indices for memory-mapped array access (avoids loading full volume)
        z_idx = target_coords[:, 0].numpy()
        y_idx = target_coords[:, 1].numpy()
        x_idx = target_coords[:, 2].numpy()
        values = self.norm_volume_tensor[z_idx, y_idx, x_idx]
        # Ensure native byte order, normalize to [0, 1] on the small batch only
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
    stochastic_alpha: float = 0.0,
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

    coords_for_model = torch.stack([
        source_coords_flat[:, 2],
        source_coords_flat[:, 1],
        source_coords_flat[:, 0],
    ], dim=-1)

    coords_for_model = coords_for_model.float()
    coords_for_model.requires_grad_(False)

    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float())
    total_loss = data_loss * 100

    return {
        "reconstruction_loss": data_loss * 100,
        "total_loss": total_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="/workspace/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--psf_path", type=str, default="/workspace/psf_t0_v0.tif")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/lsm_mouse_heart_constantLR.pth")
    parser.add_argument("--logdir", type=str, default="../runs/lsm_mouse_heart_constantLR")
    parser.add_argument("--num_mc_samples", type=int, default=100, help="Number of PSF samples per pixel")
    parser.add_argument("--num_pixels_per_step", type=int, default=100000, help="Number of pixels per step")
    parser.add_argument("--progressive_steps", type=int, default=1000, help="Number of steps to unlock progressively")

    # Stochastic training params
    parser.add_argument("--sp_alpha_init", type=float, default=0.03,
                        help="Initial std dev for stochastic preconditioning")
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33,
                        help="Fraction of training steps over which to decay alpha to 0 (Paper suggests ~1/3)")

    # Encoder config
    parser.add_argument("--num_levels", type=int, default=20, help="Number of hash encoding levels")
    parser.add_argument("--level_dim", type=int, default=2, help="Feature dimension per level")
    parser.add_argument("--base_resolution", type=int, default=16, help="Base grid resolution")
    parser.add_argument("--log2_hashmap_size", type=int, default=23, help="Log2 of hash table size")
    parser.add_argument("--desired_resolution", type=int, default=8192, help="Finest resolution")

    # Decoder config
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of MLP layers")

    # PSF sampling mode
    parser.add_argument("--psf_mode", type=str, default="discrete", choices=["discrete", "gmm"],
                        help="PSF sampling mode: 'discrete' (multinomial integer offsets) "
                             "or 'gmm' (Gaussian Mixture Model continuous offsets)")
    parser.add_argument("--gmm_components", type=int, default=64,
                        help="[GMM mode] Number of Gaussian components")
    parser.add_argument("--gmm_covariance_type", type=str, default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="[GMM mode] Covariance type for GaussianMixture")
    parser.add_argument("--gmm_max_iter", type=int, default=500,
                        help="[GMM mode] Maximum EM iterations when fitting GMM")
    parser.add_argument("--gmm_n_init", type=int, default=3,
                        help="[GMM mode] Number of EM initialisations (best kept)")
    parser.add_argument("--gmm_checkpoint", type=str, default=None,
                        help="[GMM mode] Path to a pre-fitted GMMPsf .pkl file. "
                             "If not set, the GMM is fitted from --psf_path before training.")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        vol_norm = tifffile.memmap(args.volume_tif, mode='r')
        print(f"Volume memory-mapped: shape={vol_norm.shape}, dtype={vol_norm.dtype}")
    except (ValueError, NotImplementedError):
        print("WARNING: tifffile.memmap not supported for this file (likely compressed). ")
        print("Falling back to full load — consider converting to uncompressed TIFF or .npy.")
        vol_norm = tifffile.imread(args.volume_tif)  # numpy array, original dtype
        print(f"Volume loaded: shape={vol_norm.shape}, dtype={vol_norm.dtype}")

    n_dims = len(vol_norm.shape)

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

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")

    n_levels = encoder_config["n_levels"]
    base = encoder_config["base_resolution"]
    per_level_scale = encoder_config["per_level_scale"]
    log2_hashmap_size = encoder_config["log2_hashmap_size"]
    hashmap_max = 2 ** log2_hashmap_size

    total_nodes = 0
    for i in range(n_levels):
        res = int(base * (per_level_scale ** i))
        nodes = min(hashmap_max, res ** 3)
        total_nodes += nodes
    print(f"  Total hashgrid nodes: {total_nodes:,}")

    # Load PSF
    if args.psf_path.endswith(".tif") or args.psf_path.endswith(".tiff"):
        discrete_psf_np = tifffile.imread(args.psf_path).astype(np.float32)
    elif args.psf_path.endswith(".npy"):
        discrete_psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PSF file format: {args.psf_path}. Expected .tif, .tiff, or .npy")

    discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
    print(f"Discrete PSF shape: {discrete_psf.shape}")
    print(f"Discrete PSF min: {discrete_psf.min():.6f}, max: {discrete_psf.max():.6f}")

    if len(discrete_psf.shape) != n_dims:
        raise ValueError(f"Discrete PSF dimensions {len(discrete_psf.shape)} does not match data dimensions {n_dims}D")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=args.logdir)

    # ── PSF sampling mode setup ──────────────────────────────────────────
    gmm_psf = None
    if args.psf_mode == "gmm":
        if args.gmm_checkpoint is not None and os.path.isfile(args.gmm_checkpoint):
            print(f"Loading pre-fitted GMM PSF from {args.gmm_checkpoint}")
            gmm_psf = GMMPsf.load(args.gmm_checkpoint)
        else:
            print("Fitting GMM PSF from discrete PSF ...")
            gmm_psf = GMMPsf.from_discrete_psf(
                psf=discrete_psf_np,
                n_components=args.gmm_components,
                covariance_type=args.gmm_covariance_type,
                max_iter=args.gmm_max_iter,
                n_init=args.gmm_n_init,
                verbose=True,
            )
            # Auto-save alongside the volume checkpoint
            gmm_save_path = args.save_path.replace(".pth", "_gmm.pkl")
            gmm_psf.save(gmm_save_path)
        print(f"PSF mode: GMM  (K={gmm_psf.n_components} components, "
              f"covariance={gmm_psf.gmm.covariance_type})")
    else:
        print("PSF mode: discrete (multinomial integer offsets)")

    dataset = VolumeDataset(
        norm_volume_tensor=vol_norm,  # numpy memmap — NOT converted to tensor
        num_mc_samples=args.num_mc_samples,
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
        discrete_psf=discrete_psf,
        gmm_psf=gmm_psf,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )
    data_iter = iter(dataloader)

    # Progressive training configuration
    initial_levels = 4
    steps_per_level = args.progressive_steps // (n_levels - initial_levels)
    print(f"Progressive training enabled:")
    print(f"  Starting with {initial_levels} levels, unlocking 1 level every {steps_per_level} steps")
    print(f"  Total levels: {n_levels}")

    sp_decay_steps = int(args.steps * args.sp_decay_fraction)
    last_set_level = -1

    pbar = tqdm(
        total=args.steps,
        desc="Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )

    for step in range(args.steps):
        # Update progressive training level
        current_level = min(initial_levels + (step // steps_per_level), n_levels)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level
            if step > 0:
                if current_level < n_levels:
                    print(f"\nStep {step}: Unlocked level {current_level}/{n_levels}")
                else:
                    print(f"\nStep {step}: Unlocked level {n_levels}/{n_levels} (max)")

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
            stochastic_alpha=current_alpha,
        )

        for key, value in loss_dict.items():
            writer.add_scalar(f"train/{key}", value.item() if value is not None else 0, step)

        loss = loss_dict["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()

        current_lr = optimizer.param_groups[0]['lr']
        #scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)
        writer.add_scalar("train/GradClipped", float(grad_norm > 100.0), step)

        pbar.update(1)
        postfix_dict = {
            'recon_loss': f'{loss_dict["reconstruction_loss"].item():.6f}',
        }
        if current_alpha > 1e-5:
            postfix_dict['sp_alpha'] = f'{current_alpha:.4f}'
        pbar.set_postfix(postfix_dict)

    pbar.close()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }, args.save_path)
    print(f"\nSaved final model to {args.save_path}")

    writer.close()


if __name__ == "__main__":
    main()