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
from gmm_psf import GMMPsf


class ImageDataset(Dataset):
    """
    Provides (target_coords, target_values) batches from a 2D image.

    PSF offset generation has been moved to the GPU training step
    (see generate_offsets_on_gpu) to eliminate a large CPU→GPU transfer.
    """
    def __init__(
        self,
        norm_image_tensor,
        num_pixels_per_step,
        num_batches,
    ):
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
            'target_values': target_values,              # (P,)    ~0.4 MB
        }


def generate_offsets_on_gpu(
    n: int,
    discrete_psf: torch.Tensor,
    device: torch.device,
    gmm_psf: GMMPsf = None,
) -> torch.Tensor:
    """
    Generate n PSF offset samples directly on the GPU.

    discrete mode : torch.multinomial on the GPU PSF tensor (integer offsets, 2D).
    GMM mode      : GMMPsf.sample_gpu (continuous sub-pixel offsets).

    Returns
    -------
    offsets : torch.Tensor, shape (n, 2), float32, on *device*
        Pixel-unit offsets centred at the PSF origin.
    """
    if gmm_psf is not None:
        return gmm_psf.sample_gpu(n, device)

    # ── Discrete branch ──────────────────────────────────────────────────
    psf_flat = discrete_psf.flatten()
    psf_flat = psf_flat / psf_flat.sum()
    idx = torch.multinomial(psf_flat, n, replacement=True)
    h, w = discrete_psf.shape
    y_idx = idx // w
    x_idx = idx % w
    return torch.stack([
        y_idx.float() - (h - 1) / 2.0,
        x_idx.float() - (w - 1) / 2.0,
    ], dim=1)


def compute_gradient_losses(
    model: nn.Module,
    coords: torch.Tensor,
    grad_sample_size: int,
    fd_eps: float,
    welsch_sigma: float,          # was: l0_beta
    stochastic_alpha: float = 0.0,
) -> tuple:
    N = coords.shape[0]

    # ── Subsample from the already-resident target_coords ─────────────────
    if grad_sample_size < N:
        perm = torch.randperm(N, device=coords.device)[:grad_sample_size]
        pts = coords[perm]                          # (S, 2) — detached view
    else:
        pts = coords                                # use all

    # ── Build perturbed coordinate sets ──────────────────────────────────
    # coords are (y, x); clamp to stay in [0, 1]
    pts_y = torch.clamp(pts + torch.tensor([fd_eps, 0.0], device=pts.device), 0.0, 1.0)
    pts_x = torch.clamp(pts + torch.tensor([0.0, fd_eps], device=pts.device), 0.0, 1.0)

    def _forward(c):
        # model expects (x, y) — swap from our (y, x) storage order
        c_model = torch.stack([c[:, 1], c[:, 0]], dim=-1).float()
        pred, _ = model(c_model, variance=None, stochastic_alpha=stochastic_alpha)
        return pred   # (S,)

    val_c = _forward(pts)
    val_y = _forward(pts_y)
    val_x = _forward(pts_x)

    dy = (val_y - val_c) / fd_eps
    dx = (val_x - val_c) / fd_eps

    tv_loss = (dy.abs() + dx.abs()).mean()

    grad_mag_sq = dy ** 2 + dx ** 2
    welsch_loss = (1.0 - torch.exp(-grad_mag_sq / (2.0 * welsch_sigma ** 2))).mean()

    return tv_loss, welsch_loss


def psf_uniform_sampling_step(
    model: nn.Module,
    batch: dict,
    num_mc_samples: int,
    device: torch.device,
    discrete_psf: torch.Tensor,
    inv_shape: torch.Tensor,
    gmm_psf: GMMPsf = None,
    stochastic_alpha: float = 0.0,
    # ── sharpness regularisation ─────────────────────────────────────────
    lambda_tv: float = 0.0,
    lambda_welsch: float = 0.0,      # was: lambda_l0
    grad_sample_size: int = 50_000,
    fd_eps: float = 1e-3,
    welsch_sigma: float = 5.0,       # was: l0_beta
    model_chunk_size: int = 1_000_000,
) -> dict:
    target_coords = batch['target_coords'].to(device, non_blocking=True)
    target_values = batch['target_values'].to(device, non_blocking=True)

    num_pixels = target_coords.shape[0]
    n_dims = target_coords.shape[1]
    sampling_budget = num_pixels * num_mc_samples

    # Generate offsets on GPU — no 120 MB CPU→GPU transfer per step
    sampled_offsets = generate_offsets_on_gpu(
        sampling_budget, discrete_psf=discrete_psf, device=device, gmm_psf=gmm_psf,
    )  # (N, n_dims), pixel units

    # Normalise to [0,1] coordinate space (same as target_coords)
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

    pred_chunks = []
    for i in range(0, coords_for_model.shape[0], model_chunk_size):
        chunk = coords_for_model[i : i + model_chunk_size]
        pred_chunk, _ = model(chunk, variance=None, stochastic_alpha=stochastic_alpha)
        pred_chunks.append(pred_chunk)
    pred_flat = torch.cat(pred_chunks, dim=0)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float())
    total_loss = data_loss * 100

    result = {
        "reconstruction_loss": data_loss * 100,
        "tv_loss":      torch.zeros(1, device=device),
        "welsch_loss":  torch.zeros(1, device=device),   # was: l0_loss
        "total_loss":   total_loss,
    }

    if lambda_tv > 0.0 or lambda_welsch > 0.0:
        tv_loss, welsch_loss = compute_gradient_losses(
            model=model,
            coords=target_coords,
            grad_sample_size=grad_sample_size,
            fd_eps=fd_eps,
            welsch_sigma=welsch_sigma,            # was: l0_beta=l0_beta
            stochastic_alpha=stochastic_alpha,
        )

        if lambda_tv > 0.0:
            total_loss = total_loss + lambda_tv * tv_loss
            result["tv_loss"] = tv_loss

        if lambda_welsch > 0.0:                           # was: lambda_l0
            total_loss = total_loss + lambda_welsch * welsch_loss
            result["welsch_loss"] = welsch_loss           # was: l0_loss

    result["total_loss"] = total_loss
    return result
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/temp/W_DIP/datasets/levin/blur/im1_kernel2_img.png")
    parser.add_argument("--psf_path", type=str, default=None,
                        help="Path to discrete 2D PSF file (image or .npy). "
                             "Required for 'discrete' mode or 'gmm' without --gmm_checkpoint.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/im1_kernel2_img.pth")
    parser.add_argument("--logdir", type=str, default="../runs/im1_kernel2_img")
    parser.add_argument("--num_mc_samples", type=int, default=100, help="Number of PSF samples per pixel")
    parser.add_argument("--model_chunk_size", type=int, default=1_000_000,
                        help="Max coords per model forward pass (avoids cuBLAS size limits)")
    parser.add_argument("--progressive_steps", type=int, default=300, help="Number of steps to unlock progressively")

    # Stochastic training params
    parser.add_argument("--sp_alpha_init", type=float, default=0.03,
                        help="Initial std dev for stochastic preconditioning")
    parser.add_argument("--sp_decay_fraction", type=float, default=0.33,
                        help="Fraction of training steps over which to decay alpha to 0 (Paper suggests ~1/3)")

    # Encoder config
    parser.add_argument("--num_levels", type=int, default=16, help="Number of hash encoding levels")
    parser.add_argument("--level_dim", type=int, default=2, help="Feature dimension per level")
    parser.add_argument("--base_resolution", type=int, default=16, help="Base grid resolution")
    parser.add_argument("--log2_hashmap_size", type=int, default=24, help="Log2 of hash table size")
    parser.add_argument("--desired_resolution", type=int, default=512, help="Finest resolution")

    # Decoder config
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of MLP")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of MLP layers")

    # PSF sampling mode
    parser.add_argument("--psf_mode", type=str, default="gmm", choices=["discrete", "gmm"],
                        help="PSF sampling mode: 'discrete' or 'gmm'")
    parser.add_argument("--gmm_checkpoint", type=str, default="../checkpoints/levin_kernel2.pkl",
                        help="[gmm mode] Path to a pre-fitted GMMPsf .pkl file (from gmm_psf.py).")

    parser.add_argument("--lambda_tv", type=float, default=1e-3,
                        help="Weight for L1-TV loss (stage 1, early training). "
                             "Suppresses hash-grid noise. Suggested: 1e-4 ~ 1e-2.")
    parser.add_argument("--lambda_welsch", type=float, default=0.0,
                        help="Weight for Welsch gradient loss (stage 2, late training). "
                             "Polarises gradients toward 0-or-large → sharpens edges. "
                             "Suggested: 5e-4 ~ 5e-3.")
    parser.add_argument("--tv_end_fraction", type=float, default=1.0,
                        help="Fraction of total steps at which TV switches OFF. "
                             "E.g. 0.4 → TV active for steps ")
    parser.add_argument("--welsch_start_fraction", type=float, default=0.4,
                        help="Fraction of total steps at which Welsch switches ON. "
                             "Set equal to tv_end_fraction for instant switch (recommended). "
                             "Set higher to insert a no-reg gap between stages.")
    parser.add_argument("--grad_sample_size", type=int, default=50_000,
                        help="Points subsampled from target_coords for gradient estimation.")
    parser.add_argument("--welsch_sigma", type=float, default=5.0,
                        help="Welsch saturation threshold (normalised-coord gradient units). "
                             "Gradients >> sigma → treated as edges, left unpenalised. "
                             "Rule of thumb: sigma ≈ 0.3 / fd_eps for a medium-contrast edge.")

    args = parser.parse_args()

    fd_eps = 3.0 / max(args.desired_resolution, 1)
    print(f"fd_eps auto-set to {fd_eps:.5f}  (3 / desired_resolution={args.desired_resolution})")

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

    print(f"Image loaded: shape={image_norm.shape}, dtype={image_norm.dtype}")
    num_pixels_per_step = image_norm.shape[0] * image_norm.shape[1]
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
        nodes = min(hashmap_max, res ** 2)
        total_nodes += nodes
    print(f"  Total hashgrid nodes: {total_nodes:,}")

    # ── Load PSF ───────────────────────────────────────────────────────
    if args.psf_mode == "discrete":
        if args.psf_path is None:
            raise ValueError("--psf_path is required for discrete mode.")
        if args.psf_path.endswith(".npy"):
            discrete_psf_np = np.load(args.psf_path).astype(np.float32)
        else:
            discrete_psf_np = cv2.imread(args.psf_path, cv2.IMREAD_UNCHANGED)
            if discrete_psf_np is None:
                raise ValueError(f"Unsupported or unreadable PSF format: {args.psf_path}")
            if discrete_psf_np.ndim == 3:
                discrete_psf_np = cv2.cvtColor(discrete_psf_np, cv2.COLOR_BGR2GRAY)
            discrete_psf_np = discrete_psf_np.astype(np.float32)
        if discrete_psf_np.ndim != 2:
            raise ValueError(f"Only 2D PSF is supported, got shape={discrete_psf_np.shape}")
        psf_sum = float(discrete_psf_np.sum())
        if psf_sum <= 0:
            raise ValueError("PSF sum must be positive.")
        discrete_psf_np = discrete_psf_np / psf_sum
        discrete_psf = torch.from_numpy(discrete_psf_np).float().to(device)
        print(f"Discrete PSF: shape={discrete_psf.shape}, "
              f"range=[{discrete_psf.min():.4f}, {discrete_psf.max():.4f}]")
        if len(discrete_psf.shape) != n_dims:
            raise ValueError(f"PSF dims {len(discrete_psf.shape)} != image dims {n_dims}")
    else:
        discrete_psf = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)

    tv_end_step       = int(args.steps * args.tv_end_fraction)
    welsch_start_step = int(args.steps * args.welsch_start_fraction)
    reg_enabled = args.lambda_tv > 0.0 or args.lambda_welsch > 0.0
    if reg_enabled:
        print(f"\nTwo-stage sharpness regularisation:")
        print(f"  Stage 1 — TV     : steps [0, {tv_end_step})        lambda={args.lambda_tv}")
        print(f"  Stage 2 — Welsch : steps [{welsch_start_step}, {args.steps}]  lambda={args.lambda_welsch}")
        print(f"  welsch_sigma={args.welsch_sigma}  fd_eps={fd_eps:.5f}  grad_sample_size={args.grad_sample_size}")
        if tv_end_step < welsch_start_step:
            print(f"  Gap (no reg)     : steps [{tv_end_step}, {welsch_start_step})")
        else:
            print(f"  Instant switch at step {tv_end_step}.")
    else:
        print("\nSharpness regularisation disabled (lambda_tv=0, lambda_welsch=0). "
              "Pass --lambda_tv / --lambda_welsch to enable.")
    # ═══════════════════════════════════════════════════════════════════════

    # ── PSF sampling mode setup ──────────────────────────────────────────
    gmm_psf = None
    if args.psf_mode == "gmm":
        if not args.gmm_checkpoint:
            raise ValueError("--gmm_checkpoint is required for gmm mode. "
                             "Run gmm_psf.py first to fit and save the GMM.")
        gmm_psf = GMMPsf.load(args.gmm_checkpoint)
        if gmm_psf.n_dims != 2:
            raise ValueError(f"Only 2D GMM PSF is supported, got n_dims={gmm_psf.n_dims}")
        print(f"PSF mode: GMM  (K={gmm_psf.n_components}, cov={gmm_psf.gmm.covariance_type})")
    else:
        print("PSF mode: discrete (multinomial integer offsets)")

    # Pre-cache GMM tensors on GPU to avoid repeated CPU→GPU copies
    if gmm_psf is not None:
        gmm_psf.prepare_gpu(device)

    dataset = ImageDataset(
        norm_image_tensor=image_norm,
        num_pixels_per_step=num_pixels_per_step,
        num_batches=args.steps,
    )
    inv_shape = dataset.inv_shape  # keep a reference for the training step

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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

        step_lambda_tv     = args.lambda_tv     if step < tv_end_step       else 0.0
        step_lambda_welsch = args.lambda_welsch if step >= welsch_start_step else 0.0

        loss_dict = psf_uniform_sampling_step(
            model=model,
            batch=batch,
            num_mc_samples=args.num_mc_samples,
            device=device,
            discrete_psf=discrete_psf,
            inv_shape=inv_shape,
            gmm_psf=gmm_psf,
            stochastic_alpha=current_alpha,
            lambda_tv=step_lambda_tv,
            lambda_welsch=step_lambda_welsch,    # was: lambda_l0
            grad_sample_size=args.grad_sample_size,
            fd_eps=fd_eps,
            welsch_sigma=args.welsch_sigma,      # was: l0_beta
            model_chunk_size=args.model_chunk_size,
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
        scheduler.step()
        writer.add_scalar("train/LearningRate", current_lr, step)
        writer.add_scalar("train/GradNorm", grad_norm.item(), step)
        writer.add_scalar("train/GradClipped", float(grad_norm > 100.0), step)

        pbar.update(1)
        postfix_dict = {
            'recon_loss': f'{loss_dict["reconstruction_loss"].item():.6f}',
        }
        if reg_enabled:
            if step_lambda_tv > 0.0:
                postfix_dict['tv'] = f'{loss_dict["tv_loss"].item():.4f}'
            if step_lambda_welsch > 0.0:
                postfix_dict['welsch'] = f'{loss_dict["welsch_loss"].item():.4f}'
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