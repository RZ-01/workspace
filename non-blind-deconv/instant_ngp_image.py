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
    l0_beta: float,
    stochastic_alpha: float = 0.0,
) -> tuple:
    """
    Estimate INR spatial gradients via finite difference and compute:
      - tv_loss  : L1-TV  (isotropic, penalises all gradients → suppresses noise)
      - l0_loss  : soft-L0 (penalises small *non-zero* gradients → polarises edges)

    Inputs
    ------
    coords : (N, 2), float32, on GPU, normalised [0,1] in (y, x) order.
             Reuses target_coords already resident on GPU — zero extra CPU→GPU cost.
    grad_sample_size : number of points to subsample from coords.
    fd_eps  : finite-difference step in normalised coordinate units (~1e-3).
    l0_beta : temperature for soft-L0 sigmoid (higher → harder threshold).

    Returns
    -------
    (tv_loss, l0_loss) — both scalar tensors with gradients.

    Gradient estimation cost
    ------------------------
    3 model forwards over `grad_sample_size` points.
    """
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

    # ── 3 compact model forwards ──────────────────────────────────────────
    val_c = _forward(pts)
    val_y = _forward(pts_y)
    val_x = _forward(pts_x)

    dy = (val_y - val_c) / fd_eps
    dx = (val_x - val_c) / fd_eps

    # ── TV loss: L1 on each gradient component (isotropic L1-TV) ─────────
    # Minimising this suppresses noise-driven small gradients uniformly.
    tv_loss = (dy.abs() + dx.abs()).mean()

    # ── Soft-L0 loss: sigmoid approximation of L0 gradient sparsity ───────
    # sigmoid(β·‖∇‖ − 1) ≈ 0 when ‖∇‖ ≈ 0  (flat region, no penalty)
    #                      ≈ 1 when ‖∇‖ >> 1/β  (large gradient, no penalty)
    #                      ≈ 0.5 at ‖∇‖ = 1/β  (penalises the "mushy middle")
    # Minimising this drives gradients toward either 0 or large values —
    # exactly the polarisation needed for sharp edges.
    grad_mag = torch.sqrt(dy ** 2 + dx ** 2 + 1e-8)
    l0_loss = torch.sigmoid(l0_beta * grad_mag - 1.0).mean()

    return tv_loss, l0_loss


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
    lambda_l0: float = 0.0,
    grad_sample_size: int = 50_000,
    fd_eps: float = 1e-3,
    l0_beta: float = 10.0,
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

    pred_flat, _ = model(coords_for_model, variance=None, stochastic_alpha=stochastic_alpha)
    pred_samples = pred_flat.view(num_pixels, num_mc_samples)
    simulated_values = pred_samples.mean(dim=1)

    data_loss = F.mse_loss(simulated_values.float(), target_values.float())
    total_loss = data_loss * 100

    result = {
        "reconstruction_loss": data_loss * 100,
        "tv_loss": torch.zeros(1, device=device),
        "l0_loss": torch.zeros(1, device=device),
        "total_loss": total_loss,
    }

    # ── Sharpness regularisation (TV + soft-L0) ───────────────────────────
    # Both terms reuse target_coords already on GPU — no extra data transfer.
    # Cost: 3 model forwards over grad_sample_size points.
    if lambda_tv > 0.0 or lambda_l0 > 0.0:
        tv_loss, l0_loss = compute_gradient_losses(
            model=model,
            coords=target_coords,
            grad_sample_size=grad_sample_size,
            fd_eps=fd_eps,
            l0_beta=l0_beta,
            stochastic_alpha=stochastic_alpha,
        )

        if lambda_tv > 0.0:
            total_loss = total_loss + lambda_tv * tv_loss
            result["tv_loss"] = tv_loss

        if lambda_l0 > 0.0:
            total_loss = total_loss + lambda_l0 * l0_loss
            result["l0_loss"] = l0_loss

    result["total_loss"] = total_loss
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/temp/workspace/non-blind-deconv/1_P1MouseHeart_LSM_3.2x_2um_Angle0.tif")
    parser.add_argument("--psf_path", type=str, default="/workspace/temp/workspace/psf_t0_v0.tif",
                        help="Path to discrete 2D PSF file (image or .npy). "
                             "Required for 'discrete' mode or 'gmm' without --gmm_checkpoint.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/lsm_mouse_heart_constantLR.pth")
    parser.add_argument("--logdir", type=str, default="../runs/lsm_mouse_heart_constantLR")
    parser.add_argument("--num_mc_samples", type=int, default=100, help="Number of PSF samples per pixel")
    parser.add_argument("--num_pixels_per_step", type=int, default=180000, help="Number of pixels per step")
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
                        help="PSF sampling mode: 'discrete' or 'gmm'")
    parser.add_argument("--gmm_checkpoint", type=str, default=None,
                        help="[gmm mode] Path to a pre-fitted GMMPsf .pkl file (from gmm_psf.py).")

    # ── Sharpness regularisation ──────────────────────────────────────────
    parser.add_argument("--lambda_tv", type=float, default=1e-5,
                        help="Weight for L1-TV loss. Suppresses noise-driven small gradients. "
                             "Suggested starting range: 1e-4 ~ 1e-2. Default 0 (disabled).")
    parser.add_argument("--lambda_l0", type=float, default=1e-3,
                        help="Weight for soft-L0 gradient loss. Polarises gradients toward "
                             "0-or-large, sharpening edges. Suggested range: 1e-3 ~ 1e-1. "
                             "Default 0 (disabled).")
    parser.add_argument("--grad_sample_size", type=int, default=50_000,
                        help="Points subsampled from target_coords for gradient estimation.")
    parser.add_argument("--fd_eps", type=float, default=1e-3,
                        help="Finite-difference step size in normalised [0,1] coordinates. "
                            "~1e-3 is roughly one pixel for a 1000-pixel axis.")
    parser.add_argument("--l0_beta", type=float, default=10.0,
                        help="Temperature for soft-L0 sigmoid. Higher = harder threshold. "
                             "Threshold sits at grad_mag = 1/beta (default: 0.1 normalised units).")
    parser.add_argument("--reg_warmup_fraction", type=float, default=0.3,
                        help="Fraction of total steps before regularisation is switched on. "
                             "E.g. 0.3 means TV+L0 start at 30%% of training. Default 0 (always on).")

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

    print(f"Image loaded: shape={image_norm.shape}, dtype={image_norm.dtype}")
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
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=args.logdir)

    # ── Regularisation config summary ────────────────────────────────────
    reg_enabled = args.lambda_tv > 0.0 or args.lambda_l0 > 0.0
    reg_warmup_step = int(args.steps * args.reg_warmup_fraction)
    if reg_enabled:
        print(f"\nSharpness regularisation enabled:")
        print(f"  lambda_tv={args.lambda_tv}  lambda_l0={args.lambda_l0}")
        print(f"  grad_sample_size={args.grad_sample_size}  fd_eps={args.fd_eps}  l0_beta={args.l0_beta}")
        if reg_warmup_step > 0:
            print(f"  Warm-up: regularisation activates at step {reg_warmup_step} "
                  f"({args.reg_warmup_fraction*100:.0f}% of training)")
    else:
        print("\nSharpness regularisation disabled (lambda_tv=0, lambda_l0=0). "
              "Pass --lambda_tv / --lambda_l0 to enable.")

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
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
    )
    inv_shape = dataset.inv_shape  # keep a reference for the training step

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

        # Disable regularisation until warm-up step is reached
        step_lambda_tv = args.lambda_tv if step >= reg_warmup_step else 0.0
        step_lambda_l0 = args.lambda_l0 if step >= reg_warmup_step else 0.0

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
            lambda_l0=step_lambda_l0,
            grad_sample_size=args.grad_sample_size,
            fd_eps=args.fd_eps,
            l0_beta=args.l0_beta,
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
        if reg_enabled and step >= reg_warmup_step:
            if args.lambda_tv > 0.0:
                postfix_dict['tv'] = f'{loss_dict["tv_loss"].item():.4f}'
            if args.lambda_l0 > 0.0:
                postfix_dict['l0'] = f'{loss_dict["l0_loss"].item():.4f}'
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