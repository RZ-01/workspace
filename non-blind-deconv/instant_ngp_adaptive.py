import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.instantngp import InstantNGPTorchModel


class ImageDataset(Dataset):
    def __init__(self, image: np.ndarray, num_pixels_per_step: int, num_batches: int):
        self.image = image
        self.num_pixels = num_pixels_per_step
        self.num_batches = num_batches
        h, w = image.shape
        self.inv_shape = torch.tensor([1.0 / max(h - 1, 1), 1.0 / max(w - 1, 1)], dtype=torch.float32)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        h, w = self.image.shape
        y = torch.randint(0, h, (self.num_pixels,))
        x = torch.randint(0, w, (self.num_pixels,))
        coords = torch.stack([y, x], dim=1).float() * self.inv_shape
        values = torch.from_numpy(np.ascontiguousarray(self.image[y.numpy(), x.numpy()], dtype=np.float32))
        return coords, values


def generate_offsets_on_gpu(n: int, discrete_psf: torch.Tensor, device: torch.device) -> torch.Tensor:
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


def build_model(encoder_config, decoder_config, device):
    model = InstantNGPTorchModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    return model


def build_encoder_config(args):
    return {
        "otype": "HashGrid",
        "n_levels": args.num_levels,
        "n_features_per_level": args.level_dim,
        "log2_hashmap_size": args.log2_hashmap_size,
        "base_resolution": args.base_resolution,
        "per_level_scale": np.exp(
            (np.log(args.desired_resolution) - np.log(args.base_resolution)) / (args.num_levels - 1)
        ),
    }


def build_decoder_config(args):
    return {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": args.hidden_dim,
        "n_hidden_layers": args.num_layers,
    }


def load_gray_image_v2(path: str) -> np.ndarray:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Failed to read image: {path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    norm = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        norm /= float(np.iinfo(raw.dtype).max)
    elif norm.max() > 1.0:
        norm /= norm.max()
    return norm


def train_direct(model, image: np.ndarray, args, writer: SummaryWriter, device, tag: str = "phase1"):
    """Phase 1: directly fit the image (no PSF)."""
    dataset = ImageDataset(image, args.num_pixels_per_step, args.warmup_steps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    inv_shape = dataset.inv_shape

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup_steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')

    initial_levels = 4
    steps_per_level = args.progressive_steps // max(args.num_levels - initial_levels, 1)
    last_level = -1

    pbar = tqdm(total=args.warmup_steps, desc=f"[{tag}] direct fit", dynamic_ncols=True)
    for step, (coords, values) in enumerate(dataloader):
        coords = coords.squeeze(0).to(device)
        values = values.squeeze(0).to(device)

        current_level = min(initial_levels + step // steps_per_level, args.num_levels)
        if current_level != last_level:
            model.set_max_level(current_level)
            last_level = current_level

        xy = torch.stack([coords[:, 1], coords[:, 0]], dim=-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred, _ = model(xy, variance=None, stochastic_alpha=0.0)
        loss = F.mse_loss(pred.squeeze(-1).float(), values.float()) * 100

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step = step
        writer.add_scalar(f"{tag}/loss", loss.item(), global_step)
        writer.add_scalar(f"{tag}/lr", optimizer.param_groups[0]['lr'], global_step)
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.3e}'})

    pbar.close()
    return inv_shape

def save_warmup(model, encoder_config, decoder_config, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_config':   encoder_config,
        'decoder_config':   decoder_config,
    }, path)
    print(f"Phase 1 checkpoint saved to {path}")


def train_mc(model, image: np.ndarray, discrete_psf: torch.Tensor, inv_shape: torch.Tensor,
             args, writer: SummaryWriter, device, tag: str = "phase2"):
    """Phase 2: MC deconvolution starting from the warm-started model."""
    dataset = ImageDataset(image, args.num_pixels_per_step, args.steps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')

    sp_decay_steps = int(args.steps * args.sp_decay_fraction)
    model.set_max_level(args.num_levels)   # full resolution for deconv phase

    pbar = tqdm(total=args.steps, desc=f"[{tag}] MC deconv", dynamic_ncols=True)
    for step, (coords, values) in enumerate(dataloader):
        coords = coords.squeeze(0).to(device)
        values = values.squeeze(0).to(device)

        # stochastic preconditioning schedule
        if step < sp_decay_steps:
            alpha = args.sp_alpha_init * np.exp(-5.0 * step / sp_decay_steps)
        else:
            alpha = 0.0

        num_pixels = coords.shape[0]
        sampling_budget = num_pixels * args.num_mc_samples
        inv_shape_gpu = inv_shape.to(device)

        offsets = generate_offsets_on_gpu(sampling_budget, discrete_psf, device)  # (N, 2)
        offsets = (offsets * inv_shape_gpu).view(num_pixels, args.num_mc_samples, 2)

        source_coords = torch.clamp(coords.unsqueeze(1) + offsets, 0.0, 1.0)
        source_flat = source_coords.view(-1, 2)
        xy = torch.stack([source_flat[:, 1], source_flat[:, 0]], dim=-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred_flat, _ = model(xy, variance=None, stochastic_alpha=alpha)
            simulated = pred_flat.view(num_pixels, args.num_mc_samples).mean(dim=1)
        loss = F.mse_loss(simulated.float(), values.float()) * 100

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        writer.add_scalar(f"{tag}/loss", loss.item(), step)
        writer.add_scalar(f"{tag}/lr", optimizer.param_groups[0]['lr'], step)
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.3e}'})

    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",        type=str, default="/workspace/temp/Deblur-INR/datasets/lai/im05_ker04_fft_circular.png")
    parser.add_argument("--psf_path",          type=str, default="/workspace/temp/Deblur-INR/results/ker04_truth.png")
    parser.add_argument("--warmup_save_path",  type=str, default="../checkpoints/warmup_im05_ker04.pth",
                        help="Where to save the Phase 1 checkpoint (and load it for Phase 2).")
    parser.add_argument("--save_path",         type=str, default="../checkpoints/deconv_im05_ker04.pth")
    parser.add_argument("--logdir",            type=str, default="../runs/im05_ker04_warmup")

    # Phase 1 — direct fit
    parser.add_argument("--warmup_steps",        type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=1e-2)
    parser.add_argument("--skip_warmup",         action="store_true",
                        help="Skip Phase 1 and load warmup_save_path directly for Phase 2.")

    # Phase 2 — MC deconv
    parser.add_argument("--steps",               type=int,   default=1500)
    parser.add_argument("--num_mc_samples",      type=int,   default=300)
    parser.add_argument("--sp_alpha_init",       type=float, default=0.03)
    parser.add_argument("--sp_decay_fraction",   type=float, default=0.33)

    # Shared
    parser.add_argument("--num_pixels_per_step", type=int,   default=180000)
    parser.add_argument("--progressive_steps",   type=int,   default=300)

    # Encoder
    parser.add_argument("--num_levels",          type=int,   default=21)
    parser.add_argument("--level_dim",           type=int,   default=2)
    parser.add_argument("--base_resolution",     type=int,   default=16)
    parser.add_argument("--log2_hashmap_size",   type=int,   default=24)
    parser.add_argument("--desired_resolution",  type=int,   default=1600)

    # Decoder
    parser.add_argument("--hidden_dim",          type=int,   default=64)
    parser.add_argument("--num_layers",          type=int,   default=2)

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_norm = load_gray_image_v2(args.image_path)
    print(f"Image loaded: shape={image_norm.shape}")

    if args.psf_path.endswith(".npy"):
        psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        psf_np = cv2.imread(args.psf_path, cv2.IMREAD_UNCHANGED)
        if psf_np is None:
            raise ValueError(f"Unreadable PSF: {args.psf_path}")
        if psf_np.ndim == 3:
            psf_np = cv2.cvtColor(psf_np, cv2.COLOR_BGR2GRAY)
        psf_np = psf_np.astype(np.float32)
    psf_np /= psf_np.sum()
    discrete_psf = torch.flip(torch.from_numpy(psf_np).float().to(device), [0, 1])
    print(f"PSF loaded: shape={discrete_psf.shape}")

    encoder_config = build_encoder_config(args)
    decoder_config = build_decoder_config(args)
    writer = SummaryWriter(log_dir=args.logdir)

    # ── Phase 1: warm-start ───────────────────────────────────────────────
    if not args.skip_warmup:
        print("\n=== Phase 1: warm-start (direct fit to blurry image) ===")
        model = build_model(encoder_config, decoder_config, device)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        model.train()
        inv_shape = train_direct(model, image_norm, args, writer, device, tag="phase1")
        save_warmup(model, encoder_config, decoder_config, args.warmup_save_path)
        del model
    else:
        # derive inv_shape from image without building a dataset
        h, w = image_norm.shape
        inv_shape = torch.tensor([1.0 / max(h - 1, 1), 1.0 / max(w - 1, 1)], dtype=torch.float32)

    # ── Phase 2: load pretrained init, then MC deconv ─────────────────────
    print("\n=== Phase 2: load warm-start checkpoint, MC deconvolution ===")
    ckpt = torch.load(args.warmup_save_path, map_location=device, weights_only=False)
    model = build_model(ckpt['encoder_config'], ckpt['decoder_config'], device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded Phase 1 checkpoint from {args.warmup_save_path}")
    model.train()
    train_mc(model, image_norm, discrete_psf, inv_shape, args, writer, device, tag="phase2")

    # ── Save final model ──────────────────────────────────────────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_config':   ckpt['encoder_config'],
        'decoder_config':   ckpt['decoder_config'],
    }, args.save_path)
    print(f"\nSaved final model to {args.save_path}")
    writer.close()


if __name__ == "__main__":
    main()
