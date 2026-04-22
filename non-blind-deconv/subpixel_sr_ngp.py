import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from models.instantngp import InstantNGPModel


class SubpixelShiftDataset(Dataset):
    def __init__(
        self,
        stack: torch.Tensor,  # Shape: [N_frames, H, W]
        base_shift_pixels: float,  # Shift per frame in pixel units
        num_pixels_per_step: int,
        num_batches: int,
        n_samples_per_pixel: int = 16,  # Number of MC samples per pixel
        fg_ratio: float = 0.8,  # Ratio of foreground (non-zero) samples
    ):
        self.stack = stack  # [N, H, W]
        self.n_frames, self.H, self.W = stack.shape
        self.base_shift_pixels = base_shift_pixels
        self.num_pixels_per_step = num_pixels_per_step
        self.num_batches = num_batches
        self.n_samples_per_pixel = n_samples_per_pixel
        self.fg_ratio = fg_ratio
        
        # Precompute shifts for each frame (diagonal shift: same for x and y)
        self.shifts = torch.tensor(
            [i * base_shift_pixels for i in range(self.n_frames)],
            dtype=torch.float32
        )
        
        # Precompute nonzero (foreground) pixel indices for biased sampling
        # nz: [M, 3] where each row is (frame_idx, y, x)
        self.nz = torch.nonzero(stack > 0, as_tuple=False)
        self.n_fg_pixels = self.nz.shape[0]
        
        # Max shift determines the extended coordinate range
        self.max_shift = self.shifts[-1].item() if self.n_frames > 1 else 0.0
        
        # Normalization factors for coordinates
        # The high-res coordinate space spans [0, W-1 + max_shift] x [0, H-1 + max_shift]
        self.inv_H = 1.0 / (self.H - 1 + self.max_shift)
        self.inv_W = 1.0 / (self.W - 1 + self.max_shift)
        
        print(f"SubpixelShiftDataset initialized:")
        print(f"  Frames: {self.n_frames}, Size: {self.H}x{self.W}")
        print(f"  Base shift: {base_shift_pixels:.6f} pixels/frame")
        print(f"  Total shift range: 0 to {self.max_shift:.4f} pixels")
        print(f"  Effective high-res range: [0, {self.W - 1 + self.max_shift:.4f}] x [0, {self.H - 1 + self.max_shift:.4f}]")
        print(f"  MC samples per pixel: {n_samples_per_pixel}")
        print(f"  Foreground ratio: {fg_ratio:.1%}, FG pixels: {self.n_fg_pixels:,}")
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        N = self.num_pixels_per_step
        n_fg = int(N * self.fg_ratio)
        n_bg = N - n_fg
        
        # 1) Foreground samples: sample from non-zero pixels
        fg_idx = torch.randint(0, self.n_fg_pixels, (n_fg,))
        fg_samples = self.nz[fg_idx]  # [n_fg, 3] -> (frame, y, x)
        fg_frame = fg_samples[:, 0]
        fg_py = fg_samples[:, 1]
        fg_px = fg_samples[:, 2]
        fg_values = self.stack[fg_frame, fg_py, fg_px]
        
        # 2) Background/uniform samples: sample uniformly from all pixels
        bg_frame = torch.randint(0, self.n_frames, (n_bg,))
        bg_py = torch.randint(0, self.H, (n_bg,))
        bg_px = torch.randint(0, self.W, (n_bg,))
        bg_values = self.stack[bg_frame, bg_py, bg_px]
        
        # 3) Concatenate foreground and background samples
        frame_indices = torch.cat([fg_frame, bg_frame], dim=0)
        py = torch.cat([fg_py, bg_py], dim=0)
        px = torch.cat([fg_px, bg_px], dim=0)
        pixel_values = torch.cat([fg_values, bg_values], dim=0)
        
        # Compute high-resolution coordinates with subpixel shifts
        # For each frame i, the shift is shifts[i] pixels in both x and y (diagonal)
        shifts_for_samples = self.shifts[frame_indices]
        
        # Monte Carlo sampling within each pixel
        # For each pixel, sample n_samples_per_pixel points uniformly within the pixel area
        # Pixel (px, py) with shift covers the area:
        #   x: [px + shift - 0.5, px + shift + 0.5]
        #   y: [py + shift - 0.5, py + shift + 0.5]
        
        # Generate uniform random offsets in [-0.5, 0.5] for each pixel and each sample
        # Shape: [num_pixels_per_step, n_samples_per_pixel]
        offset_x = torch.rand(self.num_pixels_per_step, self.n_samples_per_pixel) - 0.5
        offset_y = torch.rand(self.num_pixels_per_step, self.n_samples_per_pixel) - 0.5
        
        # Compute sample coordinates for each pixel
        # Shape: [num_pixels_per_step, n_samples_per_pixel]
        sample_x = (px.float().unsqueeze(1) - shifts_for_samples.unsqueeze(1) + offset_x) * self.inv_W
        sample_y = (py.float().unsqueeze(1) - shifts_for_samples.unsqueeze(1) + offset_y) * self.inv_H
        
        sample_x = torch.clamp(sample_x, 0.0, 1.0)
        sample_y = torch.clamp(sample_y, 0.0, 1.0)
        
        # Flatten samples: [num_pixels_per_step * n_samples_per_pixel, 2]
        coords = torch.stack([sample_x.flatten(), sample_y.flatten()], dim=1)
        
        return {
            'coords': coords,  # [N * n_samples_per_pixel, 2] normalized high-res coordinates
            'values': pixel_values.float(),  # [N] pixel values (one per pixel)
            'n_samples_per_pixel': self.n_samples_per_pixel,
        }


def train_step(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> dict:
    """
    Training step with per-pixel MC averaging.
    For each pixel, we have n_samples_per_pixel samples.
    We predict values at all samples, average within each pixel, then compute loss.
    This implements: (1/n) Σ f(pᵢ) ≈ I_{i,j}
    """
    coords = batch['coords'].to(device, non_blocking=True)
    target_values = batch['values'].to(device, non_blocking=True)
    n_samples_per_pixel = batch['n_samples_per_pixel']
    
    # Forward pass - predict at all sample points
    predictions, _ = model(coords, variance=None)
    predictions = predictions.view(-1)
    
    # Reshape and average within each pixel
    # predictions shape: [num_pixels * n_samples_per_pixel]
    # Reshape to [num_pixels, n_samples_per_pixel] and average
    num_pixels = target_values.shape[0]
    predictions = predictions.view(num_pixels, n_samples_per_pixel)
    pixel_predictions = predictions.mean(dim=1)  # [num_pixels]
    
    # Compute MSE loss between averaged predictions and pixel values
    loss = F.mse_loss(pixel_predictions, target_values) * 100
    
    return {"reconstruction_loss": loss, "total_loss": loss}


def infer_super_resolution(
    model: nn.Module,
    original_shape: tuple,  # (H, W)
    upscale_factor: int,
    device: torch.device,
    max_shift: float = 0.0,  # Max shift in pixels, needed for consistent normalization
    batch_size: int = 100000,
) -> np.ndarray:
    """
    Infer the super-resolved image at the target resolution.
    The coordinate normalization must match the training normalization.
    """
    H, W = original_shape
    sr_H, sr_W = H * upscale_factor, W * upscale_factor
    
    print(f"Inferring super-resolution image: {H}x{W} -> {sr_H}x{sr_W}")
    print(f"  Using max_shift={max_shift:.4f} for coordinate normalization")
    
    model.eval()
    
    # The effective high-res pixel range is [0, W-1 + max_shift] x [0, H-1 + max_shift]
    # We want to sample the original image region [0, W-1] x [0, H-1] at higher resolution
    # Normalized coordinates for the original region:
    #   x_norm_max = (W - 1) / (W - 1 + max_shift)
    #   y_norm_max = (H - 1) / (H - 1 + max_shift)
    x_norm_max = (W - 1) / (W - 1 + max_shift)
    y_norm_max = (H - 1) / (H - 1 + max_shift)
    
    # Create normalized coordinate grid for the super-resolved image
    # Sample from [0, x_norm_max] x [0, y_norm_max] to get the original image region
    y_coords = torch.linspace(0, y_norm_max, sr_H)
    x_coords = torch.linspace(0, x_norm_max, sr_W)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten and stack as (x, y) for model input
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [sr_H*sr_W, 2]
    
    # Infer in batches to avoid OOM
    predictions = []
    n_coords = coords.shape[0]
    
    with torch.no_grad():
        for start in tqdm(range(0, n_coords, batch_size), desc="Inference"):
            end = min(start + batch_size, n_coords)
            batch_coords = coords[start:end].to(device)
            pred, _ = model(batch_coords, variance=None)
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions, dim=0).view(sr_H, sr_W).numpy()
    
    model.train()
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Subpixel Shifting Super-Resolution with InstantNGP")
    
    # Data parameters
    parser.add_argument("--stack_path", type=str, default="/workspace/Target_3.2Mag_1umX1umY_Bin4__Stack(Thresholded).tif")
    parser.add_argument("--base_shift_nm", type=float, default=1000)
    parser.add_argument("--pixel_size_um", type=float, default=8.0)
    parser.add_argument("--upscale_factor", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=2, help="Maximum number of frames to use from the stack")
    parser.add_argument("--frame_step", type=int, default=4, help="Step size for frame selection (1=every frame, 2=every other frame for 1/4 shift)")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_pixels_per_step", type=int, default=100000)
    parser.add_argument("--n_samples_per_pixel", type=int, default=32, help="Number of MC samples per pixel for averaging")
    parser.add_argument("--fg_ratio", type=float, default=0.8, help="Ratio of foreground (non-zero) samples vs uniform")
    
    # Output parameters
    parser.add_argument("--save_path", type=str, default="../checkpoints/subpixel_sr_16.pth")
    parser.add_argument("--output_path", type=str, default="/workspace/utils/2d/data/model_sr_8192_16.png")
    parser.add_argument("--logdir", type=str, default="../runs/subpixel_sr_16")
    
    # Model configuration
    parser.add_argument("--num_levels", type=int, default=20)
    parser.add_argument("--level_dim", type=int, default=2)
    parser.add_argument("--base_resolution", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=23)
    parser.add_argument("--desired_resolution", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    
    # Progressive training
    parser.add_argument("--progressive_steps", type=int, default=500)
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load TIFF stack
    print(f"Loading stack from {args.stack_path}")
    stack_np = tifffile.imread(args.stack_path).astype(np.float32)
    print(f"Stack shape: {stack_np.shape}")
    
    # Assume multiple frames
    assert stack_np.ndim == 3, "Stack must be 3D (H, W, C)"

    # Select frames with step and limit
    # frame_step=1: every frame, frame_step=2: every other frame (for 1/4 shift amount)
    stack_np = stack_np[::args.frame_step]
    if args.max_frames is not None and stack_np.shape[0] > args.max_frames:
        stack_np = stack_np[:args.max_frames]
    print(f"Selected frames with step={args.frame_step}, new shape: {stack_np.shape}")
    
    # Normalize to [0, 1]
    stack_max = stack_np.max()
    stack_np = stack_np / stack_max
    stack = torch.from_numpy(stack_np)
    
    # Calculate base shift in pixel units
    # base_shift_nm is the physical shift, pixel_size_um is the pixel size
    # Convert: nm -> um -> pixels
    # Effective shift per selected frame = base_shift * frame_step
    base_shift_pixels = args.base_shift_nm * args.frame_step / (args.pixel_size_um * 1000)
    print(f"Base shift: {args.base_shift_nm} nm * step {args.frame_step} = {base_shift_pixels:.6f} pixels/frame")
    
    n_frames, H, W = stack.shape
    
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
    
    model = InstantNGPModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_input_dims=2,  # 2D image
        learn_variance=False,
    ).to(device)
    
    model.train()
    
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"Total parameters: {total_params:,}")
    
    dataset = SubpixelShiftDataset(
        stack=stack,
        base_shift_pixels=base_shift_pixels,
        num_pixels_per_step=args.num_pixels_per_step,
        num_batches=args.steps,
        n_samples_per_pixel=args.n_samples_per_pixel,
        fg_ratio=args.fg_ratio,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    gamma = 10 ** (-1 / args.steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # GradScaler for FP16 training
    scaler = GradScaler('cuda')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)
    
    # Progressive training configuration
    n_levels = encoder_config["n_levels"]
    initial_levels = 4
    steps_per_level = args.progressive_steps // max(1, n_levels - initial_levels)
    print(f"Progressive training: starting with {initial_levels} levels, unlocking 1 level every {steps_per_level} steps")
    
    # Training loop
    data_iter = iter(dataloader)
    last_set_level = -1
    
    pbar = tqdm(
        total=args.steps,
        desc="Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    for step in range(args.steps):
        # Progressive training level update
        current_level = min(initial_levels + (step // steps_per_level), n_levels)
        if current_level != last_set_level:
            model.set_max_level(current_level)
            last_set_level = current_level
            if step > 0 and current_level <= n_levels:
                print(f"\nStep {step}: Unlocked level {current_level}/{n_levels}")
        
        batch = next(data_iter)
        batch = {k: v.squeeze(0) for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        
        loss_dict = train_step(model=model, batch=batch, device=device)
        loss = loss_dict["total_loss"]
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        writer.add_scalar("train/reconstruction_loss", loss.item(), step)
        writer.add_scalar("train/learning_rate", current_lr, step)
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'lr': f'{current_lr:.2e}',
        })
    
    pbar.close()
    
    # Save model checkpoint
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
        'args': vars(args),
    }
    torch.save(save_obj, args.save_path)
    print(f"\nSaved model to {args.save_path}")
    
    # Calculate max_shift for consistent normalization
    max_shift = (n_frames - 1) * base_shift_pixels
    
    # Infer super-resolved image
    sr_image = infer_super_resolution(
        model=model,
        original_shape=(H, W),
        upscale_factor=args.upscale_factor,
        device=device,
        max_shift=max_shift,
    )
    
    # Rescale to original intensity range
    sr_image = np.clip(sr_image, 0, 1)
    sr_image_tif = sr_image * stack_max
    sr_image_png = (sr_image * 255).astype(np.uint8)
    print("png range:", sr_image_png.min(), sr_image_png.max())
    print("tif range:", sr_image_tif.min(), sr_image_tif.max())
    
    # Determine output paths
    #if args.output_path is None:
    #    output_path_png = "/workspace/utils/2d/data/model_sr_8192_16.png"
    #else:
    #    output_path_png = args.output_path

    #cv2.imwrite(output_path_png, sr_image_png)
    #tifffile.imwrite("/workspace/utils/2d/data/model_sr_8192_16.tif", sr_image_tif.astype(np.uint16))

    # save raw tif
    tifffile.imwrite("/workspace/utils/2d/data/model_sr_8192_2_raw.tif", sr_image.astype(np.float32))
   # print(f"Saved superres to {output_path_png}")
    
    writer.close()


if __name__ == "__main__":
    main()
