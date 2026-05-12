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
    def __init__(self, image_tensor, num_pixels_per_step, num_batches):
        self.image = image_tensor
        self.num_pixels = num_pixels_per_step
        self.num_batches = num_batches
        h, w = image_tensor.shape
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/workspace/temp/Deblur-INR/results/im05.png")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="../checkpoints/gt_im05.pth")
    parser.add_argument("--logdir", type=str, default="../runs/gt_im05")
    parser.add_argument("--num_pixels_per_step", type=int, default=180000)
    parser.add_argument("--progressive_steps", type=int, default=300)
    # Encoder
    parser.add_argument("--num_levels", type=int, default=21)
    parser.add_argument("--level_dim", type=int, default=2)
    parser.add_argument("--base_resolution", type=int, default=16)
    parser.add_argument("--log2_hashmap_size", type=int, default=24)
    parser.add_argument("--desired_resolution", type=int, default=1600)
    # Decoder
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
    image_norm = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        image_norm /= float(np.iinfo(image.dtype).max)
    elif image_norm.max() > 1.0:
        image_norm /= image_norm.max()
    print(f"Image loaded: shape={image_norm.shape}")

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
        n_input_dims=2,
        learn_variance=False,
    ).to(device)
    model.train()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = ImageDataset(image_norm, args.num_pixels_per_step, args.steps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.logdir)

    n_levels = args.num_levels
    initial_levels = 4
    steps_per_level = args.progressive_steps // max(n_levels - initial_levels, 1)
    last_level = -1

    pbar = tqdm(total=args.steps, desc="Training", dynamic_ncols=True)
    for step, (coords, values) in enumerate(dataloader):
        coords = coords.squeeze(0).to(device)   # (P, 2)  [y_norm, x_norm]
        values = values.squeeze(0).to(device)   # (P,)

        current_level = min(initial_levels + step // steps_per_level, n_levels)
        if current_level != last_level:
            model.set_max_level(current_level)
            last_level = current_level

        # model expects (x_norm, y_norm)
        xy = torch.stack([coords[:, 1], coords[:, 0]], dim=-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred, _ = model(xy, variance=None, stochastic_alpha=0.0)
            loss = F.mse_loss(pred.squeeze(-1), values) * 100

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    pbar.close()
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
    }, args.save_path)
    print(f"Saved model to {args.save_path}")
    writer.close()


if __name__ == "__main__":
    main()
