import numpy as np
import torch

def _random_unit_vectors(n: int) -> np.ndarray:
    """Sample n random 3D unit vectors uniformly on S^2."""
    v = np.random.normal(size=(n, 3)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

import numpy as np
import torch

def _random_unit_vectors(n: int) -> np.ndarray:
    """Sample n random 3D unit vectors uniformly on S^2."""
    v = np.random.normal(size=(n, 3)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def sample_sdf_coords(
    mesh,
    n,
    volume_shape,  # (D, H, W) = (z, y, x)
    device="cpu",
    near_voxel_min=1.0,
    near_voxel_max=2.0,
    margin_voxels=2,     # 防止贴到边界面
    max_tries=50,        # 重采次数上限
):
    # surface:near_surface:uniform = 2:2:1
    n_uniform = n // 5
    n_near_surface = 2 * n // 5
    n_surface = n - n_uniform - n_near_surface

    vz, vy, vx = volume_shape

    # voxel size in normalized coord (x,y,z)
    voxel = np.array([1.0/(vx-1), 1.0/(vy-1), 1.0/(vz-1)], dtype=np.float32)

    # normalized margin box [lo, hi]
    m = margin_voxels * voxel
    lo, hi = m, 1.0 - m

    # Uniform in inner box
    uniform_coords = lo + np.random.rand(n_uniform, 3).astype(np.float32) * (hi - lo)

    # Surface
    surface_coords = mesh.sample(n_surface).astype(np.float32)

    # Near-surface ring sampling (1-2 voxels)
    near = []
    remain = n_near_surface
    tries = 0
    while remain > 0 and tries < max_tries:
        tries += 1

        # oversample to reduce loop cost
        batch = int(remain * 2.0) + 64
        base = mesh.sample(batch).astype(np.float32)

        dirs = _random_unit_vectors(batch)
        # sample radius in [1,2] voxels (isotropic in voxel units)
        r_vox = np.random.uniform(near_voxel_min, near_voxel_max, size=(batch, 1)).astype(np.float32)

        # convert voxel radius to normalized coord (per-axis scale)
        offset = dirs * (r_vox * voxel)  # (batch,3)

        cand = base + offset

        # reject out-of-range (use margin box, NOT clip)
        mask = np.all((cand >= lo) & (cand <= hi), axis=1)
        cand = cand[mask]

        if len(cand) > 0:
            take = min(remain, len(cand))
            near.append(cand[:take])
            remain -= take

    if remain > 0:
        # fallback: if mesh is too close to boundary, fill the rest with surface points (still safe)
        base = mesh.sample(remain).astype(np.float32)
        base = np.clip(base, lo, hi)
        near.append(base)

    near_surface_coords = np.concatenate(near, axis=0) if len(near) else np.zeros((0,3), np.float32)

    # Combine (all already in [lo,hi])
    all_coords = np.concatenate([uniform_coords, surface_coords, near_surface_coords], axis=0)

    # Convert to indices (mesh uses x,y,z; volume expects z,y,x)
    z_indices = (all_coords[:, 2] * (vz - 1)).astype(np.int64)
    y_indices = (all_coords[:, 1] * (vy - 1)).astype(np.int64)
    x_indices = (all_coords[:, 0] * (vx - 1)).astype(np.int64)

    # final safety clip
    z_indices = np.clip(z_indices, 0, vz - 1)
    y_indices = np.clip(y_indices, 0, vy - 1)
    x_indices = np.clip(x_indices, 0, vx - 1)

    coords = torch.from_numpy(np.stack([z_indices, y_indices, x_indices], axis=1))
    return coords.to(device).long()