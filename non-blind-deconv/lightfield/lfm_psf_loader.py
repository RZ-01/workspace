"""
LFM 5D PSF Loader

Loads precomputed 5D PSF from HDF5 file and provides forward projection.

HDF5 Structure:
- /camera/ - System parameters (M, NA, wavelength, etc.)
- /resolution/ - depths, sensor_resolution, voxel_resolution
- /lenslet_centers/ - pixel_coords, voxel_coords
- /psf_forward/depth_XXX/pattern_XXX_XXX - Forward PSF patterns
- /psf_backward/depth_XXX/pattern_XXX_XXX - Backward PSF patterns
"""

import h5py
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


class LFM5DPSF:
    
    def __init__(self, h5_path: str, device: str = 'cuda'):
        """
        Args:
            h5_path: Path to PSF_operators.h5 file
            device: PyTorch device
        """
        self.h5_path = h5_path
        self.device = device
        
        # Open HDF5 file and load metadata
        self._load_metadata()
        
    def _load_metadata(self):
        """Load camera parameters and resolution info from HDF5."""
        with h5py.File(self.h5_path, 'r') as f:
            # Camera parameters (stored as 1-element arrays)
            self.M = float(f['/camera/M'][0])
            self.NA = float(f['/camera/NA'][0])
            self.wavelength = float(f['/camera/wavelength'][0])
            self.lens_pitch = float(f['/camera/lens_pitch'][0])
            self.pixel_pitch = float(f['/camera/pixel_pitch'][0])
            self.fm = float(f['/camera/fm'][0])  # Microlens focal length in µm
            
            # Resolution info
            self.depths = f['/resolution/depths'][:]  # Array of depth values in µm
            self.num_depths = len(self.depths)
            
            # sensor_resolution and voxel_resolution are µm/pixel, not pixel counts
            self.sensor_um_per_pixel = tuple(f['/resolution/sensor_resolution'][:])
            self.voxel_um = tuple(f['/resolution/voxel_resolution'][:])
            
            # Nnum: pixels per lenslet (e.g., [15, 15])
            self.Nnum = tuple(int(x) for x in f['/resolution/Nnum'][:])
            self.TexNnum = tuple(int(x) for x in f['/resolution/TexNnum'][:])
            
            # Lenslet count from lenslet_centers shape: [2, 49, 48] -> (49, 48)
            lenslet_centers_shape = f['/lenslet_centers/pixel_coords'].shape
            self.num_lenslets_s = lenslet_centers_shape[1]  # rows
            self.num_lenslets_t = lenslet_centers_shape[2]  # cols
            self.lenslet_shape = (self.num_lenslets_s, self.num_lenslets_t)
            
            # Load lenslet centers
            self.lenslet_centers_px = f['/lenslet_centers/pixel_coords'][:]
            self.lenslet_centers_vox = f['/lenslet_centers/voxel_coords'][:]
            
            # Get pattern shape from first pattern
            depth_key = 'depth_001'  # 1-indexed
            pattern_key = list(f[f'/psf_forward/{depth_key}'].keys())[0]
            self.pattern_shape = f[f'/psf_forward/{depth_key}/{pattern_key}'].shape
        
        # Compute sensor resolution in pixels: lenslets * pixels_per_lenslet
        # This is approximate - actual sensor may be slightly larger
        self.sensor_resolution = (
            self.num_lenslets_s * self.Nnum[0],  # H
            self.num_lenslets_t * self.Nnum[1],  # W
        )
        
        # Compute voxel_resolution as volume grid dimensions (D, H, W)
        # D = number of depth planes
        # H, W = match the lenslet grid (each lenslet samples one lateral position)
        self.voxel_resolution = (
            self.num_depths,              # D (depth planes)
            self.num_lenslets_s,          # H (lateral rows)
            self.num_lenslets_t,          # W (lateral cols)
        )
        
        print(f"Loaded PSF metadata:")
        print(f"  Depths: {self.num_depths} ({self.depths[0]:.1f} to {self.depths[-1]:.1f} µm)")
        print(f"  Lenslets: {self.num_lenslets_s} x {self.num_lenslets_t}")
        print(f"  Pixels per lenslet: {self.Nnum}")
        print(f"  Sensor resolution: {self.sensor_resolution}")
        print(f"  Voxel resolution (D, H, W): {self.voxel_resolution}")
        print(f"  Pattern shape: {self.pattern_shape}")
        

    def depth_to_index(self, z_um: float) -> int:
        """Convert depth in µm to nearest index."""
        idx = np.argmin(np.abs(self.depths - z_um))
        return int(idx)
    
    def forward_project_volume(
        self,
        volume: torch.Tensor,
        mc_samples: int = 10000,
        use_shift_invariant: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Forward project a 3D volume to 2D sensor image using Monte Carlo sampling.
        
        Args:
            volume: 3D volume tensor [D, H, W] where D matches num_depths
            mc_samples: Number of (depth, lenslet) combinations to sample.
                       Recommended: 10000-50000 for good quality.
            use_shift_invariant: If True, tile 8x8 PSF patterns across all lenslets
            verbose: Print progress information
            
        Returns:
            2D sensor image [sensor_H, sensor_W]
        """
        import random
        
        D, Hv, Wv = volume.shape
        assert D == self.num_depths, f"Volume depth {D} != PSF depths {self.num_depths}"
        
        sensor_image = torch.zeros(self.sensor_resolution, device=self.device)
        psf_size = 8  # PSF covers 8x8 lenslets
        
        total_combinations = self.num_depths * self.num_lenslets_s * self.num_lenslets_t
        scale_factor = total_combinations / mc_samples
        
        if verbose:
            print(f"MC sampling {mc_samples}/{total_combinations} ({100*mc_samples/total_combinations:.1f}%)")
        
        for _ in range(mc_samples):
            d_idx = random.randint(0, self.num_depths - 1)
            s = random.randint(1, self.num_lenslets_s)
            t = random.randint(1, self.num_lenslets_t)
            
            volume_slice = volume[d_idx]
            
            if use_shift_invariant:
                s_psf = ((s - 1) % psf_size) + 1
                t_psf = ((t - 1) % psf_size) + 1
            else:
                s_psf, t_psf = s, t
            
            pattern = self.get_forward_pattern(d_idx, s_psf, t_psf)
            if pattern is not None:
                self._apply_pattern_to_sensor_scaled(
                    volume_slice, pattern, sensor_image, d_idx, s, t, scale_factor
                )
        
        return sensor_image
    
    def _apply_pattern_to_sensor(
        self,
        volume_slice: torch.Tensor,
        pattern: torch.Tensor,
        sensor_image: torch.Tensor,
        d_idx: int,
        s: int, 
        t: int
    ):
        """
        Apply PSF pattern contribution to sensor image.
        
        The pattern represents how a point source at depth d_idx, lenslet (s, t)
        spreads onto the sensor. We add the pattern weighted by the volume intensity
        at the corresponding voxel location.
        
        Args:
            volume_slice: [H_vol, W_vol] volume slice at depth d_idx
            pattern: [pattern_H, pattern_W] PSF intensity pattern
            sensor_image: [sensor_H, sensor_W] output image to accumulate into
            d_idx: depth index
            s: lenslet row (1-indexed)
            t: lenslet column (1-indexed)
        """
        # Get voxel intensity for this lenslet position
        # s, t are 1-indexed; volume_slice is 0-indexed
        vox_y = s - 1
        vox_x = t - 1
        
        if vox_y >= volume_slice.shape[0] or vox_x >= volume_slice.shape[1]:
            return
        
        voxel_intensity = volume_slice[vox_y, vox_x]
        
        if abs(voxel_intensity) < 1e-8:
            return  # Skip zero-intensity voxels
        
        # Compute lenslet center in sensor pixels from grid indices
        # Each lenslet covers Nnum pixels, so center is at (index * Nnum) + Nnum/2
        center_y = (s - 1) * self.Nnum[0] + self.Nnum[0] // 2
        center_x = (t - 1) * self.Nnum[1] + self.Nnum[1] // 2
        
        # Compute where the pattern should be placed on the sensor
        # Pattern center aligns with lenslet center
        ph, pw = pattern.shape
        half_h, half_w = ph // 2, pw // 2
        
        # Sensor region bounds
        sensor_y0 = center_y - half_h
        sensor_y1 = center_y + (ph - half_h)
        sensor_x0 = center_x - half_w  
        sensor_x1 = center_x + (pw - half_w)
        
        # Clip to sensor bounds
        sh, sw = sensor_image.shape
        src_y0 = max(0, -sensor_y0)
        src_y1 = ph - max(0, sensor_y1 - sh)
        src_x0 = max(0, -sensor_x0)
        src_x1 = pw - max(0, sensor_x1 - sw)
        
        dst_y0 = max(0, sensor_y0)
        dst_y1 = min(sh, sensor_y1)
        dst_x0 = max(0, sensor_x0)
        dst_x1 = min(sw, sensor_x1)
        
        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return  # Pattern is fully outside sensor
        
        # Add weighted pattern contribution
        sensor_image[dst_y0:dst_y1, dst_x0:dst_x1] += (
            voxel_intensity * pattern[src_y0:src_y1, src_x0:src_x1]
        )
    
    def _apply_pattern_to_sensor_scaled(
        self,
        volume_slice: torch.Tensor,
        pattern: torch.Tensor,
        sensor_image: torch.Tensor,
        d_idx: int,
        s: int, 
        t: int,
        scale_factor: float
    ):
        """
        Apply PSF pattern contribution with MC scaling factor.
        Same as _apply_pattern_to_sensor but scales the contribution
        to compensate for Monte Carlo sampling.
        """
        vox_y = s - 1
        vox_x = t - 1
        
        if vox_y >= volume_slice.shape[0] or vox_x >= volume_slice.shape[1]:
            return
        
        voxel_intensity = volume_slice[vox_y, vox_x]
        
        if abs(voxel_intensity) < 1e-8:
            return
        
        center_y = (s - 1) * self.Nnum[0] + self.Nnum[0] // 2
        center_x = (t - 1) * self.Nnum[1] + self.Nnum[1] // 2
        
        ph, pw = pattern.shape
        half_h, half_w = ph // 2, pw // 2
        
        sensor_y0 = center_y - half_h
        sensor_y1 = center_y + (ph - half_h)
        sensor_x0 = center_x - half_w  
        sensor_x1 = center_x + (pw - half_w)
        
        sh, sw = sensor_image.shape
        src_y0 = max(0, -sensor_y0)
        src_y1 = ph - max(0, sensor_y1 - sh)
        src_x0 = max(0, -sensor_x0)
        src_x1 = pw - max(0, sensor_x1 - sw)
        
        dst_y0 = max(0, sensor_y0)
        dst_y1 = min(sh, sensor_y1)
        dst_x0 = max(0, sensor_x0)
        dst_x1 = min(sw, sensor_x1)
        
        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return
        
        # Scale the contribution to compensate for MC sampling
        sensor_image[dst_y0:dst_y1, dst_x0:dst_x1] += (
            scale_factor * voxel_intensity * pattern[src_y0:src_y1, src_x0:src_x1]
        )
    
    def forward_project_points(
        self,
        coords_3d: torch.Tensor,
        intensities: torch.Tensor,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward project a set of 3D points to 2D sensor image.
        
        More efficient than full volume projection when only sparse points are active.
        
        Args:
            coords_3d: [N, 3] array of (x, y, z) coordinates in voxel space
            intensities: [N] array of intensity values
            output_shape: Output sensor shape, defaults to self.sensor_resolution
            
        Returns:
            2D sensor image
        """
        if output_shape is None:
            output_shape = self.sensor_resolution
            
        sensor_image = torch.zeros(output_shape, device=self.device)
        
        for i in range(coords_3d.shape[0]):
            x, y, z = coords_3d[i]
            intensity = intensities[i]
            
            # Find depth index
            d_idx = self.depth_to_index(z.item())
            
            # Find which lenslet this point projects to
            s, t = self._voxel_to_lenslet(x.item(), y.item())
            
            if s < 1 or s > self.num_lenslets_s or t < 1 or t > self.num_lenslets_t:
                continue
            
            pattern = self.get_forward_pattern(d_idx, s, t)
            if pattern is None:
                continue
            
            # Add weighted pattern to sensor image
            # This depends on pattern format
            self._add_point_contribution(sensor_image, pattern, intensity, s, t)
        
        return sensor_image
    
    def _voxel_to_lenslet(self, x_vox: float, y_vox: float) -> Tuple[int, int]:
        """Map voxel coordinates to lenslet indices."""
        # Use lenslet centers to find nearest lenslet
        # This is a simplified version - actual mapping depends on optics
        
        # Assuming regular grid of lenslets
        Hv, Wv = self.voxel_resolution[1], self.voxel_resolution[2]
        
        s = int((y_vox / Hv) * self.num_lenslets_s) + 1
        t = int((x_vox / Wv) * self.num_lenslets_t) + 1
        
        return s, t
    
    def _add_point_contribution(
        self,
        sensor_image: torch.Tensor,
        pattern: torch.Tensor,
        intensity: float,
        s: int,
        t: int
    ):
        """Add a point's PSF contribution to sensor image."""
        # Placeholder - depends on pattern format
        pass
    
    def clear_cache(self):
        """Clear the lazy loading cache."""
        self._psf_cache = {}


class LFMForwardModel:
    """
    Complete LFM forward model using 5D PSF.
    
    Combines:
    - LFM geometry (lenslet layout, pixel mapping)
    - 5D PSF (depth and position dependent blur)
    - Volume representation (InstantNGP or grid)
    """
    
    def __init__(
        self,
        psf: LFM5DPSF,
        vol_bounds: Tuple[float, float, float] = (100, 100, 80),  # x, y, z in µm
        device: str = 'cuda'
    ):
        self.psf = psf
        self.vol_bounds = vol_bounds
        self.device = device
        
        # Compute coordinate mappings
        self._setup_coordinate_mapping()
        
    def _setup_coordinate_mapping(self):
        """Set up mappings between volume coordinates and sensor pixels."""
        # Voxel size in µm
        self.voxel_size = (
            2 * self.vol_bounds[0] / self.psf.voxel_resolution[2],  # x
            2 * self.vol_bounds[1] / self.psf.voxel_resolution[1],  # y
            2 * self.vol_bounds[2] / self.psf.voxel_resolution[0],  # z
        )
        
    def sample_volume_grid(self) -> torch.Tensor:
        """
        Generate 3D grid of coordinates matching the PSF voxel resolution.
        
        Returns:
            coords: [D, H, W, 3] tensor of (x, y, z) in normalized [0, 1] coords
        """
        D, H, W = self.psf.voxel_resolution
        
        # Create normalized coordinate grid [0, 1]
        z = torch.linspace(0, 1, D, device=self.device)
        y = torch.linspace(0, 1, H, device=self.device)
        x = torch.linspace(0, 1, W, device=self.device)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1)  # [D, H, W, 3]
        
        return coords
    
    def forward(
        self,
        model: torch.nn.Module,
        chunk_size: int = 65536
    ) -> torch.Tensor:
        """
        Forward project: query model at grid points, apply PSF to get sensor image.
        
        Args:
            model: Neural network that takes [N, 3] coords and returns [N, 1] densities
            chunk_size: Batch size for model queries
            
        Returns:
            sensor_image: [H, W] predicted sensor image
        """
        # Sample the full volume grid
        coords = self.sample_volume_grid()  # [D, H, W, 3]
        D, H, W, _ = coords.shape
        
        # Query model in chunks
        coords_flat = coords.reshape(-1, 3)  # [D*H*W, 3]
        
        densities = []
        for i in range(0, coords_flat.shape[0], chunk_size):
            chunk = coords_flat[i:i+chunk_size]
            with torch.no_grad():
                out = model(chunk)
                if isinstance(out, tuple):
                    out = out[0]
            densities.append(out)
        
        densities = torch.cat(densities, dim=0)
        volume = densities.reshape(D, H, W)  # [D, H, W]
        
        # Forward project using 5D PSF
        sensor_image = self.psf.forward_project_volume(volume)
        
        return sensor_image


def test_psf_loading(h5_path: str):
    """Test that PSF loads correctly."""
    psf = LFM5DPSF(h5_path, preload=False)
    
    print(f"\nTest loading pattern at depth 80 (index ~80), lenslet (1,1):")
    pattern = psf.get_forward_pattern(80, 1, 1)
    if pattern is not None:
        print(f"  Pattern shape: {pattern.shape}")
        print(f"  Pattern range: [{pattern.min():.4f}, {pattern.max():.4f}]")
    else:
        print("  Pattern not found")
    
    return psf
