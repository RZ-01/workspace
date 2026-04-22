"""
Gaussian Mixture Model (GMM) based PSF representation.

Fits a discrete PSF using sklearn's GaussianMixture and provides
continuous (sub-pixel) coordinate sampling from the fitted distribution.

Usage
-----
Standalone fitting:
    python gmm_psf.py --psf_path /path/to/psf.tif --output_path psf.pkl

As a library:
    from gmm_psf import GMMPsf
    gmm = GMMPsf.from_discrete_psf(psf_np, n_components=64)
    offsets = gmm.sample(10000)   # (10000, n_dims) float32 pixel offsets
"""

import argparse
import os
import pickle

import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class GMMPsf:
    """
    Gaussian Mixture Model representation of a PSF.

    The GMM is fitted to the PSF treated as a probability distribution
    over pixel coordinates (PSF values are used as sample weights).

    Sampling returns continuous (sub-pixel) offsets in pixel units,
    centred at the PSF origin (centre of the array).

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        A fitted GaussianMixture object.
    psf_shape : tuple
        Original PSF shape, e.g. (D, H, W) or (H, W).
    """

    def __init__(self, gmm: GaussianMixture, psf_shape: tuple):
        self.gmm = gmm
        self.psf_shape = psf_shape
        self.n_dims = len(psf_shape)
        self.n_components = gmm.n_components

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_discrete_psf(
        cls,
        psf: np.ndarray,
        n_components: int = 64,
        covariance_type: str = "full",
        max_iter: int = 500,
        n_init: int = 3,
        random_state: int = 42,
        verbose: bool = True,
    ) -> "GMMPsf":
        """
        Fit a GMM to a discrete PSF array.

        The PSF values are normalised to a probability distribution and used
        as sample weights for fitting.

        Parameters
        ----------
        psf : np.ndarray
            Discrete PSF array, shape (H, W) or (D, H, W).  Values must be
            non-negative.
        n_components : int
            Number of Gaussian components.
        covariance_type : str
            Covariance type for GaussianMixture ('full', 'diag', 'tied', 'spherical').
        max_iter : int
            Maximum EM iterations.
        n_init : int
            Number of initialisations; the best is kept.
        random_state : int
            Random seed.
        verbose : bool
            Print progress information.
        """
        psf = psf.astype(np.float64)
        psf_shape = psf.shape
        n_dims = len(psf_shape)

        # --- Build pixel-coordinate array (centred at 0) ---
        axes = [np.arange(s) - (s - 1) / 2.0 for s in psf_shape]
        grids = np.meshgrid(*axes, indexing="ij")           # each: psf_shape
        coords = np.stack([g.ravel() for g in grids], axis=1)  # (N, n_dims)

        # --- Normalise PSF to a probability distribution ---
        weights = psf.ravel().astype(np.float64)
        weights = np.clip(weights, 0.0, None)
        weight_sum = weights.sum()
        if weight_sum == 0:
            raise ValueError("PSF has all-zero values; cannot fit GMM.")
        weights /= weight_sum

        # sklearn GaussianMixture does not accept sample_weights directly in
        # fit(), but we can use the weighted-sample trick: subsample the
        # coordinate array with replacement using PSF probabilities, then fit
        # the GMM to those samples (which implicitly encodes the distribution).
        #
        # We use n_samples proportional to the number of non-zero pixels so
        # that the GMM has enough data to represent fine structure.
        n_nonzero = int((weights > 0).sum())
        n_samples = max(100_000, n_nonzero * 10)
        n_samples = min(n_samples, 2_000_000)   # cap to avoid OOM

        if verbose:
            print(f"PSF shape: {psf_shape}, n_dims: {n_dims}")
            print(f"Non-zero pixels: {n_nonzero:,} / {len(weights):,}")
            print(f"Drawing {n_samples:,} weighted samples for GMM fitting ...")

        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(coords), size=n_samples, replace=True, p=weights)
        X = coords[sample_idx]   # (n_samples, n_dims)

        if verbose:
            print(
                f"Fitting GMM: n_components={n_components}, "
                f"covariance_type={covariance_type}, n_init={n_init} ..."
            )

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            verbose=2 if verbose else 0,
            verbose_interval=50,
        )
        gmm.fit(X)

        if verbose:
            ll = gmm.score(X)
            print(f"GMM fitting done.  Log-likelihood per sample: {ll:.4f}")
            print(f"  Converged: {gmm.converged_}")
            print(f"  n_iter: {gmm.n_iter_}")

        return cls(gmm=gmm, psf_shape=psf_shape)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, device: torch.device = None) -> torch.Tensor:
        """
        Draw *n* continuous (sub-pixel) offsets from the fitted GMM.

        Returns
        -------
        offsets : torch.Tensor, shape (n, n_dims), dtype float32
            Pixel-unit offsets centred at the PSF origin.  These are
            drop-in replacements for the integer offsets produced by the
            discrete multinomial sampler.
        """
        samples_np, _ = self.gmm.sample(n)             # (n, n_dims) float64
        offsets = torch.from_numpy(samples_np.astype(np.float32))
        if device is not None:
            offsets = offsets.to(device)
        return offsets

    def log_prob(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluate GMM log-probability density at given coordinates.

        Parameters
        ----------
        coords : np.ndarray, shape (N, n_dims)
            Pixel-unit coordinates centred at the PSF origin.

        Returns
        -------
        log_prob : np.ndarray, shape (N,)
        """
        return self.gmm.score_samples(coords)

    # ------------------------------------------------------------------
    # Reconstruction helper (for visualisation / evaluation)
    # ------------------------------------------------------------------

    def reconstruct(self, shape: tuple = None) -> np.ndarray:
        """
        Reconstruct a PSF image from the GMM by evaluating the density
        on a regular grid matching the original PSF shape (or a custom shape).

        Parameters
        ----------
        shape : tuple, optional
            Grid shape to reconstruct on.  Defaults to `self.psf_shape`.

        Returns
        -------
        psf_recon : np.ndarray
            Reconstructed PSF, normalised to sum to 1.
        """
        if shape is None:
            shape = self.psf_shape

        axes = [np.arange(s) - (s - 1) / 2.0 for s in shape]
        grids = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([g.ravel() for g in grids], axis=1)   # (N, n_dims)

        log_p = self.log_prob(coords)
        p = np.exp(log_p - log_p.max())     # relative probability
        p = p.reshape(shape)
        p /= p.sum()
        return p.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the GMM PSF to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"gmm": self.gmm, "psf_shape": self.psf_shape}, f)
        print(f"GMM PSF saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GMMPsf":
        """Load a GMM PSF from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(gmm=data["gmm"], psf_shape=data["psf_shape"])
        print(f"GMM PSF loaded from {path}  (n_components={obj.n_components}, shape={obj.psf_shape})")
        return obj


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_gmm_psf_comparison(
    psf: np.ndarray,
    gmm_psf: GMMPsf,
    num_slices: int = 5,
    save_path: str = None,
) -> None:
    """
    Visualise original PSF vs GMM reconstruction at multiple Z slices (3D)
    or as a single panel (2D).

    Parameters
    ----------
    psf : np.ndarray
        Original discrete PSF, shape (D, H, W) or (H, W).
    gmm_psf : GMMPsf
        Fitted GMM PSF.
    num_slices : int
        Number of Z slices to show (3D only).
    save_path : str, optional
        If given, saves the figure to this path.
    """
    n_dims = len(psf.shape)

    # Normalise original for comparison
    psf_norm = psf / psf.sum() if psf.sum() > 0 else psf

    # Reconstruct GMM on original grid
    recon = gmm_psf.reconstruct(psf.shape)

    if n_dims == 3:
        D, H, W = psf.shape
        slice_indices = np.linspace(0, D - 1, num_slices, dtype=int)
        fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4 * num_slices))

        for i, z_idx in enumerate(slice_indices):
            orig_slice = psf_norm[z_idx]
            recon_slice = recon[z_idx]
            error = np.abs(orig_slice - recon_slice)

            im0 = axes[i, 0].imshow(orig_slice, cmap="hot", interpolation="nearest")
            axes[i, 0].set_title(f"Original PSF (Z={z_idx})")
            axes[i, 0].axis("off")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

            im1 = axes[i, 1].imshow(recon_slice, cmap="hot", interpolation="nearest")
            axes[i, 1].set_title(f"GMM Recon (Z={z_idx})")
            axes[i, 1].axis("off")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            im2 = axes[i, 2].imshow(error, cmap="viridis", interpolation="nearest")
            axes[i, 2].set_title(f"Abs Error (Z={z_idx})")
            axes[i, 2].axis("off")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

            center_y = H // 2
            axes[i, 3].plot(orig_slice[center_y, :], "b-", label="Original", linewidth=2)
            axes[i, 3].plot(recon_slice[center_y, :], "r--", label="GMM Recon", linewidth=2)
            mse = np.mean((orig_slice - recon_slice) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            axes[i, 3].set_title(f"Center Line (Z={z_idx})")
            axes[i, 3].set_xlabel("X")
            axes[i, 3].set_ylabel("Intensity")
            axes[i, 3].legend()
            axes[i, 3].grid(True, alpha=0.3)
            axes[i, 3].text(
                0.02, 0.98,
                f"MSE: {mse:.2e}\nPSNR: {psnr:.1f} dB",
                transform=axes[i, 3].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    elif n_dims == 2:
        H, W = psf.shape
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        error = np.abs(psf_norm - recon)

        im0 = axes[0].imshow(psf_norm, cmap="hot", interpolation="nearest")
        axes[0].set_title("Original PSF")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(recon, cmap="hot", interpolation="nearest")
        axes[1].set_title("GMM Recon")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(error, cmap="viridis", interpolation="nearest")
        axes[2].set_title("Abs Error")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        center_y = H // 2
        axes[3].plot(psf_norm[center_y, :], "b-", label="Original", linewidth=2)
        axes[3].plot(recon[center_y, :], "r--", label="GMM Recon", linewidth=2)
        mse = np.mean((psf_norm - recon) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        axes[3].set_title("Center Line Profile")
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Intensity")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].text(
            0.02, 0.98,
            f"MSE: {mse:.2e}\nPSNR: {psnr:.1f} dB",
            transform=axes[3].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    else:
        raise ValueError(f"Unsupported PSF dimensionality: {n_dims}")

    plt.suptitle(f"GMM PSF Comparison  (K={gmm_psf.n_components} components)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Slice comparison saved to {save_path}")

    plt.close()


def visualize_gmm_sampling_distribution(
    psf: np.ndarray,
    gmm_psf: "GMMPsf",
    n_samples: int = 200_000,
    save_path: str = None,
) -> None:
    """
    Visualise the GMM sampling distribution by comparing per-axis marginal
    histograms of GMM samples against the PSF's marginal projections.

    For a 3D PSF this produces a 3-row figure (Z, Y, X axes).
    For a 2D PSF this produces a 2-row figure (Y, X axes).

    Parameters
    ----------
    psf : np.ndarray
        Original discrete PSF, shape (D, H, W) or (H, W).
    gmm_psf : GMMPsf
        Fitted GMM PSF.
    n_samples : int
        Number of GMM samples to draw for the histograms.
    save_path : str, optional
        Path to save the figure (.pdf or .png).
    """
    n_dims = len(psf.shape)
    dim_labels = (["Z", "Y", "X"] if n_dims == 3 else ["Y", "X"])
    axes_names = dim_labels

    # Pixel-centre positions for each axis (centred at 0)
    axis_coords = [np.arange(s) - (s - 1) / 2.0 for s in psf.shape]

    # PSF marginal projections (sum over all other axes)
    psf_norm = psf / psf.sum()
    psf_marginals = []
    for ax in range(n_dims):
        other = tuple(i for i in range(n_dims) if i != ax)
        psf_marginals.append(psf_norm.sum(axis=other))

    # GMM samples
    samples = gmm_psf.sample(n_samples).numpy()  # (N, n_dims)

    fig, axes_list = plt.subplots(n_dims, 1, figsize=(10, 4 * n_dims))
    if n_dims == 1:
        axes_list = [axes_list]

    for dim_idx, (ax, label, coords, psf_marg) in enumerate(
        zip(axes_list, axes_names, axis_coords, psf_marginals)
    ):
        # PSF marginal as a step plot
        ax.plot(
            coords, psf_marg / psf_marg.max(),
            color="steelblue", linewidth=2,
            label="PSF marginal (normalised)",
        )
        # GMM sample histogram for this axis
        lo, hi = coords[0], coords[-1]
        ax.hist(
            samples[:, dim_idx],
            bins=len(coords),
            range=(lo - 0.5, hi + 0.5),
            density=True,
            alpha=0.55,
            color="tomato",
            label="GMM samples (density)",
        )
        # Normalise hist so peak == 1 for easy visual comparison
        # (already density=True, just rescale axis label)
        ax.set_xlabel(f"{label} offset (pixels)", fontsize=12)
        ax.set_ylabel("Relative density", fontsize=11)
        ax.set_title(f"Marginal distribution along {label}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GMM Sampling Distribution vs PSF Marginals\n"
        f"K={gmm_psf.n_components} components, {n_samples:,} samples",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Sampling distribution saved to {save_path}")

    plt.close()


# ---------------------------------------------------------------------------
# CLI entry-point (standalone fitting)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit a Gaussian Mixture Model to a discrete PSF for continuous sampling."
    )
    parser.add_argument("--psf_path", type=str, required=True,
                        help="Path to discrete PSF file (.tif, .tiff, or .npy)")
    parser.add_argument("--output_path", type=str, default="../checkpoints/psf_gmm.pkl",
                        help="Path to save the fitted GMM PSF (.pkl)")
    parser.add_argument("--n_components", type=int, default=64,
                        help="Number of Gaussian components")
    parser.add_argument("--covariance_type", type=str, default="full",
                        choices=["full", "diag", "tied", "spherical"],
                        help="Covariance type for GaussianMixture")
    parser.add_argument("--max_iter", type=int, default=500,
                        help="Maximum EM iterations")
    parser.add_argument("--n_init", type=int, default=3,
                        help="Number of initialisations (best kept)")
    parser.add_argument("--num_viz_slices", type=int, default=5,
                        help="Number of Z slices for slice-comparison figure (3D only)")
    parser.add_argument("--viz_path", type=str, default=None,
                        help="Path to save the slice-comparison figure. "
                             "Extension determines format: .pdf (default) or .png. "
                             "Defaults to <output_path stem>_comparison.pdf")
    parser.add_argument("--sampling_viz_path", type=str, default=None,
                        help="Path to save the marginal-histogram sampling figure. "
                             "Defaults to <output_path stem>_sampling.pdf")
    parser.add_argument("--n_sampling_viz", type=int, default=200_000,
                        help="Number of GMM samples to draw for the sampling-distribution figure")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip all visualisation")
    args = parser.parse_args()

    # --- Load PSF ---
    print(f"Loading PSF from {args.psf_path}")
    if args.psf_path.endswith((".tif", ".tiff")):
        psf_np = tifffile.imread(args.psf_path).astype(np.float32)
    elif args.psf_path.endswith(".npy"):
        psf_np = np.load(args.psf_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {args.psf_path}")

    print(f"PSF shape: {psf_np.shape}, range: [{psf_np.min():.4f}, {psf_np.max():.4f}]")

    # --- Fit GMM ---
    gmm_psf = GMMPsf.from_discrete_psf(
        psf=psf_np,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        n_init=args.n_init,
        verbose=True,
    )

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    gmm_psf.save(args.output_path)

    # --- Demo sampling ---
    print("\n--- Demo: Continuous Sampling ---")
    sample_offsets = gmm_psf.sample(10)
    print(f"Sample offsets (shape {sample_offsets.shape}):\n{sample_offsets}")

    # --- Visualise ---
    if not args.no_viz:
        stem = args.output_path.replace(".pkl", "")

        # Figure 1: slice-by-slice comparison (original vs GMM reconstruction)
        viz_path = args.viz_path or f"{stem}_comparison.pdf"
        visualize_gmm_psf_comparison(
            psf=psf_np,
            gmm_psf=gmm_psf,
            num_slices=args.num_viz_slices,
            save_path=viz_path,
        )

        # Figure 2: per-axis marginal histograms vs PSF projections
        sampling_viz_path = args.sampling_viz_path or f"{stem}_sampling.pdf"
        visualize_gmm_sampling_distribution(
            psf=psf_np,
            gmm_psf=gmm_psf,
            n_samples=args.n_sampling_viz,
            save_path=sampling_viz_path,
        )


if __name__ == "__main__":
    main()
