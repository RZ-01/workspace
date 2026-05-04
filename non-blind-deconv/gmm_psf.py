"""
PSF 密度估计：KDE（默认）或 Normalizing Flow（zuko）

两种后端，接口完全相同：
  - 'kde'  : Parzen 窗，每个非零像素一个 Gaussian，无需拟合，精确
  - 'flow' : zuko NSF（神经样条流），可拟合任意连续分布，需要 pip install zuko

Usage
-----
    python gmm_psf.py --psf_path psf.png --backend kde   --output_path out.pkl
    python gmm_psf.py --psf_path psf.png --backend flow  --output_path out.pkl

Library:
    from gmm_psf import GMMPsf
    psf = GMMPsf.from_discrete_psf(psf_np, backend='kde')   # or 'flow'
    offsets = psf.sample_gpu(10000, device)
"""

import argparse, os, pickle
import cv2, numpy as np, torch, tifffile, matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# KDE backend (Parzen window)
# ──────────────────────────────────────────────────────────────────────────────

class _KDEBackend:
    def __init__(self, means, weights, bandwidth):
        self.means_np    = means.astype(np.float32)
        self.weights_np  = weights.astype(np.float32)
        self.bandwidth   = float(bandwidth)
        self._gpu_device = None

    def fit(self, *_, **__): pass   # no-op, KDE needs no fitting

    def sample_cpu(self, n):
        rng = np.random.default_rng()
        idx = rng.choice(len(self.weights_np), size=n, replace=True,
                         p=self.weights_np)
        D = self.means_np.shape[1]
        return (self.means_np[idx]
                + rng.normal(0, self.bandwidth, (n, D)).astype(np.float32))

    def prepare_gpu(self, device):
        self._gpu_device  = device
        self._means_gpu   = torch.tensor(self.means_np,   device=device)
        self._weights_gpu = torch.tensor(self.weights_np, device=device)

    def sample_gpu(self, n, device):
        if self._gpu_device != device:
            self.prepare_gpu(device)
        k   = torch.multinomial(self._weights_gpu, n, replacement=True)
        eps = torch.randn(n, self.means_np.shape[1], device=device) * self.bandwidth
        return (self._means_gpu[k] + eps).float()

    def log_prob_np(self, coords):
        """Evaluate KDE log-density at coords (N, D) → (N,) numpy."""
        X = torch.tensor(coords, dtype=torch.float32)         # (N, D)
        M = torch.tensor(self.means_np, dtype=torch.float32)  # (K, D)
        W = torch.tensor(self.weights_np, dtype=torch.float32)
        sq = ((X[:, None] - M[None]) ** 2).sum(-1)            # (N, K)
        D  = coords.shape[1]
        lc = (-0.5 * D * np.log(2 * np.pi * self.bandwidth**2)
              - sq / (2 * self.bandwidth**2))                  # (N, K)
        lp = torch.logsumexp(lc + torch.log(W + 1e-300), dim=1)
        return lp.numpy()

    def state_dict(self):
        return {"means": self.means_np, "weights": self.weights_np,
                "bandwidth": self.bandwidth}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["means"], d["weights"], d["bandwidth"])


# ──────────────────────────────────────────────────────────────────────────────
# Normalizing Flow backend (zuko NSF)
# ──────────────────────────────────────────────────────────────────────────────

class _FlowBackend:
    """
    Neural Spline Flow via zuko.  Fits a bijective neural network that maps
    a standard Gaussian to the PSF distribution.  Can represent any
    continuous distribution given enough capacity.

    Requires:  pip install zuko
    """

    def __init__(self, n_dims, n_transforms=6, hidden=128, bins=16):
        import zuko
        self._n_dims       = n_dims
        self._n_transforms = n_transforms
        self._hidden       = hidden
        self._bins         = bins
        self.flow = zuko.flows.NSF(
            features        = n_dims,
            transforms      = n_transforms,
            hidden_features = [hidden, hidden],
            bins            = bins,
        )
        self._gpu_device = None

    def fit(self, coords_np, weights_np, n_epochs=2000, lr=3e-3,
            device="cuda", verbose=True):
        """
        Fit flow to weighted pixel coordinates.
        coords_np  : (K, D) nonzero pixel centres
        weights_np : (K,)   normalised PSF values
        """
        self.flow = self.flow.to(device)
        coords_t  = torch.tensor(coords_np,  dtype=torch.float32, device=device)
        weights_t = torch.tensor(weights_np, dtype=torch.float32, device=device)

        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

        best_loss, patience, patience_limit = float("inf"), 0, 200
        for epoch in range(n_epochs):
            opt.zero_grad()
            # Weighted NLL: -E_w[log p(x)]
            log_p = self.flow().log_prob(coords_t)          # (K,)
            loss  = -(log_p * weights_t).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
            opt.step(); sched.step()

            l = loss.item()
            if verbose and epoch % 200 == 0:
                print(f"  [flow] epoch {epoch:4d}  loss={l:.4f}")
            if l < best_loss - 1e-4:
                best_loss = l; patience = 0
            else:
                patience += 1
            if patience >= patience_limit:
                if verbose:
                    print(f"  [flow] early stop at epoch {epoch}")
                break

    def sample_cpu(self, n):
        with torch.no_grad():
            s = self.flow().sample((n,))
        return s.cpu().numpy().astype(np.float32)

    def prepare_gpu(self, device):
        self._gpu_device = device
        self.flow = self.flow.to(device)

    def sample_gpu(self, n, device):
        if self._gpu_device != device:
            self.prepare_gpu(device)
        with torch.no_grad():
            return self.flow().sample((n,)).float()

    def log_prob_np(self, coords):
        device = next(self.flow.parameters()).device
        x = torch.tensor(coords, dtype=torch.float32, device=device)
        with torch.no_grad():
            lp = self.flow().log_prob(x)
        return lp.cpu().numpy()

    def state_dict(self):
        return {"flow_state": self.flow.state_dict(),
                "flow_config": {
                    "n_dims":       self._n_dims,
                    "n_transforms": self._n_transforms,
                    "hidden":       self._hidden,
                    "bins":         self._bins,
                }}

    @classmethod
    def from_state_dict(cls, d):
        cfg = d["flow_config"]
        obj = cls(cfg["n_dims"], cfg["n_transforms"],
                  cfg.get("hidden", 128), cfg.get("bins", 16))
        obj.flow.load_state_dict(d["flow_state"])
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────────────────────────────────────

class GMMPsf:

    def __init__(self, backend, psf_shape):
        self._backend     = backend
        self.psf_shape    = psf_shape
        self.n_dims       = len(psf_shape)
        # expose for viz titles
        self.n_components = getattr(backend, "means_np",
                                    np.zeros((1,1))).shape[0]
        self.bandwidth    = getattr(backend, "bandwidth", None)

    @classmethod
    def from_discrete_psf(
        cls,
        psf: np.ndarray,
        backend: str = "kde",
        bandwidth: float = 0.4,
        # flow params
        n_transforms: int = 6,
        flow_hidden:  int = 128,
        flow_bins:    int = 16,
        flow_epochs:  int = 2000,
        flow_lr:      float = 3e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
    ) -> "GMMPsf":
        psf = psf.astype(np.float32)
        psf_shape = psf.shape
        n_dims = len(psf_shape)

        axes   = [np.arange(s) - (s-1)/2.0 for s in psf_shape]
        grids  = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)

        w = psf.ravel().clip(0.0)
        w /= w.sum()
        mask = w > 0
        nz_coords, nz_weights = coords[mask], w[mask]
        nz_weights /= nz_weights.sum()

        if verbose:
            print(f"PSF shape={psf_shape}, nonzero pixels={mask.sum()}, backend={backend}")

        if backend == "kde":
            b = _KDEBackend(nz_coords, nz_weights, bandwidth)

        elif backend == "flow":
            b = _FlowBackend(n_dims, n_transforms, flow_hidden, flow_bins)
            if verbose:
                print(f"Fitting normalizing flow ({n_transforms} transforms, "
                      f"hidden={flow_hidden}, bins={flow_bins}) ...")
            b.fit(nz_coords, nz_weights,
                  n_epochs=flow_epochs, lr=flow_lr,
                  device=device, verbose=verbose)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'kde' or 'flow'.")

        return cls(b, psf_shape)

    # ── sampling ──────────────────────────────────────────────────────

    def sample(self, n, device=None):
        t = torch.from_numpy(self._backend.sample_cpu(n))
        return t if device is None else t.to(device)

    def prepare_gpu(self, device):
        self._backend.prepare_gpu(device)

    def sample_gpu(self, n, device):
        return self._backend.sample_gpu(n, device)

    # ── density ───────────────────────────────────────────────────────

    def log_prob(self, coords: np.ndarray) -> np.ndarray:
        return self._backend.log_prob_np(coords)

    def reconstruct(self, shape=None):
        if shape is None: shape = self.psf_shape
        axes  = [np.arange(s)-(s-1)/2.0 for s in shape]
        grids = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)
        lp = self.log_prob(coords)
        p  = np.exp(lp - lp.max()).reshape(shape)
        return (p / p.sum()).astype(np.float32)

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        btype = "kde" if isinstance(self._backend, _KDEBackend) else "flow"
        with open(path, "wb") as f:
            pickle.dump({"backend_type": btype,
                         "backend_state": self._backend.state_dict(),
                         "psf_shape": self.psf_shape}, f)
        print(f"PSF saved ({btype}) → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        btype = d.get("backend_type", "kde")
        B = _KDEBackend if btype == "kde" else _FlowBackend
        b = B.from_state_dict(d["backend_state"])
        return cls(b, d["psf_shape"])

    @classmethod
    def load_or_fit(cls, checkpoint_path, psf=None, **kw):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            return cls.load(checkpoint_path)
        if psf is None:
            raise ValueError("No checkpoint and psf=None")
        obj = cls.from_discrete_psf(psf, **kw)
        if checkpoint_path:
            obj.save(checkpoint_path)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation — FIXED: compare at discrete pixel positions, not histogram
# ──────────────────────────────────────────────────────────────────────────────

def visualize_gmm_psf_comparison(psf, gmm_psf, num_slices=5, save_path=None):
    n_dims   = len(psf.shape)
    psf_norm = psf / psf.sum()
    recon    = gmm_psf.reconstruct(psf.shape)

    if n_dims == 3:
        D, H, W = psf.shape
        sidx = np.linspace(0, D-1, num_slices, dtype=int)
        fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4*num_slices))
        for i, z in enumerate(sidx):
            o, r, e = psf_norm[z], recon[z], np.abs(psf_norm[z]-recon[z])
            for col, dat, title, cmap in zip(
                    range(3), [o,r,e],
                    [f"Original (Z={z})", f"Recon (Z={z})", f"Error (Z={z})"],
                    ["hot","hot","viridis"]):
                im = axes[i,col].imshow(dat, cmap=cmap, interpolation="nearest")
                axes[i,col].set_title(title); axes[i,col].axis("off")
                plt.colorbar(im, ax=axes[i,col], fraction=0.046)
            cy = H//2
            axes[i,3].plot(o[cy], "b-", label="Original", lw=2)
            axes[i,3].plot(r[cy], "r--",label="Recon",    lw=2)
            mse = np.mean((o-r)**2)
            axes[i,3].set_title(f"Line Z={z}"); axes[i,3].legend(); axes[i,3].grid(alpha=0.3)
            axes[i,3].text(0.02,0.98,f"MSE:{mse:.2e}",transform=axes[i,3].transAxes,va="top",
                           bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))
    else:
        H, W = psf.shape
        fig, axes = plt.subplots(1,4,figsize=(16,4))
        o, r, e = psf_norm, recon, np.abs(psf_norm-recon)
        for col, dat, title, cmap in zip(range(3),[o,r,e],
                ["Original","Recon","Error"],["hot","hot","viridis"]):
            im = axes[col].imshow(dat,cmap=cmap,interpolation="nearest")
            axes[col].set_title(title); axes[col].axis("off")
            plt.colorbar(im,ax=axes[col],fraction=0.046)
        cy = H//2
        axes[3].plot(o[cy],"b-",label="Original",lw=2)
        axes[3].plot(r[cy],"r--",label="Recon",lw=2)
        mse = np.mean((o-r)**2)
        axes[3].legend(); axes[3].grid(alpha=0.3)
        axes[3].text(0.02,0.98,f"MSE:{mse:.2e}",transform=axes[3].transAxes,va="top",
                     bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))

    plt.suptitle(f"PSF Reconstruction", fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_gmm_sampling_distribution(psf, gmm_psf, n_samples=200_000, save_path=None):
    n_dims      = len(psf.shape)
    dim_labels  = ["Z","Y","X"] if n_dims == 3 else ["Y","X"]
    axis_coords = [np.arange(s)-(s-1)/2.0 for s in psf.shape]

    psf_norm = psf / psf.sum()

    # PSF marginals at discrete pixel positions
    psf_marginals = []
    for ax in range(n_dims):
        other = tuple(i for i in range(n_dims) if i != ax)
        psf_marginals.append(psf_norm.sum(axis=other))

    # ── KEY FIX: evaluate MODEL density at the same discrete positions ──
    # Build 1-D grids for each dimension, evaluate the full n-D density,
    # then marginalise analytically via reconstruct() and sum.
    recon = gmm_psf.reconstruct(psf.shape)
    model_marginals = []
    for ax in range(n_dims):
        other = tuple(i for i in range(n_dims) if i != ax)
        model_marginals.append(recon.sum(axis=other))

    # Also draw raw samples for the histogram (visual sanity check)
    samples = gmm_psf.sample(n_samples).numpy()

    fig, axes_list = plt.subplots(n_dims, 1, figsize=(10, 4*n_dims))
    if n_dims == 1: axes_list = [axes_list]

    for dim_idx, (ax, label, coords, psf_m, mod_m) in enumerate(
            zip(axes_list, dim_labels, axis_coords, psf_marginals, model_marginals)):

        lo, hi = coords[0]-0.5, coords[-1]+0.5
        bins   = len(coords) * 4   # finer bins for the raw-samples histogram

        # Raw samples histogram (background reference)
        ax.hist(samples[:, dim_idx], bins=bins, range=(lo, hi),
                density=True, alpha=0.35, color="tomato", label="Raw samples")

        # Model marginal (evaluated analytically) — should match PSF exactly for KDE
        ax.plot(coords, mod_m / mod_m.sum() / (coords[1]-coords[0] if len(coords)>1 else 1),
                "r-", lw=2, label="Model marginal (analytic)")

        # Ground-truth PSF marginal
        ax.plot(coords, psf_m / psf_m.sum() / (coords[1]-coords[0] if len(coords)>1 else 1),
                "b--", lw=2, label="PSF marginal (ground truth)")

        ax.set_xlabel(f"{label} offset (pixels)", fontsize=12)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Marginal along {label}", fontsize=12)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

    bname = "KDE" if isinstance(gmm_psf._backend, _KDEBackend) else "Flow"
    fig.suptitle(f"{bname} PSF — Marginal Comparison\n{n_samples:,} samples", fontsize=13)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_psf(path):
    if path.endswith((".tif",".tiff")): return tifffile.imread(path).astype(np.float32)
    if path.endswith(".npy"):           return np.load(path).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim==3 and img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    elif img.ndim==3:                   img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psf_path",    default="/workspace/temp/W_DIP/results/levin/WDIP/im1_kernel2_img_k.png")
    p.add_argument("--output_path", default="../checkpoints/levin_kernel2.pkl")
    p.add_argument("--backend",     default="kde", choices=["kde","flow"])
    p.add_argument("--bandwidth",   type=float, default=0.4)
    p.add_argument("--flow_epochs", type=int,   default=2000)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_viz",      action="store_true")
    p.add_argument("--num_viz_slices", type=int, default=5)
    p.add_argument("--viz_path",          default=None)
    p.add_argument("--sampling_viz_path", default=None)
    p.add_argument("--n_sampling_viz",    type=int, default=200_000)
    args = p.parse_args()

    psf_np  = _load_psf(args.psf_path)
    gmm_psf = GMMPsf.from_discrete_psf(
        psf_np, backend=args.backend, bandwidth=args.bandwidth,
        flow_epochs=args.flow_epochs, device=args.device, verbose=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    gmm_psf.save(args.output_path)

    device = torch.device(args.device)
    print("CPU sample:", gmm_psf.sample(3))
    print("GPU sample:", gmm_psf.sample_gpu(3, device))

    if not args.no_viz:
        stem = args.output_path.replace(".pkl","")
        visualize_gmm_psf_comparison(psf_np, gmm_psf, args.num_viz_slices,
                                     args.viz_path or f"{stem}_comparison.pdf")
        visualize_gmm_sampling_distribution(psf_np, gmm_psf, args.n_sampling_viz,
                                            args.sampling_viz_path or f"{stem}_sampling.pdf")


if __name__ == "__main__":
    main()