# cross_scale.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def decimate2(x: torch.Tensor):
    """
    Dyadic downsampling by decimation:
    (x↓2)[i,j] = x[2i, 2j]
    x: [B, C, H, W]
    """
    return x[..., ::2, ::2]

def avgpool2(x: torch.Tensor):
    """Optional anti-aliasing downsample (not exact identity)."""
    return F.avg_pool2d(x, kernel_size=2, stride=2)

def ifftshift2d(x: torch.Tensor):
    """Move center of kernel to (0,0) for FFT-based circular convolution."""
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(x, shifts=(-h // 2, -w // 2), dims=(-2, -1))

def pad_kernel_to(k: torch.Tensor, H: int, W: int):
    """
    Pad kernel to image size, keeping kernel centered.
    k: [B,1,hk,wk] or [1,1,hk,wk]
    returns padded k: [B,1,H,W]
    """
    hk, wk = k.shape[-2], k.shape[-1]
    pad_h = H - hk
    pad_w = W - wk
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Kernel larger than target size: {hk}x{wk} vs {H}x{W}")
    # pad evenly around
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    k_pad = F.pad(k, (pad_left, pad_right, pad_top, pad_bottom))
    return k_pad

def conv2d_fft(x: torch.Tensor, k: torch.Tensor):
    """
    Circular conv via FFT (matches paper's frequency-domain view).
    x: [B,1,H,W]
    k: [B,1,hk,wk] or [1,1,hk,wk]
    """
    B, C, H, W = x.shape
    if k.shape[0] == 1 and B > 1:
        k = k.expand(B, -1, -1, -1)

    k_pad = pad_kernel_to(k, H, W)
    k_pad = ifftshift2d(k_pad)

    X = torch.fft.fft2(x)
    K = torch.fft.fft2(k_pad)
    Y = X * K
    y = torch.fft.ifft2(Y).real
    return y

def qmf_filters(k: torch.Tensor):
    """
    QMF filters g1,g2,g3 from Eq.(10):
      g1[m,n] = (-1)^m k[m,n]
      g2[m,n] = (-1)^n k[m,n]
      g3[m,n] = (-1)^(m+n) k[m,n]
    k: [B,1,Hk,Wk]
    """
    device = k.device
    hk, wk = k.shape[-2], k.shape[-1]
    m = torch.arange(hk, device=device).view(1, 1, hk, 1)
    n = torch.arange(wk, device=device).view(1, 1, 1, wk)
    sign_m = (1.0 - 2.0 * (m % 2)).to(k.dtype)   # +1 for even, -1 for odd
    sign_n = (1.0 - 2.0 * (n % 2)).to(k.dtype)

    g1 = k * sign_m
    g2 = k * sign_n
    g3 = k * (sign_m * sign_n)
    return g1, g2, g3

class CrossScaleConsistencyLoss(nn.Module):
    """
    Cross-scale consistency loss from Eq.(12)-(13).
    For each scale s:
      left  = 4 * (x^(s+1) ⊗ k^(s+1))
      right = (x^s ⊗ k^s)↓2 + Σ_d (x^s ⊗ g_d^s)↓2
      Lcross^s = || F(left) - F(right) ||_1
    Summed over s in [0, S0-1].

    Notes:
    - Uses decimation downsample by default to match the exact identity.
    - If you want anti-aliasing, set downsample_mode="avgpool" (approximate).
    """
    def __init__(self, S0=2, downsample_mode="decimate", normalize_kernel=True, eps=1e-12):
        super().__init__()
        assert downsample_mode in ["decimate", "avgpool"]
        self.S0 = S0
        self.downsample_mode = downsample_mode
        self.normalize_kernel = normalize_kernel
        self.eps = eps

    def down2(self, x):
        return decimate2(x) if self.downsample_mode == "decimate" else avgpool2(x)

    def multiscale_pyramid(self, x, S0):
        xs = [x]
        for _ in range(S0):
            xs.append(self.down2(xs[-1]))
        return xs  # list length S0+1

    def multiscale_kernel_pyramid(self, k, S0):
        ks = [k]
        for _ in range(S0):
            ks.append(self.down2(ks[-1])) 
        return ks

    def forward(self, x_pred: torch.Tensor, k: torch.Tensor):
        """
        x_pred: [B,1,H,W] latent prediction at finest scale (from InstantNGP)
        k:      [B,1,Hk,Wk] or [1,1,Hk,Wk] kernel/PSF at finest scale
        """
        if x_pred.dim() != 4:
            raise ValueError("x_pred must be [B,1,H,W]")
        if k.dim() != 4:
            raise ValueError("k must be [B,1,Hk,Wk]")

        B = x_pred.shape[0]
        if k.shape[0] == 1 and B > 1:
            k = k.expand(B, -1, -1, -1)

        # Build dyadic pyramids
        xs = self.multiscale_pyramid(x_pred, self.S0)
        ks = self.multiscale_kernel_pyramid(k, self.S0)

        total = 0.0
        for s in range(self.S0):
            x_s, x_s1 = xs[s], xs[s + 1]
            k_s, k_s1 = ks[s], ks[s + 1]

            # QMF filters at scale s
            g1_s, g2_s, g3_s = qmf_filters(k_s)

            # left = 4 * (x^{s+1} ⊗ k^{s+1})
            left = 4.0 * conv2d_fft(x_s1, k_s1)

            # right = (x^s ⊗ k^s)↓2 + Σ (x^s ⊗ g_d^s)↓2
            main = self.down2(conv2d_fft(x_s, k_s))
            side1 = self.down2(conv2d_fft(x_s, g1_s))
            side2 = self.down2(conv2d_fft(x_s, g2_s))
            side3 = self.down2(conv2d_fft(x_s, g3_s))
            right = main + side1 + side2 + side3

            # frequency-domain L1
            F_left = torch.fft.fft2(left)
            F_right = torch.fft.fft2(right)
            loss_s = (F_left - F_right).abs().mean()
            total = total + loss_s

        return total

def compute_cross_scale_loss_from_model(
    model: nn.Module,
    psf_kernel: torch.Tensor,
    cross_loss_fn: nn.Module,
    patch_size: int,
    device: torch.device,
):
    """
    Render a patch latent image from InstantNGP, then compute cross-scale loss.
    Only for 2D.
    """
    # sample a random top-left in [0,1] coord space
    # build a regular grid patch in normalized coords (y,x order then reorder to x,y)
    ys = torch.linspace(0.0, 1.0, steps=patch_size, device=device)
    xs = torch.linspace(0.0, 1.0, steps=patch_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords_yx = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # [N,2] in (y,x)

    coords_for_model = torch.stack([coords_yx[:, 1], coords_yx[:, 0]], dim=-1)  # (x,y)
    pred_flat, _ = model(coords_for_model, variance=None)
    x_patch = pred_flat.view(1, 1, patch_size, patch_size)

    return cross_loss_fn(x_patch, psf_kernel)
