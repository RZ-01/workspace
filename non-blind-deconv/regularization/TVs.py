import torch
import torch.nn as nn

def compute_tvl1_loss(
    model: nn.Module,
    coords_for_model: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    coords_for_model: [N, D], already in correct order for model (x,y) or (x,y,z)
    eps: small finite-difference step in coordinate space
    """
    N, D = coords_for_model.shape
    device = coords_for_model.device
    
    # Repeat twice: one for original coords, one for shifted coords
    coords_rep = coords_for_model.unsqueeze(1).repeat(1, D, 1)   # [N, D, D]
    
    # Create D shifted versions: add eps on each dimension
    shift = torch.eye(D, device=device) * eps                    # [D, D]
    coords_shifted = coords_rep + shift.unsqueeze(0)            # [N, D, D]
    
    # Flatten for one forward pass
    coords_full = torch.cat([
        coords_for_model,                 # original N points
        coords_shifted.view(-1, D)        # shifted N*D points
    ], dim=0)
    
    # Forward once
    pred_full, _ = model(coords_full, variance=None)             # [N + N*D, 1]
    pred_full = pred_full.squeeze(-1)
    
    # Recover original + shifted preds
    pred_orig = pred_full[:N]                                    # [N]
    pred_shifted = pred_full[N:].view(N, D)                      # [N, D]
    
    # Finite-diff TV
    tv = (pred_shifted - pred_orig.unsqueeze(1)).abs().mean()
    return tv

def compute_finite_difference_tv_loss(model, coords_grid, eps=1e-3):
    """
    coords_grid: [N, D] rule grid (e.g. 64x64 flattened)
    eps: finite difference step
    """
    N, D = coords_grid.shape
    device = coords_grid.device

    # 原始值 f(x)
    values, _ = model(coords_grid, variance=None)
    values = values.squeeze(-1)

    # 轴向偏移向量
    shift = torch.eye(D, device=device) * eps   # [D, D]

    grad_list = []
    for d in range(D):
        coords_shifted = coords_grid + shift[d]         # x+eps_d
        values_shift, _ = model(coords_shifted, variance=None)
        grad = (values_shift.squeeze(-1) - values)      # difference
        grad_list.append(grad)

    grads = torch.stack(grad_list, dim=-1)     # [N, D]
    tv = grads.abs().mean()

    return tv

def compute_hyperlaplacian_tv(model, coords_grid, alpha=0.67, variance=None):
    coords = coords_grid.clone().detach().requires_grad_(True)
    values, _ = model(coords, variance=variance)
    
    grads = torch.autograd.grad(
        outputs=values,
        inputs=coords,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad_mag = grads.norm(dim=-1)
    tv_values = torch.pow(grad_mag, alpha)
    
    return tv_values