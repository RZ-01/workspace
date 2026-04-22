import torch

def compute_levin_gaussian_prior(
    model,
    coords_for_model: torch.Tensor,
    variance=None,
    per_dim_weights: torch.Tensor = None,
    reduction: str = "mean",
):
    """
    Gaussian image prior:
        sum_{i,k} | g_k * x |_2^2
    where g_k are derivative filters (gx=[1,-1], gy=[1,-1]^T, etc.).
    For an INR f(coords), this becomes an L2 penalty on spatial gradients:
        E[ ||∇f||_2^2 ].
    """
    coords = coords_for_model.clone().detach().requires_grad_(True)

    preds, _ = model(coords, variance=variance)
    preds = preds.squeeze(-1)  # [N]

    grads = torch.autograd.grad(
        outputs=preds,
        inputs=coords,
        grad_outputs=torch.ones_like(preds),
        create_graph=True,   # IMPORTANT: needed so prior affects params
        retain_graph=True,
        only_inputs=True
    )[0]  # [N, D]

    if per_dim_weights is not None:
        grads = grads * per_dim_weights.view(1, -1)

    grad_sq = (grads ** 2).sum(dim=-1)  # [N], ||∇f||^2

    if reduction == "sum":
        return grad_sq.sum()
    return grad_sq.mean()