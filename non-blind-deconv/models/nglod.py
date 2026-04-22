import sys
import os
import torch
import torch.nn as nn
from types import SimpleNamespace


class _ParamGroup(nn.Module):
    """Thin wrapper so encoder / decoder expose .parameters() the same way
    InstantNGPTorchModel does, while still being part of the same graph."""

    def __init__(self, module_list):
        super().__init__()
        self.modules_list = nn.ModuleList(module_list)

    def parameters(self, recurse=True):
        return self.modules_list.parameters(recurse=recurse)


class NglodModel(nn.Module):
    """
    Parameters
    ----------
    n_input_dims : int
        Must be 3 (OctreeSDF is 3-D only).
    num_lods : int
        Total number of octree levels (coarse → fine).
    base_lod : int
        Index of the coarsest LOD that is used in the loss.
    feature_dim : int
        Feature dimensionality stored at each octree node.
    feature_size : int
        Spatial resolution of the feature grid at the base level.
    hidden_dim : int
        Hidden width of the per-LOD MLP decoders.
    num_layers : int
        Number of hidden layers in each MLP decoder.
    sdfnet_root : str or None
        Absolute path to the sdf-net directory.  If None the constructor
        tries <this_file>/../../sdf-net.
    """

    def __init__(
        self,
        n_input_dims: int = 3,
        num_lods: int = 5,
        base_lod: int = 3,
        feature_dim: int = 16,
        feature_size: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        sdfnet_root: str = None,
    ):
        super().__init__()

        if n_input_dims != 3:
            raise ValueError("NglodModel only supports 3-D inputs (n_input_dims=3).")

        # ── locate sdf-net ──────────────────────────────────────────────────
        if sdfnet_root is None:
            sdfnet_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'sdf-net'
            )
        sdfnet_root = os.path.abspath(sdfnet_root)
        if sdfnet_root not in sys.path:
            sys.path.insert(0, sdfnet_root)

        from lib.models.OctreeSDF import OctreeSDF  

        # ── build args namespace expected by OctreeSDF ──────────────────────
        args = SimpleNamespace(
            net='OctreeSDF',
            feature_dim=feature_dim,
            feature_size=feature_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            input_dim=3,
            num_lods=num_lods,
            base_lod=base_lod,
            interpolate=None,
            growth_strategy='increase',
            grow_every=-1,
            pos_enc=False,
            ff_dim=-1,
            pos_invariant=False,
            joint_decoder=False,
            feat_sum=False,
            return_lst=True,
        )

        self.base_lod = base_lod
        self.num_lods = num_lods
        # current_stage is the highest active LOD index (0-based from base_lod)
        self._current_stage = base_lod  # start at coarsest

        self._net = OctreeSDF(args)

        # expose encoder / decoder so the training script can count params
        # the same way it does for InstantNGPTorchModel
        self.encoder = _ParamGroup(list(self._net.features))
        self.decoder = _ParamGroup(list(self._net.louts))

    def set_max_level(self, level: int):
        active = min(self.base_lod + level - 1, self.num_lods - 1)
        self._current_stage = active

    def _active_lod_indices(self):
        """Return list of LOD indices (into OctreeSDF's output list) that are
        currently active, from base_lod up to _current_stage (inclusive)."""
        return list(range(self.base_lod, self._current_stage + 1))

    def forward(
        self,
        coords: torch.Tensor,
        variance=None,           # ignored, kept for interface compatibility
        stochastic_alpha: float = 0.0,  # ignored, kept for interface compatibility
    ):
        # remap [0, 1] → [-1, 1] as OctreeSDF expects
        coords_nglod = coords * 2.0 - 1.0  # [N, 3]

        all_preds = self._net.sdf(coords_nglod, return_lst=True)  # list of [N, 1]

        # use the highest active LOD
        pred = all_preds[self._current_stage].squeeze(-1)  # [N]
        return pred, None

    def forward_multilod(self, coords: torch.Tensor):
        coords_nglod = coords * 2.0 - 1.0
        all_preds = self._net.sdf(coords_nglod, return_lst=True)
        active = self._active_lod_indices()
        return torch.stack([all_preds[i] for i in active], dim=0)  # [S, N, 1]