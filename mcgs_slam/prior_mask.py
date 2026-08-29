"""Validity of the monocular prior, derived from its stored normals.

MoGe-2 zeroes depth and normal for pixels without geometry (sky, undefined).
The tracker keeps only the normals at full resolution, so a zero normal is the
validity mask: no prior geometry there, so no Gaussian is seeded and no depth
loss is applied.
"""

import torch
from jaxtyping import Bool, Float

_MIN_NORMAL_LENGTH: float = 0.5


def prior_valid(normals: Float[torch.Tensor, "n 3 h w"]) -> Bool[torch.Tensor, "n h w"]:
    """True where the stored prior normal is a unit vector (MoGe marked the pixel valid)."""
    return normals.norm(dim=1) > _MIN_NORMAL_LENGTH


def mask_invalid_depth(
    depths: Float[torch.Tensor, "n h w"], normals: Float[torch.Tensor, "n 3 h w"]
) -> Float[torch.Tensor, "n h w"]:
    """Return depths with prior-invalid pixels set to 0 (skipped by seeding and the depth loss)."""
    return torch.where(prior_valid(normals).to(depths.device), depths, torch.zeros_like(depths))
