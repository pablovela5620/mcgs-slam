"""Projective geometry error contracts."""

import pytest
import torch
from lietorch import SE3

from geom.projective_ops import projective_transform


def test_jacobian_rejects_precomputed_and_rig_transforms_together() -> None:
    identity_vectors: torch.Tensor = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2],
        dtype=torch.float32,
    )
    poses = SE3(identity_vectors)
    depths: torch.Tensor = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    intrinsics: torch.Tensor = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    indices: torch.Tensor = torch.tensor([0], dtype=torch.long)
    rig_transform = SE3(identity_vectors[:, :1])
    relative_transform = SE3(identity_vectors[:, :1])

    with pytest.raises(ValueError, match="Gij and Tcb"):
        projective_transform(
            poses,
            depths,
            intrinsics,
            indices,
            indices + 1,
            jacobian=True,
            Tcb=rig_transform,
            Gij=relative_transform,
        )
