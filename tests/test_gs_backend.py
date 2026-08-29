"""Gaussian-backend camera projection contracts."""

import pytest
import torch

from gs_backend import GSBackEnd, build_camera_projection


def test_projection_uses_each_cameras_focal_length() -> None:
    image_hw: tuple[int, int] = (100, 200)
    front = build_camera_projection(
        torch.tensor([100.0, 120.0, 100.0, 50.0]), image_hw, device="cpu"
    )
    side = build_camera_projection(
        torch.tensor([200.0, 240.0, 100.0, 50.0]), image_hw, device="cpu"
    )

    assert front.K[:4] == [100.0, 120.0, 100.0, 50.0]
    assert side.K[:4] == [200.0, 240.0, 100.0, 50.0]
    assert front.matrix[0, 0].item() == pytest.approx(1.0)
    assert side.matrix[0, 0].item() == pytest.approx(2.0)
    assert not torch.equal(front.matrix, side.matrix)


def test_eval_rendering_requires_a_cached_camera_projection() -> None:
    backend: GSBackEnd = GSBackEnd.__new__(GSBackEnd)
    backend.camera_projections = {}

    with pytest.raises(RuntimeError, match=r"no projection.*camera 3"):
        backend.eval_rendering({}, None, None, [], cam_idx=3)
