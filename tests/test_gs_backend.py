"""Gaussian-backend camera projection contracts."""

from types import SimpleNamespace

import lietorch
import pytest
import torch

from camera_packet import merge_camera_packets
from gs_backend import GSBackEnd, build_camera_projection
from test_catalog_packet import _packet


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


def test_global_corrections_follow_gaussian_provenance_and_skip_missing_views() -> None:
    backend: GSBackEnd = GSBackEnd.__new__(GSBackEnd)
    backend.gaussians = SimpleNamespace(
        unique_kfIDs=torch.tensor([0, 1, 0, 99], dtype=torch.int32),
        get_xyz=torch.tensor(
            [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        get_scaling=torch.tensor(
            [[2.0, 2.0, 2.0], [8.0, 8.0, 8.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]],
            dtype=torch.float32,
        ),
        get_rotation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            dtype=torch.float32,
        ),
    )
    backend.gaussians._xyz = backend.gaussians.get_xyz.clone()
    backend.gaussians._scaling = backend.gaussians.get_scaling.clone()
    backend.gaussians._rotation = backend.gaussians.get_rotation.clone()
    backend.gaussians.scaling_inverse_activation = lambda value: value
    packet = merge_camera_packets(
        [_packet([0, 1], cam_idx=0, n_cameras=1)],
        pose_updates=lietorch.SE3.InitFromVec(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        ),
        scale_updates=torch.tensor([[2.0], [4.0]], dtype=torch.float32),
    )

    backend._apply_global_corrections(packet)

    assert torch.allclose(
        backend.gaussians._xyz,
        torch.tensor(
            [[1.0, 0.0, 0.0], [27.5, 0.0, 0.0], [1.5, 0.0, 0.0], [7.0, 0.0, 0.0]]
        ),
    )
    assert torch.allclose(
        backend.gaussians._scaling,
        torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        ),
    )
    assert torch.equal(backend.gaussians._rotation[3], backend.gaussians.get_rotation[3])


def test_global_corrections_reject_duplicate_view_rows() -> None:
    backend: GSBackEnd = GSBackEnd.__new__(GSBackEnd)
    backend.gaussians = SimpleNamespace(unique_kfIDs=torch.tensor([0], dtype=torch.int32))
    packet = merge_camera_packets([_packet([0], cam_idx=0, n_cameras=1)])
    packet["view_ids"] = torch.tensor([0, 0], dtype=torch.long)
    packet["pose_update_indices"] = torch.tensor([0, 0], dtype=torch.long)
    packet["pose_updates"] = lietorch.SE3.Identity(1)
    packet["scale_updates"] = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(AssertionError, match="at most one correction row"):
        backend._apply_global_corrections(packet)
