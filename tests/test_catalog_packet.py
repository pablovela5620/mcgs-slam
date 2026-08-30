"""Gaussian-backend packet contracts for trusted camera inputs."""

import subprocess
import sys

import lietorch
import pytest
import torch

from camera_packet import (
    CameraPacket,
    build_camera_packet,
    merge_camera_packets,
)


def test_catalog_packet_preserves_metric_units_and_masks_invalid_depth() -> None:
    images_rgb_n3hw: torch.Tensor = torch.arange(
        2 * 3 * 4 * 5,
        dtype=torch.uint8,
    ).reshape(2, 3, 4, 5)
    poses_camera_from_world_n7: torch.Tensor = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    depth_metres_nhw: torch.Tensor = torch.full((2, 4, 5), 2.5, dtype=torch.float32)
    normals_n3hw: torch.Tensor = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
    normals_n3hw[:, 2, :, :3] = 1.0
    intrinsics_n4: torch.Tensor = torch.tensor(
        [[224.06, 224.06, 320.0, 180.0], [224.06, 224.06, 320.0, 180.0]],
        dtype=torch.float32,
    )

    packet = build_camera_packet(
        frame_ids=torch.tensor([7, 8], dtype=torch.long),
        cam_idx=2,
        n_cameras=4,
        poses_camera_from_world_n7=poses_camera_from_world_n7,
        images_rgb_n3hw=images_rgb_n3hw,
        depth_metres_nhw=depth_metres_nhw,
        normals_n3hw=normals_n3hw,
        intrinsics_n4=intrinsics_n4,
        map_scale=1.0,
    )

    assert packet["frame_ids"].shape == (2,) and packet["frame_ids"].dtype == torch.int64
    assert packet["frame_ids"].tolist() == [7, 8]
    assert packet["view_ids"].tolist() == [30, 34]
    assert packet["poses"].shape == (2, 7) and packet["poses"].dtype == torch.float32
    assert torch.equal(packet["poses"][:, :3], poses_camera_from_world_n7[:, :3])
    assert packet["images"].shape == (2, 3, 4, 5) and packet["images"].dtype == torch.uint8
    assert packet["normals"].shape == (2, 3, 4, 5) and packet["normals"].dtype == torch.float32
    assert packet["depths"].shape == (2, 4, 5) and packet["depths"].dtype == torch.float32
    assert torch.all(packet["depths"][:, :, :3] == 2.5)
    assert torch.all(packet["depths"][:, :, 3:] == 0.0)
    assert packet["intrinsics"].shape == (2, 4) and packet["intrinsics"].dtype == torch.float32
    assert packet["cam_idx"] == 2


def _packet(frame_ids: list[int], cam_idx: int, n_cameras: int = 3) -> CameraPacket:
    """Build a compact valid packet for identity and merge tests."""
    n_frames: int = len(frame_ids)
    return build_camera_packet(
        frame_ids=torch.tensor(frame_ids, dtype=torch.long),
        cam_idx=cam_idx,
        n_cameras=n_cameras,
        poses_camera_from_world_n7=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * n_frames,
            dtype=torch.float32,
        ),
        images_rgb_n3hw=torch.zeros((n_frames, 3, 2, 2), dtype=torch.uint8),
        depth_metres_nhw=torch.ones((n_frames, 2, 2), dtype=torch.float32),
        normals_n3hw=torch.tensor(
            [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]
            * n_frames,
            dtype=torch.float32,
        ),
        intrinsics_n4=torch.tensor(
            [[10.0, 10.0, 1.0, 1.0]] * n_frames,
            dtype=torch.float32,
        ),
        map_scale=1.0,
    )


def test_view_ids_are_unique_across_cameras_and_buffer_compaction() -> None:
    before_compaction: CameraPacket = _packet([0, 1, 499], cam_idx=1)
    after_compaction: CameraPacket = _packet([500, 501], cam_idx=0)
    other_camera: CameraPacket = _packet([500, 501], cam_idx=2)

    view_ids: list[int] = torch.cat(
        [
            before_compaction["view_ids"],
            after_compaction["view_ids"],
            other_camera["view_ids"],
        ]
    ).tolist()

    assert len(view_ids) == len(set(view_ids))
    assert after_compaction["view_ids"].tolist() == [1500, 1503]


def test_merge_camera_packets_uses_tensor_camera_indices_and_explicit_updates() -> None:
    packets: list[CameraPacket] = [_packet([7, 8], cam_idx=0), _packet([7, 8], cam_idx=2)]
    pose_updates = lietorch.SE3.InitFromVec(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    )
    scale_updates: torch.Tensor = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    merged = merge_camera_packets(
        packets,
        pose_updates=pose_updates,
        scale_updates=scale_updates,
    )

    assert merged["frame_ids"].tolist() == [7, 8, 7, 8]
    assert merged["view_ids"].tolist() == [21, 24, 23, 26]
    assert merged["cam_indices"].tolist() == [0, 0, 2, 2]
    assert merged["pose_update_indices"].tolist() == [0, 1, 0, 1]
    assert isinstance(merged["pose_updates"], lietorch.SE3)
    assert torch.equal(merged["pose_updates"].data, pose_updates.data)
    assert torch.equal(merged["scale_updates"], scale_updates)


def test_merge_camera_packets_normalizes_tensor_pose_updates() -> None:
    updates_n7 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    merged = merge_camera_packets(
        [_packet([7, 8], cam_idx=0), _packet([7, 8], cam_idx=1)],
        pose_updates=updates_n7,
        scale_updates=torch.ones((2, 1), dtype=torch.float32),
    )

    assert isinstance(merged["pose_updates"], lietorch.SE3)
    assert torch.equal(merged["pose_updates"].data, updates_n7)


def test_merge_camera_packets_rejects_mismatched_frame_ids() -> None:
    with pytest.raises(ValueError, match="same frame ids"):
        merge_camera_packets(
            [_packet([7, 8], cam_idx=0), _packet([7, 9], cam_idx=1)],
        )


def test_importing_catalog_entry_point_does_not_load_droid_backends() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import infer_catalog; assert 'droid_backends' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
