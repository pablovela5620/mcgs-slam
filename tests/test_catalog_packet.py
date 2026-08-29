"""Gaussian-backend packet contract for trusted catalog inputs."""

import torch

from mcgs import build_camera_packet


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
        viz_idx=torch.tensor([7, 8], dtype=torch.long),
        cam_idx=2,
        poses_camera_from_world_n7=poses_camera_from_world_n7,
        images_rgb_n3hw=images_rgb_n3hw,
        depth_metres_nhw=depth_metres_nhw,
        normals_n3hw=normals_n3hw,
        intrinsics_n4=intrinsics_n4,
        scale_factor=1.0,
    )

    assert packet["viz_idx"].shape == (2,) and packet["viz_idx"].dtype == torch.int64
    assert packet["tstamp"].tolist() == [1007, 1008]
    assert packet["poses"].shape == (2, 7) and packet["poses"].dtype == torch.float32
    assert torch.equal(packet["poses"][:, :3], poses_camera_from_world_n7[:, :3])
    assert packet["images"].shape == (2, 3, 4, 5) and packet["images"].dtype == torch.uint8
    assert packet["normals"].shape == (2, 3, 4, 5) and packet["normals"].dtype == torch.float32
    assert packet["depths"].shape == (2, 4, 5) and packet["depths"].dtype == torch.float32
    assert torch.all(packet["depths"][:, :, :3] == 2.5)
    assert torch.all(packet["depths"][:, :, 3:] == 0.0)
    assert packet["intrinsics"].shape == (2, 4) and packet["intrinsics"].dtype == torch.float32
    assert packet["cam_idx"] == 2
    assert packet["pose_updates"] is None and packet["scale_updates"] is None
