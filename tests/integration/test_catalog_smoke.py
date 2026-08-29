"""Live robocap catalog and TorchCodec decode smoke test."""

import pytest
import torch

from catalog_stream import RobocapSegment, camera_from_world_poses

pytestmark = pytest.mark.integration


def test_live_segment_decodes_one_rectified_frame_per_camera() -> None:
    segment = RobocapSegment(
        catalog_url="rerun+http://pablo-dl-server.ilish-ruler.ts.net:51235",
        dataset_id="18CFB19109CFDB071d88fb8b48ef65e9",
        segment_id="robocap__f408193e6447b3b0__s00000021",
        decoder="cuda",
    )

    with segment.open_window(
        start_seconds=0.0,
        end_seconds=2.0,
        kf_dist=1000.0,
        kf_angle=180.0,
    ) as window:
        frame = next(window.keyframes())

        assert len(window.relay_streams) == 4
        assert all(
            stream.samples and any(stream.is_keyframe)
            for stream in window.relay_streams
        )

    assert frame.images_rgb.shape == (4, 3, 360, 640)
    assert frame.images_rgb.dtype == torch.uint8
    assert segment.virtual_intrinsics.shape == (4, 4)
    assert segment.virtual_intrinsics.dtype == torch.float32
    camera_from_world = camera_from_world_poses(
        frame.world_from_rig,
        segment.rig_calibration,
    )
    assert camera_from_world.shape == (4, 7)
    assert camera_from_world.dtype == torch.float32
    assert all(stream.decoder is None for stream in window.relay_streams)
