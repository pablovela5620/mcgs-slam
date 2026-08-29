"""Live robocap catalog and TorchCodec decode smoke test."""

import pytest
import torch

from catalog_stream import RobocapSegment

pytestmark = pytest.mark.integration


def test_live_segment_decodes_one_rectified_frame_per_camera() -> None:
    segment = RobocapSegment(decoder="cuda")

    frame = next(
        segment.iter_keyframes(
            start_seconds=0.0,
            end_seconds=2.0,
            kf_dist=1000.0,
            kf_angle=180.0,
        )
    )

    assert frame.images_rgb.shape == (4, 3, 360, 640)
    assert frame.images_rgb.dtype == torch.uint8
    assert frame.virtual_K.shape == (4, 4) and frame.virtual_K.dtype == torch.float32
    assert frame.camera_from_world.shape == (4, 7)
    assert frame.camera_from_world.dtype == torch.float32
    assert len(frame.relay_streams) == 4
    assert all(stream.samples and any(stream.is_keyframe) for stream in frame.relay_streams)
