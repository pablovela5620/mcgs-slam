"""Hermetic contracts for the robocap catalog stream."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rig import CameraSensor, RigCalibration

import catalog_stream
from catalog_stream import (
    CatalogKeyframe,
    CatalogWindow,
    FisheyeRectifier,
    KeyframeSelector,
    RobocapSegment,
    camera_from_world_pose,
    catalog_camera_path,
)


def _fisheye_camera(
    intrinsic_33: np.ndarray,
    distortion_4: np.ndarray,
    camera_id: int = 0,
) -> CameraSensor:
    """Build one SimpleCV camera for rectification examples."""
    extrinsics = Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))
    intrinsics = Intrinsics.from_k_matrix(
        camera_conventions="RDF",
        k_matrix=intrinsic_33,
        height=1080,
        width=1920,
    )
    distortion = KannalaBrandtDistortion(
        k1=float(distortion_4[0]),
        k2=float(distortion_4[1]),
        k3=float(distortion_4[2]),
        k4=float(distortion_4[3]),
    )
    return CameraSensor(
        index=camera_id,
        name=f"camera-{camera_id}",
        kind="rgb",
        pinhole=Fisheye62Parameters(
            name=f"camera-{camera_id}",
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=distortion,
        ),
    )


def _project_kb4(
    point_xyz: np.ndarray,
    intrinsic_33: np.ndarray,
    distortion_4: np.ndarray,
) -> np.ndarray:
    """Project one camera-frame point with the independent KB4 definition."""
    normalized_xy: np.ndarray = point_xyz[:2] / point_xyz[2]
    radius: float = float(np.linalg.norm(normalized_xy))
    theta: float = float(np.arctan(radius))
    theta2: float = theta * theta
    polynomial: float = 1.0 + sum(
        float(coefficient) * theta2 ** (order + 1)
        for order, coefficient in enumerate(distortion_4)
    )
    distorted_xy: np.ndarray = normalized_xy * (theta * polynomial / radius)
    return np.array(
        [
            intrinsic_33[0, 0] * distorted_xy[0] + intrinsic_33[0, 2],
            intrinsic_33[1, 1] * distorted_xy[1] + intrinsic_33[1, 2],
        ],
        dtype=np.float64,
    )


def test_kb4_rectification_lands_on_virtual_pinhole_projection() -> None:
    intrinsic_33: np.ndarray = np.array(
        [[610.0, 0.0, 958.0], [0.0, 608.0, 541.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion_4: np.ndarray = np.array([0.07, -0.03, 0.02, -0.006], dtype=np.float64)
    point_xyz: np.ndarray = np.array([0.55, -0.22, 1.8], dtype=np.float64)
    distorted_xy: np.ndarray = _project_kb4(point_xyz, intrinsic_33, distortion_4)
    rectifier = FisheyeRectifier(_fisheye_camera(intrinsic_33, distortion_4))

    rectified_xy: np.ndarray = rectifier.rectify_points(distorted_xy[None])[0]
    expected_xy: np.ndarray = np.array(
        [
            rectifier.virtual_camera.intrinsics.fl_x * point_xyz[0] / point_xyz[2]
            + 320.0,
            rectifier.virtual_camera.intrinsics.fl_y * point_xyz[1] / point_xyz[2]
            + 180.0,
        ]
    )

    assert np.linalg.norm(rectified_xy - expected_xy) <= 0.5
    assert isinstance(rectifier.virtual_camera, PinholeParameters)


def _z_rotation(degrees: float) -> np.ndarray:
    """Return a literal Z-axis rotation for the selector examples."""
    radians: float = np.deg2rad(degrees)
    cosine: float = float(np.cos(radians))
    sine: float = float(np.sin(radians))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_keyframe_rule_uses_motion_since_last_keyframe() -> None:
    poses_n44: np.ndarray = np.repeat(np.eye(4, dtype=np.float64)[None], 6, axis=0)
    poses_n44[1, 0, 3] = 0.20
    poses_n44[2, 0, 3] = 0.31
    poses_n44[3, 0, 3] = 0.31
    poses_n44[3, :3, :3] = _z_rotation(14.0)
    poses_n44[4, 0, 3] = 0.31
    poses_n44[4, :3, :3] = _z_rotation(15.1)
    poses_n44[5, 0, 3] = 0.59
    poses_n44[5, :3, :3] = _z_rotation(15.1)

    selected: np.ndarray = KeyframeSelector(
        distance_metres=0.3,
        angle_degrees=15.0,
    ).select(poses_n44)

    assert selected.tolist() == [0, 2, 4]


def test_camera_from_world_pose_composes_catalog_transforms() -> None:
    world_from_rig_44: np.ndarray = np.eye(4, dtype=np.float64)
    world_from_rig_44[:3, :3] = _z_rotation(90.0)
    world_from_rig_44[:3, 3] = [1.0, 2.0, 0.0]
    camera_from_rig_44: np.ndarray = np.eye(4, dtype=np.float64)
    camera_from_rig_44[0, 3] = 0.1

    camera_from_world_7: np.ndarray = camera_from_world_pose(
        world_from_rig_44,
        camera_from_rig_44,
    )

    assert np.allclose(camera_from_world_7[:3], [-1.9, 1.0, 0.0], atol=1e-6)
    assert np.allclose(
        camera_from_world_7[3:],
        [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-6,
    )


def test_mapper_camera_zero_recovers_basalt_rig_placement() -> None:
    world_from_rig_44: np.ndarray = np.eye(4, dtype=np.float64)
    world_from_rig_44[:3, :3] = _z_rotation(25.0)
    world_from_rig_44[:3, 3] = [2.0, -3.0, 0.4]

    rig_from_world_7: np.ndarray = camera_from_world_pose(
        world_from_rig_44,
        np.eye(4, dtype=np.float64),
    )
    rig_from_world_44: np.ndarray = np.eye(4, dtype=np.float64)
    quaternion_xyzw: np.ndarray = rig_from_world_7[3:]
    x, y, z, w = quaternion_xyzw
    rig_from_world_44[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    rig_from_world_44[:3, 3] = rig_from_world_7[:3]

    assert np.allclose(np.linalg.inv(rig_from_world_44), world_from_rig_44, atol=1e-6)


def test_catalog_keyframe_contains_only_dynamic_rig_frame_data() -> None:
    assert [field.name for field in fields(CatalogKeyframe)] == [
        "timestamp_ns",
        "images_rgb",
        "world_from_rig",
    ]


def test_catalog_segment_requires_endpoint_dataset_and_segment() -> None:
    with pytest.raises(TypeError):
        RobocapSegment()  # type: ignore[call-arg]


def _catalog_segment_with_stubbed_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_id: str | None = None,
) -> tuple[RobocapSegment, list[tuple[str | None, str | None]]]:
    """Build a segment at the public catalog boundary without live I/O."""
    dataset_requests: list[tuple[str | None, str | None]] = []

    class StubCatalogClient:
        def __init__(self, url: str) -> None:
            assert url == "rerun+http://catalog"

        def get_dataset(
            self,
            name: str | None = None,
            *,
            id: str | None = None,
        ) -> object:
            dataset_requests.append((name, id))
            return object()

    camera: CameraSensor = _fisheye_camera(
        np.array(
            [[610.0, 0.0, 958.0], [0.0, 608.0, 541.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.array([0.07, -0.03, 0.02, -0.006], dtype=np.float64),
    )
    rig_calibration = RigCalibration(cameras=[camera], reference_index=0)
    monkeypatch.setattr(catalog_stream.catalog, "CatalogClient", StubCatalogClient)
    monkeypatch.setattr(
        RobocapSegment,
        "_read_calibrations",
        lambda self: rig_calibration,
    )
    monkeypatch.setattr(RobocapSegment, "_first_packet_ns", lambda self, paths: 0)
    segment = RobocapSegment(
        catalog_url="rerun+http://catalog",
        dataset_id=dataset_id,
        segment_id="segment",
        decoder="cpu",
    )
    return segment, dataset_requests


def test_catalog_segment_resolves_default_dataset_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, dataset_requests = _catalog_segment_with_stubbed_io(monkeypatch)

    assert dataset_requests == [("robocap", None)]
    assert segment.dataset_name == "robocap"


def test_catalog_segment_allows_dataset_id_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_requests = _catalog_segment_with_stubbed_io(
        monkeypatch,
        dataset_id="dataset-id",
    )

    assert dataset_requests == [(None, "dataset-id")]


def test_catalog_window_reuses_segment_rectifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, _ = _catalog_segment_with_stubbed_io(monkeypatch)
    segment.cam00_frame0_ns = -1
    stream = SimpleNamespace(
        times_ns=np.array([0], dtype=np.int64),
        decoder=object(),
    )
    monkeypatch.setattr(segment, "_read_video_window", lambda camera, start, end: stream)
    monkeypatch.setattr(
        segment,
        "_read_poses",
        lambda start, end: (
            np.array([0], dtype=np.int64),
            np.eye(4, dtype=np.float64)[None],
        ),
    )

    window = segment.open_window(start_seconds=0.0, end_seconds=1.0)

    assert window.rectifiers is segment.rectifiers


def test_catalog_keyframes_decode_each_camera_in_bounded_chunks() -> None:
    requested_batches: list[list[list[int]]] = [[] for _ in range(4)]

    class StubDecoder:
        def __init__(self, camera_index: int) -> None:
            self.camera_index: int = camera_index

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            requested_batches[self.camera_index].append(indices)
            values: torch.Tensor = torch.tensor(indices, dtype=torch.uint8)[:, None, None, None]
            return SimpleNamespace(data=values.expand(-1, 3, 2, 2))

    window: CatalogWindow = CatalogWindow.__new__(CatalogWindow)
    window._closed = False
    window.relay_streams = tuple(
        SimpleNamespace(decoder=StubDecoder(camera_index))
        for camera_index in range(4)
    )
    window.rectifiers = tuple(
        SimpleNamespace(rectify=lambda image_rgb_hwc: image_rgb_hwc)
        for _ in range(4)
    )
    window._stream_indices = tuple(
        np.arange(17, dtype=np.int64) for _ in range(4)
    )
    window._times_ns = np.arange(17, dtype=np.int64)
    window._world_from_rig_n44 = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        17,
        axis=0,
    )

    frames: list[CatalogKeyframe] = list(window.keyframes())

    assert len(frames) == 17
    assert [len(batch) for batch in requested_batches[0]] == [8, 8, 1]
    assert all(
        [len(batch) for batch in camera_batches] == [8, 8, 1]
        for camera_batches in requested_batches
    )


def test_catalog_camera_paths_use_simplecv_zero_padding() -> None:
    assert catalog_camera_path(1) == "/world/rig_00/cam_01"
    assert catalog_camera_path(10) == "/world/rig_00/cam_10"


def test_scalar_str_unwraps_single_instance_components() -> None:
    from catalog_stream import _scalar_str

    assert _scalar_str(["left_front"]) == "left_front"
    assert _scalar_str([["rgb"]]) == "rgb"
    assert _scalar_str("plain") == "plain"
