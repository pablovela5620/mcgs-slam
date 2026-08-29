"""Hermetic contracts for the robocap catalog stream."""

import numpy as np

from catalog_stream import FisheyeRectifier, KeyframeSelector, camera_from_world_pose


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
    rectifier = FisheyeRectifier(intrinsic_33, distortion_4)

    rectified_xy: np.ndarray = rectifier.rectify_points(distorted_xy[None])[0]
    expected_xy: np.ndarray = np.array(
        [
            rectifier.virtual_K[0, 0] * point_xyz[0] / point_xyz[2] + 320.0,
            rectifier.virtual_K[1, 1] * point_xyz[1] / point_xyz[2] + 180.0,
        ]
    )

    assert np.linalg.norm(rectified_xy - expected_xy) <= 0.5


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
