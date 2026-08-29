"""Typed camera observations passed to the Gaussian mapper."""

from typing import TypeAlias, TypedDict

import lietorch
import torch
from jaxtyping import Float32, Int64, UInt8


PoseUpdates: TypeAlias = Float32[torch.Tensor, "n 7"] | lietorch.SE3 | None


class CameraPacket(TypedDict):
    """One camera's observations in Gaussian-map units."""

    frame_ids: Int64[torch.Tensor, "n"]
    view_ids: Int64[torch.Tensor, "n"]
    poses: Float32[torch.Tensor, "n 7"]
    images: UInt8[torch.Tensor, "n 3 h w"]
    normals: Float32[torch.Tensor, "n 3 h w"]
    depths: Float32[torch.Tensor, "n h w"]
    intrinsics: Float32[torch.Tensor, "n 4"]
    cam_idx: int


class GlobalCameraPacket(TypedDict):
    """Merged observations from one or more rig cameras."""

    frame_ids: Int64[torch.Tensor, "n_views"]
    view_ids: Int64[torch.Tensor, "n_views"]
    poses: Float32[torch.Tensor, "n_views 7"]
    images: UInt8[torch.Tensor, "n_views 3 h w"]
    normals: Float32[torch.Tensor, "n_views 3 h w"]
    depths: Float32[torch.Tensor, "n_views h w"]
    intrinsics: Float32[torch.Tensor, "n_views 4"]
    cam_indices: Int64[torch.Tensor, "n_views"]
    pose_updates: PoseUpdates
    scale_updates: Float32[torch.Tensor, "n_frames 1"] | None


RigPacketBatch: TypeAlias = GlobalCameraPacket


def build_camera_packet(
    *,
    frame_ids: Int64[torch.Tensor, "n"],
    cam_idx: int,
    n_cameras: int,
    poses_camera_from_world_n7: Float32[torch.Tensor, "n 7"],
    images_rgb_n3hw: UInt8[torch.Tensor, "n 3 h w"],
    depth_metres_nhw: Float32[torch.Tensor, "n h w"],
    normals_n3hw: Float32[torch.Tensor, "n 3 h w"],
    intrinsics_n4: Float32[torch.Tensor, "n 4"],
    map_scale: float,
) -> CameraPacket:
    """Build one mapper packet from trusted camera observations.

    Args:
        frame_ids: Stable input-frame identities with shape ``[n]``. These
            identities must survive tracker-buffer compaction.
        cam_idx: Zero-based camera index within the rig.
        n_cameras: Total camera count used to form collision-free view IDs.
        poses_camera_from_world_n7: Float32 ``T_camera_from_world`` vectors
            with shape ``[n, 7]`` and metric translations.
        images_rgb_n3hw: UInt8 RGB images with shape ``[n, 3, h, w]``.
        depth_metres_nhw: Float32 metric depth with shape ``[n, h, w]``.
        normals_n3hw: Float32 camera-frame normals with shape ``[n, 3, h, w]``;
            a zero normal marks invalid prior geometry.
        intrinsics_n4: Float32 ``[fx, fy, cx, cy]`` values with shape ``[n, 4]``.
        map_scale: Translation and depth scale from metres to map units.

    Returns:
        A CPU packet consumed by the Gaussian backend.

    Raises:
        ValueError: If the camera index is outside the declared rig.
    """
    if n_cameras < 1:
        raise ValueError("n_cameras must be positive")
    if not 0 <= cam_idx < n_cameras:
        raise ValueError(f"cam_idx {cam_idx} is outside a {n_cameras}-camera rig")

    frame_ids_n: torch.Tensor = frame_ids.detach().to(device="cpu", dtype=torch.long)
    view_ids_n: torch.Tensor = frame_ids_n * n_cameras + cam_idx
    poses_scaled_n7: torch.Tensor = poses_camera_from_world_n7.detach().to(
        device="cpu", dtype=torch.float32
    ).clone()
    poses_scaled_n7[:, :3] *= map_scale
    normals_cpu_n3hw: torch.Tensor = normals_n3hw.detach().to(
        device="cpu", dtype=torch.float32
    )
    depth_scaled_nhw: torch.Tensor = depth_metres_nhw.detach().to(
        device="cpu", dtype=torch.float32
    ) * map_scale
    valid_nhw: torch.Tensor = normals_cpu_n3hw.norm(dim=1) > 0.5

    return {
        "frame_ids": frame_ids_n,
        "view_ids": view_ids_n,
        "poses": poses_scaled_n7,
        "images": images_rgb_n3hw.detach().to(device="cpu", dtype=torch.uint8),
        "normals": normals_cpu_n3hw,
        "depths": torch.where(valid_nhw, depth_scaled_nhw, torch.zeros_like(depth_scaled_nhw)),
        "intrinsics": intrinsics_n4.detach().to(device="cpu", dtype=torch.float32),
        "cam_idx": cam_idx,
    }


def merge_camera_packets(
    packets: list[CameraPacket],
    *,
    pose_updates: PoseUpdates = None,
    scale_updates: Float32[torch.Tensor, "n_frames 1"] | None = None,
) -> GlobalCameraPacket:
    """Merge per-camera packets and attach optional rig-pose corrections.

    Args:
        packets: Per-camera packets in camera-major order.
        pose_updates: Optional float32 pose corrections with shape ``[n_frames, 7]``.
        scale_updates: Optional float32 scale corrections with shape ``[n_frames, 1]``.

    Returns:
        One typed batch with a tensor-valued ``cam_indices`` field.

    Raises:
        ValueError: If no camera packets are supplied.
    """
    if not packets:
        raise ValueError("at least one camera packet is required")

    tensor_keys: tuple[str, ...] = (
        "frame_ids",
        "view_ids",
        "poses",
        "images",
        "normals",
        "depths",
        "intrinsics",
    )
    merged: dict[str, object] = {
        key: torch.cat([packet[key] for packet in packets], dim=0)  # type: ignore[literal-required]
        for key in tensor_keys
    }
    merged["cam_indices"] = torch.cat(
        [
            torch.full_like(packet["frame_ids"], packet["cam_idx"], dtype=torch.long)
            for packet in packets
        ]
    )
    merged["pose_updates"] = (
        pose_updates.to(device="cpu") if pose_updates is not None else None
    )
    merged["scale_updates"] = (
        scale_updates.to(device="cpu", dtype=torch.float32)
        if scale_updates is not None
        else None
    )
    return merged  # type: ignore[return-value]
