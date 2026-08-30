"""Rerun logging for the legacy Waymo-style image-directory path."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from lietorch import SE3
from torch import Tensor

from prior_mask import prior_valid
from rerun_logger import DashboardSpec, RerunLogger

if TYPE_CHECKING:
    from depth_video import DepthVideo


COMPARE_ROOT: str = "render_vs_gt"


def _w2c_vecs_to_world_t_cams(
    poses_w2c_n7: Float[Tensor, "n 7"],
    map_scale: float,
) -> tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 4"]]:
    """Invert camera-from-world vectors and scale their translations."""
    poses_scaled_n7: Float[Tensor, "n 7"] = (
        poses_w2c_n7.detach().float().cpu().clone()
    )
    poses_scaled_n7[:, :3] *= map_scale
    world_t_cam_n7: Float[np.ndarray, "n 7"] = SE3(poses_scaled_n7).inv().data.numpy()
    return world_t_cam_n7[:, :3], world_t_cam_n7[:, 3:7]


class WaymoRerunLogger(RerunLogger):
    """Log the RDF Waymo rig, tracker depths, and Gaussian map."""

    def __init__(
        self,
        imagedirs: list[str],
        *,
        map_scale: float,
        save_path: str | None = None,
        spawn: bool = False,
        splat_every: int = 10,
        splat_cap: int = 75_000,
        max_splat_scale: float = 8.0,
        splat_scale_percentile: float = 99.7,
        max_depth: float = 60.0,
        refine_every: int = 10_000,
        image_plane_distance: float = 0.4,
        trajectory_radius: float = 0.01,
    ) -> None:
        self._calibrated: bool = False
        camera_names: list[str] = [Path(imagedir).name for imagedir in imagedirs]
        super().__init__(
            camera_names=camera_names,
            camera_ids=list(range(len(camera_names))),
            map_scale=map_scale,
            world_coordinates=rr.ViewCoordinates.RDF,
            save_path=save_path,
            spawn=spawn,
            splat_every=splat_every,
            splat_cap=splat_cap,
            max_splat_scale=max_splat_scale,
            splat_scale_percentile=splat_scale_percentile,
            max_depth=max_depth,
            refine_every=refine_every,
            image_plane_distance=image_plane_distance,
            trajectory_radius=trajectory_radius,
        )

    def _cam_path(self, cam_idx: int) -> str:
        return (
            f"world/rig/cam_{self.camera_ids[cam_idx]:02d}_"
            f"{self.cam_names[cam_idx]}"
        )

    def _pinhole_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/pinhole"

    def _image_path(self, cam_idx: int) -> str:
        return f"{self._pinhole_path(cam_idx)}/image"

    def _depth_path(self, cam_idx: int) -> str:
        return f"{self._pinhole_path(cam_idx)}/depth"

    def _video_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/video"

    def _compare_path(self, cam_idx: int) -> str:
        return f"{COMPARE_ROOT}/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _render_path(self, cam_idx: int) -> str:
        return f"{self._compare_path(cam_idx)}/rendered"

    def _ground_truth_path(self, cam_idx: int) -> str:
        return f"{self._compare_path(cam_idx)}/gt"

    def _dashboard_spec(self) -> DashboardSpec:
        """Return Waymo entity paths and RDF follow-view controls."""
        camera_indices: range = range(len(self.cam_names))
        return DashboardSpec(
            image_origins=tuple(self._pinhole_path(index) for index in camera_indices),
            image_contents=("+ $origin/image",),
            depth_origins=tuple(self._depth_path(index) for index in camera_indices),
            render_origins=tuple(self._render_path(index) for index in camera_indices),
            ground_truth_origins=tuple(
                self._ground_truth_path(index) for index in camera_indices
            ),
            excluded_3d_paths=(
                f"{COMPARE_ROOT}/**",
                *(self._depth_path(index) for index in camera_indices),
                *(self._video_path(index) for index in camera_indices),
            ),
            follow_origin="world/rig",
            follow_exclusions=("world/keyframes/**",),
            follow_eye=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(0.0, -1.2, -2.5),
                look_target=(0.0, 0.0, 1.5),
                eye_up=(0.0, -1.0, 0.0),
                spin_speed=0.0,
            ),
        )

    def log_frame(
        self,
        frame_idx: int,
        timestamp: float,
        images_bgr_n3hw: UInt8[Tensor, "n_dirs 3 h w"],
        intrinsics_n8: Float[Tensor, "n_dirs 8"],
        video: "DepthVideo",
    ) -> None:
        """Advance the timelines and log per-frame camera images."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", duration=timestamp)
        if not self._calibrated:
            self._log_calibration(video, intrinsics_n8)
            self._calibrated = True
        for index in range(len(self.cam_names)):
            image_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(
                images_bgr_n3hw[index].numpy(), "c h w -> h w c"
            )
            rr.log(
                self._image_path(index),
                rr.Image(image_hw3, color_model=rr.ColorModel.BGR).compress(
                    jpeg_quality=75
                ),
            )

    def _log_calibration(
        self,
        video: "DepthVideo",
        intrinsics_n8: Float[Tensor, "n_dirs 8"],
    ) -> None:
        """Log static rig extrinsics and per-camera pinholes."""
        rig_from_cameras_n7: Float[Tensor, "n_cams 7"] = torch.cat(
            [transform.data.reshape(1, 7) for transform in video.T_ci_c0]
        ).cpu()
        translations_n3, quaternions_n4 = _w2c_vecs_to_world_t_cams(
            rig_from_cameras_n7, self.map_scale
        )
        for index in range(len(self.cam_names)):
            rr.log(
                self._cam_path(index),
                rr.Transform3D(
                    translation=translations_n3[index],
                    quaternion=rr.Quaternion(xyzw=quaternions_n4[index]),
                ),
                static=True,
            )
            fx, fy, cx, cy = (float(value) for value in intrinsics_n8[index, :4])
            rr.log(
                self._pinhole_path(index),
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[video.wd, video.ht],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=self.image_plane_distance,
                ),
                static=True,
            )

    def log_keyframe(self, video: "DepthVideo", viz_idx: Tensor) -> None:
        """Log the newest valid rig pose, trajectory, and per-camera depth."""
        candidates: list[int] = sorted(viz_idx.tolist(), reverse=True)
        frame_index: int = next(
            (
                candidate
                for candidate in candidates
                if (video.disps_up[candidate] > 0).any()
            ),
            candidates[0],
        )
        translations_n3, quaternions_n4 = _w2c_vecs_to_world_t_cams(
            video.poses[frame_index][None], self.map_scale
        )
        rr.log(
            "world/rig",
            rr.Transform3D(
                translation=translations_n3[0],
                quaternion=rr.Quaternion(xyzw=quaternions_n4[0]),
            ),
        )

        keyframe_count: int = video.counter.value
        centers_n3, _ = _w2c_vecs_to_world_t_cams(
            video.poses[:keyframe_count], self.map_scale
        )
        self._record_trajectory(centers_n3)
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [centers_n3],
                colors=[[0, 200, 255]],
                radii=self.trajectory_radius,
            ),
        )

        for index in range(len(self.cam_names)):
            disparity_hw: Float[Tensor, "h w"] = (
                video.disps_up_list[index][frame_index].detach().cpu()
            )
            prior_ok_hw: Tensor = prior_valid(
                video.normals_list[index][frame_index][None].cpu()
            )[0]
            valid_hw: Tensor = (
                (disparity_hw > 0.0)
                & (self.map_scale < self.max_depth * disparity_hw)
                & prior_ok_hw
            )
            depth_hw: Float[Tensor, "h w"] = torch.where(
                valid_hw,
                self.map_scale / disparity_hw.clamp(min=1e-6),
                torch.zeros((), dtype=disparity_hw.dtype),
            )
            depth_mm_u16: np.ndarray = (
                depth_hw[::2, ::2].numpy() * 1000.0
            ).clip(0, 65535).astype(np.uint16)
            rr.log(
                self._depth_path(index),
                rr.DepthImage(depth_mm_u16, meter=1000.0),
            )
