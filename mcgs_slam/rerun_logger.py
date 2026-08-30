"""Schema-neutral Rerun logging shared by the Waymo and catalog paths."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from torch import Tensor

from gaussian.utils.sh_utils import SH2RGB

if TYPE_CHECKING:
    from gaussian.scene.gaussian_model import GaussianModel


@dataclass(slots=True, frozen=True)
class DashboardSpec:
    """Schema-specific paths and view controls for the shared dashboard."""

    image_origins: tuple[str, ...]
    """One image-view origin per camera."""
    image_contents: tuple[str, ...] | None
    """Optional common image-view contents query."""
    depth_origins: tuple[str, ...]
    """One depth-view origin per camera."""
    render_origins: tuple[str, ...]
    """One rendered-image origin per camera."""
    ground_truth_origins: tuple[str, ...]
    """One comparison ground-truth origin per camera."""
    excluded_3d_paths: tuple[str, ...]
    """Entity paths excluded from both 3D views."""
    follow_origin: str
    """Entity origin followed by the secondary 3D view."""
    follow_exclusions: tuple[str, ...]
    """Extra entity paths excluded only from the follow view."""
    follow_eye: rrb.EyeControls3D
    """Fixed camera controls for the follow view."""


def _percentile_bbox(
    points_n3: Float[np.ndarray, "n 3"],
) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
    """Return robust 2/98-percentile bounds for 3D points."""
    return (
        np.percentile(points_n3, 2, axis=0).astype(np.float64),
        np.percentile(points_n3, 98, axis=0).astype(np.float64),
    )


class RerunLogger:
    """Own Rerun sinks and schema-neutral Gaussian-map logging."""

    def __init__(
        self,
        *,
        camera_names: list[str],
        camera_ids: list[int],
        map_scale: float,
        world_coordinates: rr.ViewCoordinates,
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
        if len(camera_ids) != len(camera_names):
            raise ValueError("camera_ids and camera_names must have the same length")
        self.cam_names: list[str] = camera_names
        self.camera_ids: list[int] = camera_ids
        self.map_scale: float = map_scale
        self.world_coordinates: rr.ViewCoordinates = world_coordinates
        self.splat_every: int = splat_every
        self.splat_cap: int = splat_cap
        self.max_splat_scale: float = max_splat_scale
        self.splat_scale_percentile: float = splat_scale_percentile
        self.max_depth: float = max_depth
        self.refine_every: int = refine_every
        self.image_plane_distance: float = image_plane_distance
        self.trajectory_radius: float = trajectory_radius
        self._kf_updates: int = 0
        self._scene_bbox: tuple[
            Float[np.ndarray, "3"], Float[np.ndarray, "3"]
        ] | None = None
        self._trajectory_start: Float[np.ndarray, "3"] | None = None
        self._trajectory_lo: Float[np.ndarray, "3"] | None = None
        self._trajectory_hi: Float[np.ndarray, "3"] | None = None

        rr.init("mcgs_slam")
        sinks: list[object] = []
        if save_path is not None:
            sinks.append(rr.FileSink(save_path))
        if spawn:
            rr.spawn(connect=False, memory_limit="8GB")
            sinks.append(rr.GrpcSink())
        if not sinks:
            raise ValueError(
                "Rerun logger needs a .rrd path and/or a spawned viewer "
                "(--rrd / --rerun-spawn)"
            )
        rr.set_sinks(*sinks)
        rr.log("/", self.world_coordinates, static=True)

    def _dashboard_spec(self) -> DashboardSpec:
        """Return schema-specific paths and controls for the shared dashboard."""
        raise NotImplementedError

    def _blueprint(
        self,
        eye_controls: rrb.EyeControls3D | None = None,
    ) -> rrb.Blueprint:
        """Build the common image, depth, comparison, and 3D dashboard."""
        spec: DashboardSpec = self._dashboard_spec()
        camera_count: int = len(self.cam_names)
        path_groups: tuple[tuple[str, ...], ...] = (
            spec.image_origins,
            spec.depth_origins,
            spec.render_origins,
            spec.ground_truth_origins,
        )
        if any(len(paths) != camera_count for paths in path_groups):
            raise ValueError("dashboard paths must match the logged camera count")

        if spec.image_contents is None:
            image_views: list[rrb.Spatial2DView] = [
                rrb.Spatial2DView(
                    origin=spec.image_origins[index],
                    name=self.cam_names[index],
                )
                for index in range(camera_count)
            ]
        else:
            image_views = [
                rrb.Spatial2DView(
                    origin=spec.image_origins[index],
                    name=self.cam_names[index],
                    contents=spec.image_contents,
                )
                for index in range(camera_count)
            ]
        depth_views: list[rrb.Spatial2DView] = [
            rrb.Spatial2DView(
                origin=spec.depth_origins[index],
                name=f"{self.cam_names[index]} depth",
            )
            for index in range(camera_count)
        ]
        comparison_views: list[object] = [
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=spec.render_origins[index],
                    name=f"{self.cam_names[index]} render",
                ),
                rrb.Spatial2DView(
                    origin=spec.ground_truth_origins[index],
                    name=f"{self.cam_names[index]} GT",
                ),
                name=self.cam_names[index],
            )
            for index in range(camera_count)
        ]
        contents_3d: list[str] = [
            "+ /**",
            *(f"- {path}" for path in spec.excluded_3d_paths),
        ]
        follow_contents_3d: list[str] = [
            *contents_3d,
            *(f"- {path}" for path in spec.follow_exclusions),
        ]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        origin="/",
                        name="3D map",
                        contents=contents_3d,
                        eye_controls=eye_controls,
                    ),
                    rrb.Spatial3DView(
                        name="Follow",
                        origin=spec.follow_origin,
                        contents=follow_contents_3d,
                        eye_controls=spec.follow_eye,
                    ),
                    row_shares=[3.0, 2.0],
                ),
                rrb.Vertical(
                    rrb.Horizontal(*comparison_views, name="render vs GT"),
                    rrb.Horizontal(*image_views),
                    rrb.Horizontal(*depth_views),
                    row_shares=[2.0, 1.0, 1.0],
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    def send_blueprint(
        self,
        eye_controls: rrb.EyeControls3D | None = None,
    ) -> None:
        """Build and send the dashboard after concrete logger initialization."""
        rr.send_blueprint(self._blueprint(eye_controls=eye_controls))

    def _render_path(self, cam_idx: int) -> str:
        """Return the concrete logger's render entity path."""
        raise NotImplementedError

    def _ground_truth_path(self, cam_idx: int) -> str | None:
        """Return a separate GT entity, or ``None`` when an image already exists."""
        raise NotImplementedError

    def _record_trajectory(
        self,
        centers_n3: Float[np.ndarray, "n 3"],
    ) -> None:
        """Update the compact trajectory bounds used by final eye framing."""
        if len(centers_n3) == 0:
            return
        points_n3: Float[np.ndarray, "n 3"] = np.asarray(
            centers_n3, dtype=np.float64
        )
        if self._trajectory_start is None:
            self._trajectory_start = points_n3[0].copy()
            self._trajectory_lo = points_n3.min(axis=0)
            self._trajectory_hi = points_n3.max(axis=0)
            return
        assert self._trajectory_lo is not None and self._trajectory_hi is not None
        self._trajectory_lo = np.minimum(self._trajectory_lo, points_n3.min(axis=0))
        self._trajectory_hi = np.maximum(self._trajectory_hi, points_n3.max(axis=0))

    def log_gaussians(
        self,
        gaussians: "GaussianModel",
        force: bool = False,
        cap: bool = True,
    ) -> None:
        """Log one Gaussian-map snapshot with stable percentile filtering."""
        if not force:
            self._kf_updates += 1
            if (self._kf_updates % self.splat_every) != 0:
                return
        with torch.no_grad():
            centers_n3: Float[np.ndarray, "n 3"] = (
                gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
            )
            if centers_n3.shape[0] == 0:
                return
            scales_n3: Float[np.ndarray, "n 3"] = (
                gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
            )
            quats_wxyz_n4: Float[np.ndarray, "n 4"] = (
                gaussians.get_rotation.detach().cpu().numpy().astype(np.float32)
            )
            opacity_n: Float[np.ndarray, "n"] = (
                gaussians.get_opacity.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )
            features_dc_n3: Float[np.ndarray, "n 3"] = (
                gaussians.get_features[:, 0, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        largest_axis_n: Float[np.ndarray, "n"] = scales_n3.max(axis=1)
        percentile_cap: float = float(
            np.percentile(largest_axis_n, self.splat_scale_percentile)
        )
        keep_n: np.ndarray = np.flatnonzero(
            (largest_axis_n < self.max_splat_scale)
            & (largest_axis_n <= percentile_cap)
        )
        if keep_n.size == 0:
            return
        if cap and keep_n.size > self.splat_cap:
            keep_n = np.random.default_rng(0).choice(
                keep_n, self.splat_cap, replace=False
            )
        centers_n3, scales_n3, quats_wxyz_n4, opacity_n, features_dc_n3 = (
            centers_n3[keep_n],
            scales_n3[keep_n],
            quats_wxyz_n4[keep_n],
            opacity_n[keep_n],
            features_dc_n3[keep_n],
        )
        self._scene_bbox = _percentile_bbox(centers_n3)

        quats_xyzw_n4: Float[np.ndarray, "n 4"] = quats_wxyz_n4[:, [1, 2, 3, 0]]
        rgb01_n3: Float[np.ndarray, "n 3"] = np.clip(
            SH2RGB(features_dc_n3), 0.0, 1.0
        )
        rgba_n4: UInt8[np.ndarray, "n 4"] = np.clip(
            np.concatenate([rgb01_n3, opacity_n[:, None]], axis=1) * 255.0 + 0.5,
            0.0,
            255.0,
        ).astype(np.uint8)
        rr.log(
            "world/splats",
            rr.GaussianSplats3D(
                centers=centers_n3,
                scales=scales_n3,
                quaternions=quats_xyzw_n4,
                colors=rgba_n4,
                spherical_harmonics_degree=0,
            ),
        )

    def log_render_comparison(
        self,
        cam_idx: int,
        rendered_3hw: Float[Tensor, "3 h w"],
        gt_3hw: Float[Tensor, "3 h w"],
    ) -> None:
        """Log a Gaussian render and, when needed, its separate ground truth."""
        if cam_idx >= len(self.cam_names):
            raise ValueError(
                f"cam_idx {cam_idx} out of range for "
                f"{len(self.cam_names)} logged cameras"
            )
        targets: list[tuple[str, Float[Tensor, "3 h w"]]] = [
            (self._render_path(cam_idx), rendered_3hw)
        ]
        ground_truth_path: str | None = self._ground_truth_path(cam_idx)
        if ground_truth_path is not None:
            targets.append((ground_truth_path, gt_3hw))
        for path, image_3hw in targets:
            image_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(
                (image_3hw.detach().clamp(0.0, 1.0) * 255.0)
                .to(torch.uint8)
                .cpu()
                .numpy(),
                "c h w -> h w c",
            )
            rr.log(path, rr.Image(image_hw3).compress(jpeg_quality=80))

    def send_final_blueprint(self) -> None:
        """Re-send the blueprint with an eye framed from map and trajectory bounds."""
        if self._trajectory_start is None:
            return
        assert self._trajectory_lo is not None and self._trajectory_hi is not None
        bounds: tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]] = (
            self._scene_bbox
            if self._scene_bbox is not None
            else (self._trajectory_lo, self._trajectory_hi)
        )
        lo_3, hi_3 = bounds
        center_3: Float[np.ndarray, "3"] = (lo_3 + hi_3) / 2.0
        extent: float = max(float(np.linalg.norm(hi_3 - lo_3)), 1.0)
        start_3: Float[np.ndarray, "3"] = self._trajectory_start.astype(np.float64)

        if self.world_coordinates == rr.ViewCoordinates.RFU:
            vertical_axis: int = 2
            default_forward_3: np.ndarray = np.array([0.0, 1.0, 0.0])
            up_3: np.ndarray = np.array([0.0, 0.0, 1.0])
        elif self.world_coordinates == rr.ViewCoordinates.RDF:
            vertical_axis = 1
            default_forward_3 = np.array([0.0, 0.0, 1.0])
            up_3 = np.array([0.0, -1.0, 0.0])
        else:
            raise ValueError("final eye framing supports RFU and RDF coordinates")

        forward_3: Float[np.ndarray, "3"] = center_3 - start_3
        forward_3[vertical_axis] = 0.0
        norm: float = float(np.linalg.norm(forward_3))
        forward_3 = forward_3 / norm if norm > 1e-6 else default_forward_3
        eye_3: Float[np.ndarray, "3"] = (
            start_3 - 0.45 * extent * forward_3 + 0.2 * extent * up_3
        )
        eye_controls = rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=eye_3.tolist(),
            look_target=center_3.tolist(),
            eye_up=up_3.tolist(),
        )
        self.send_blueprint(eye_controls=eye_controls)

    def set_refine_iter(self, iteration: int) -> None:
        """Advance the color-refinement timeline."""
        rr.set_time("refine_iter", sequence=iteration)

    def log_text(self, text: str) -> None:
        """Log one mapper status line."""
        rr.log("slam_metrics", rr.TextLog(text))

    def flush(self, timeout_seconds: float = 30.0) -> None:
        """Flush all configured sinks before the process exits."""
        recording = rr.get_global_data_recording()
        if recording is not None:
            recording.flush(timeout_sec=timeout_seconds)
