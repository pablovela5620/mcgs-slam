"""Rerun 0.36 logging for MCGS-SLAM.

Entity schema (follows the multi-camera rig layout of rerun-io's robocap
example):

    /                                 ViewCoordinates.RDF (world = first cam0 frame)
    world/splats                      rr.GaussianSplats3D (map snapshots + final map)
    world/rig                         rr.Transform3D world_T_rig, updated per keyframe
    world/rig/cam_NN                  static rr.Transform3D rig_T_cam (from calib)
    world/rig/cam_NN/pinhole          static rr.Pinhole intrinsics
    world/rig/cam_NN/pinhole/image    per-frame RGB (jpeg-compressed)
    world/rig/cam_NN/pinhole/depth    per-keyframe estimated depth (u16 mm, half res)
    world/trajectory                  rr.LineStrips3D of the rig path, re-logged per keyframe
    render_vs_gt/cam_NN/rendered      Gaussian-map render of the latest keyframe view
    render_vs_gt/cam_NN/gt            the matching ground-truth image

All 3D quantities are in the SLAM world scaled by ``scale_factor`` (0.2), the
same frame the Gaussian map is optimized in, so splats, frustums and the
trajectory line up without any post-hoc alignment.

Timelines: ``frame`` (input frame index), ``time`` (image timestamps, seconds)
and ``refine_iter`` (final color-refinement iterations).
"""

import os
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from lietorch import SE3
from torch import Tensor

from gaussian.utils.sh_utils import SH2RGB

if TYPE_CHECKING:
    from depth_video import DepthVideo
    from gaussian.scene.gaussian_model import GaussianModel

COMPARE_ROOT: str = "render_vs_gt"
"""Entity root for the splat-render vs ground-truth image pairs."""


def _w2c_vecs_to_world_t_cams(
    poses_w2c: Float[Tensor, "n 7"], scale: float
) -> tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 4"]]:
    """Invert lietorch w2c SE3 7-vectors into (translations, quaternions_xyzw), scaling translations.

    Args:
        poses_w2c: [tx ty tz qx qy qz qw] camera-from-world, metric units.
        scale: translation scale factor into the Gaussian-map world.
    """
    poses_scaled: Float[Tensor, "n 7"] = poses_w2c.detach().float().cpu().clone()
    poses_scaled[:, :3] *= scale
    world_t_cam_vecs: Float[np.ndarray, "n 7"] = SE3(poses_scaled).inv().data.numpy()
    return world_t_cam_vecs[:, :3], world_t_cam_vecs[:, 3:7]


def _percentile_bbox(points: Float[np.ndarray, "n 3"]) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
    """2/98-percentile bounds; robust to far-flung outlier points."""
    return (
        np.percentile(points, 2, axis=0).astype(np.float64),
        np.percentile(points, 98, axis=0).astype(np.float64),
    )


class RerunLogger:
    """Logs MCGS-SLAM state (cameras, depths, trajectory, Gaussian map) to Rerun."""

    def __init__(
        self,
        imagedirs: list[str],
        stream_indices: list[int],
        scale_factor: float,
        save_path: str | None = None,
        spawn: bool = False,
        splat_every: int = 10,
        splat_cap: int = 75_000,
        max_splat_scale: float = 8.0,
        max_depth: float = 60.0,
        refine_every: int = 10_000,
    ) -> None:
        """
        Args:
            imagedirs: the raw input image directories, in stream order.
            stream_indices: for each logged camera (VIDEO order — the pipeline
                drops input dir 1, the stereo-right duplicate), its column in
                the raw input stream, e.g. [0, 2, 3]. Camera names are derived
                from the corresponding imagedirs.
            scale_factor: pose scale used by the Gaussian backend (0.2).
            save_path: write a .rrd recording here.
            spawn: also stream to a spawned live viewer.
            splat_every: log a Gaussian map snapshot every N keyframe updates.
            splat_cap: random-subsample intermediate snapshots to this many splats
                (the final map is always logged in full).
            max_splat_scale: drop splats whose largest axis exceeds this (scaled
                world units); culls the degenerate spike floaters, not the map.
            max_depth: estimated depth beyond this (in scaled world units) is
                logged as 0 (invalid) instead of saturating the colormap.
            refine_every: snapshot cadence (iterations) during color refinement.
        """
        self.cam_names = [os.path.basename(os.path.normpath(imagedirs[i])) for i in stream_indices]
        self.stream_indices = stream_indices
        self.scale = scale_factor
        self.splat_every = splat_every
        self.splat_cap = splat_cap
        self.max_splat_scale = max_splat_scale
        self.max_depth = max_depth
        self.refine_every = refine_every
        self._kf_updates: int = 0
        self._calibrated: bool = False
        self._scene_bbox: tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]] | None = None
        self._traj_centers: Float[np.ndarray, "n_kf 3"] | None = None

        rr.init("mcgs_slam")
        # Sinks replace each other in rerun >= 0.24, so a save() followed by
        # spawn() would silently drop the recording — set both at once.
        sinks: list = []
        if save_path is not None:
            sinks.append(rr.FileSink(save_path))
        if spawn:
            rr.spawn(connect=False, memory_limit="8GB")
            sinks.append(rr.GrpcSink())
        if not sinks:
            raise ValueError("RerunLogger needs a .rrd path and/or a spawned viewer (--rrd / --rerun-spawn)")
        rr.set_sinks(*sinks)
        rr.send_blueprint(self._blueprint())
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    def _cam_path(self, cam_idx: int) -> str:
        return f"world/rig/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _compare_path(self, cam_idx: int) -> str:
        return f"{COMPARE_ROOT}/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _blueprint(self, eye_controls: rrb.EyeControls3D | None = None) -> rrb.Blueprint:
        image_views = [
            rrb.Spatial2DView(
                origin=f"{self._cam_path(i)}/pinhole",
                name=self.cam_names[i],
                contents=["+ $origin/image"],
            )
            for i in range(len(self.cam_names))
        ]
        depth_views = [
            rrb.Spatial2DView(origin=f"{self._cam_path(i)}/pinhole/depth", name=f"{self.cam_names[i]} depth")
            for i in range(len(self.cam_names))
        ]
        # Per camera: Gaussian-map render on top, ground-truth image below.
        compare_views = [
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{self._compare_path(i)}/rendered", name=f"{self.cam_names[i]} render"),
                rrb.Spatial2DView(origin=f"{self._compare_path(i)}/gt", name=f"{self.cam_names[i]} GT"),
                name=self.cam_names[i],
            )
            for i in range(len(self.cam_names))
        ]
        # Keep estimated depth images out of the 3D view: their automatic
        # backprojection would double up with the Gaussian map. render_vs_gt
        # has no spatial context and doesn't belong in 3D either.
        contents_3d: list[str] = (
            ["+ /**", f"- {COMPARE_ROOT}/**"]
            + [f"- {self._cam_path(i)}/pinhole/depth" for i in range(len(self.cam_names))]
        )
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", name="3D map", contents=contents_3d, eye_controls=eye_controls),
                rrb.Vertical(
                    rrb.Horizontal(*compare_views, name="render vs GT"),
                    rrb.Horizontal(*image_views),
                    rrb.Horizontal(*depth_views),
                    row_shares=[2.0, 1.0, 1.0],
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    def log_frame(
        self,
        frame_idx: int,
        timestamp: float,
        images_bgr: UInt8[Tensor, "n_dirs 3 h w"],
        intrinsics: Float[Tensor, "n_dirs 8"],
        video: "DepthVideo",
    ) -> None:
        """Advance the timelines and log the per-frame camera images.

        On the first call, also logs the static rig calibration (extrinsics
        from video.T_ci_c0, pinholes from the stream intrinsics).
        """
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", duration=timestamp)

        if not self._calibrated:
            self._log_calibration(video, intrinsics)
            self._calibrated = True

        for i in range(len(self.cam_names)):
            image_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(images_bgr[self.stream_indices[i]].numpy(), "c h w -> h w c")
            rr.log(
                f"{self._cam_path(i)}/pinhole/image",
                rr.Image(image_hw3, color_model=rr.ColorModel.BGR).compress(jpeg_quality=75),
            )

    def _log_calibration(self, video: "DepthVideo", intrinsics: Float[Tensor, "n_dirs 8"]) -> None:
        """Log static rig extrinsics (video.T_ci_c0) and per-camera pinholes."""
        T_rig_cams: Float[Tensor, "n_cams 7"] = torch.cat([T.data.reshape(1, 7) for T in video.T_ci_c0]).cpu()
        translations, quats_xyzw = _w2c_vecs_to_world_t_cams(T_rig_cams, self.scale)
        for i in range(len(self.cam_names)):
            rr.log(
                self._cam_path(i),
                rr.Transform3D(translation=translations[i], quaternion=rr.Quaternion(xyzw=quats_xyzw[i])),
                static=True,
            )
            fx, fy, cx, cy = (float(v) for v in intrinsics[self.stream_indices[i], :4])
            rr.log(
                f"{self._cam_path(i)}/pinhole",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[video.wd, video.ht],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.4,
                ),
                static=True,
            )

    def log_keyframe(self, video: "DepthVideo", viz_idx: Tensor) -> None:
        """Log rig pose, trajectory and per-camera depth for the latest keyframe.

        Args:
            video: the DepthVideo holding poses/disps (poses are w2c, metric).
            viz_idx: keyframe indices updated by the frontend in this step.
        """
        # The newest keyframe's upsampled disparity may not be computed yet;
        # log the most recent keyframe that has one.
        candidates: list[int] = sorted(viz_idx.tolist(), reverse=True)
        idx: int = next((c for c in candidates if (video.disps_up[c] > 0).any()), candidates[0])

        translations, quats_xyzw = _w2c_vecs_to_world_t_cams(video.poses[idx][None], self.scale)
        rr.log("world/rig", rr.Transform3D(translation=translations[0], quaternion=rr.Quaternion(xyzw=quats_xyzw[0])))

        n_kf: int = video.counter.value
        centers, _ = _w2c_vecs_to_world_t_cams(video.poses[:n_kf], self.scale)
        self._traj_centers = centers
        rr.log("world/trajectory", rr.LineStrips3D([centers], colors=[[0, 200, 255]], radii=0.01))

        disps_up_list = video.disps_up_list if video.multi else [video.disps_up]
        for i in range(len(self.cam_names)):
            disp_hw: Float[Tensor, "h w"] = disps_up_list[i][idx].detach().cpu()
            valid_hw = (disp_hw > 0) & (self.scale < self.max_depth * disp_hw)
            depth_hw: Float[Tensor, "h w"] = torch.where(valid_hw, self.scale / disp_hw.clamp(min=1e-6), torch.zeros(()))
            depth_mm_u16: np.ndarray = (depth_hw[::2, ::2].numpy() * 1000.0).clip(0, 65535).astype(np.uint16)
            rr.log(f"{self._cam_path(i)}/pinhole/depth", rr.DepthImage(depth_mm_u16, meter=1000.0))

    def log_gaussians(self, gaussians: "GaussianModel", force: bool = False, cap: bool = True) -> None:
        """Log a snapshot of the Gaussian map with rr.GaussianSplats3D.

        Call once per keyframe update (the cadence is internal); force=True for
        milestone snapshots, cap=False for the final full map.
        """
        if not force:
            self._kf_updates += 1
            if (self._kf_updates % self.splat_every) != 0:
                return
        with torch.no_grad():
            centers: Float[np.ndarray, "n 3"] = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
            if centers.shape[0] == 0:
                return
            scales: Float[np.ndarray, "n 3"] = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
            quats_wxyz: Float[np.ndarray, "n 4"] = gaussians.get_rotation.detach().cpu().numpy().astype(np.float32)
            opacity: Float[np.ndarray, "n"] = gaussians.get_opacity.detach().cpu().numpy().astype(np.float32).reshape(-1)
            f_dc: Float[np.ndarray, "n 3"] = (
                gaussians.get_features[:, 0, :].detach().cpu().numpy().astype(np.float32)
            )

        keep: np.ndarray = np.flatnonzero(scales.max(axis=1) < self.max_splat_scale)
        if keep.size == 0:
            return
        if cap and keep.size > self.splat_cap:
            keep = np.random.default_rng(0).choice(keep, self.splat_cap, replace=False)
        centers, scales, quats_wxyz, opacity, f_dc = (
            centers[keep], scales[keep], quats_wxyz[keep], opacity[keep], f_dc[keep],
        )
        self._scene_bbox = _percentile_bbox(centers)

        quats_xyzw: Float[np.ndarray, "n 4"] = quats_wxyz[:, [1, 2, 3, 0]]
        rgb01: Float[np.ndarray, "n 3"] = np.clip(SH2RGB(f_dc), 0.0, 1.0)
        rgba: UInt8[np.ndarray, "n 4"] = np.clip(
            np.concatenate([rgb01, opacity[:, None]], axis=1) * 255.0 + 0.5, 0.0, 255.0
        ).astype(np.uint8)

        rr.log(
            "world/splats",
            rr.GaussianSplats3D(
                centers=centers,
                scales=scales,
                quaternions=quats_xyzw,
                colors=rgba,
                spherical_harmonics_degree=0,
            ),
        )

    def log_render_comparison(
        self,
        cam_idx: int,
        rendered_3hw: Float[Tensor, "3 h w"],
        gt_3hw: Float[Tensor, "3 h w"],
    ) -> None:
        """Log a Gaussian-map render next to its ground-truth view (both RGB in [0, 1])."""
        if cam_idx >= len(self.cam_names):
            raise ValueError(f"cam_idx {cam_idx} out of range for {len(self.cam_names)} logged cameras")
        for name, img_3hw in (("rendered", rendered_3hw), ("gt", gt_3hw)):
            img_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(
                (img_3hw.detach().clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy(), "c h w -> h w c"
            )
            rr.log(f"{self._compare_path(cam_idx)}/{name}", rr.Image(img_hw3).compress(jpeg_quality=80))

    def send_final_blueprint(self) -> None:
        """Re-send the blueprint with the 3D eye framing the reconstructed scene.

        The 2/98 percentile bounds of the last splat snapshot are used so
        far-flung sky splats don't push the camera out (robocap/gauss-surf tip).
        """
        if self._traj_centers is None:
            return
        lo, hi = self._scene_bbox if self._scene_bbox is not None else _percentile_bbox(self._traj_centers)
        center: Float[np.ndarray, "3"] = (lo + hi) / 2.0
        extent: float = float(np.linalg.norm(hi - lo))
        # World is RDF (x right, y down, z forward): up is -y. Street-level
        # chase view: behind the trajectory start, slightly elevated, looking
        # down the drive corridor at the scene center.
        start: Float[np.ndarray, "3"] = self._traj_centers[0].astype(np.float64)
        forward: Float[np.ndarray, "3"] = center - start
        forward[1] = 0.0
        norm: float = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        eye: Float[np.ndarray, "3"] = start - 0.45 * extent * forward + np.array([0.0, -0.18 * extent, 0.0])
        eye_controls = rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=eye.tolist(),
            look_target=center.tolist(),
            eye_up=[0.0, -1.0, 0.0],
        )
        rr.send_blueprint(self._blueprint(eye_controls=eye_controls))

    def set_refine_iter(self, iteration: int) -> None:
        """Advance the color-refinement timeline (frame/time stay at their last values)."""
        rr.set_time("refine_iter", sequence=iteration)

    def log_text(self, text: str) -> None:
        """Log a status line to the slam_metrics text log."""
        rr.log("slam_metrics", rr.TextLog(text))
