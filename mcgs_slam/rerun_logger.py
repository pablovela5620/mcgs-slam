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

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from lietorch import SE3
from torch import Tensor

SH_C0: float = 0.28209479177387814
"""Degree-zero real spherical-harmonic basis constant used by INRIA 3DGS."""


def _w2c_vec_to_world_t_cam(pose_w2c: Float[Tensor, "7"], scale: float) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "4"]]:
    """Invert a lietorch w2c SE3 7-vector into (translation, quaternion_xyzw), scaling translation.

    Args:
        pose_w2c: [tx ty tz qx qy qz qw] camera-from-world, metric units.
        scale: translation scale factor into the Gaussian-map world.
    """
    pose_scaled: Float[Tensor, "7"] = pose_w2c.detach().float().cpu().clone()
    pose_scaled[:3] *= scale
    world_t_cam_vec: Float[np.ndarray, "7"] = SE3(pose_scaled[None]).inv().data[0].numpy()
    return world_t_cam_vec[:3], world_t_cam_vec[3:7]


class RerunLogger:
    """Logs MCGS-SLAM state (cameras, depths, trajectory, Gaussian map) to Rerun."""

    def __init__(
        self,
        cam_names: list[str],
        stream_indices: list[int],
        scale_factor: float,
        save_path: str | None = None,
        spawn: bool = False,
        splat_every: int = 10,
        splat_cap: int = 250_000,
        max_splat_scale: float = 8.0,
        max_depth: float = 60.0,
    ) -> None:
        """
        Args:
            cam_names: one name per logged camera in VIDEO order — the pipeline
                drops input dir 1 (the stereo-right duplicate) when filling the
                video buffers, so video order is [dir0, dir2, dir3, ...]. The
                GS packets, disps_up_list and T_cami_cam0 all use this order.
            stream_indices: for each camera, its column in the raw input stream
                (imagedir order), e.g. [0, 2, 3]; used for per-frame images and
                intrinsics, which arrive in stream order.
            scale_factor: pose scale used by the Gaussian backend (0.2).
            save_path: write a .rrd recording here; preferred for headless runs.
            spawn: spawn a live viewer instead of / in addition to saving.
            splat_every: log a Gaussian-map snapshot every N keyframe updates.
            splat_cap: random-subsample intermediate snapshots to this many splats
                (the final map is always logged in full).
            max_splat_scale: drop splats whose largest axis exceeds this (scaled
                world units); culls the degenerate spike floaters, not the map.
            max_depth: estimated depth beyond this (in scaled world units) is
                logged as 0 (invalid) instead of saturating the colormap.
        """
        if len(stream_indices) != len(cam_names):
            raise ValueError(f"{len(cam_names)} camera names but {len(stream_indices)} stream indices")
        self.cam_names = cam_names
        self.stream_indices = stream_indices
        self.scale = scale_factor
        self.splat_every = splat_every
        self.splat_cap = splat_cap
        self.max_splat_scale = max_splat_scale
        self.max_depth = max_depth
        self._kf_updates: int = 0
        self._scene_bbox: tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]] | None = None
        self._traj_centers: Float[np.ndarray, "n_kf 3"] | None = None

        rr.init("mcgs_slam")
        if save_path is not None:
            rr.save(save_path)
        if spawn:
            rr.spawn(memory_limit="8GB")
        rr.send_blueprint(self._blueprint())
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    def _cam_path(self, cam_idx: int) -> str:
        return f"world/rig/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _compare_path(self, cam_idx: int) -> str:
        return f"render_vs_gt/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _blueprint(self, eye_controls=None) -> rrb.Blueprint:
        image_views = [
            rrb.Spatial2DView(
                origin=f"{self._cam_path(i)}/pinhole",
                name=self.cam_names[i],
                contents=[f"+ $origin/image"],
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
            ["+ /**", "- render_vs_gt/**"]
            + [f"- {self._cam_path(i)}/pinhole/depth" for i in range(len(self.cam_names))]
        )
        view_3d_kwargs = {} if eye_controls is None else {"eye_controls": eye_controls}
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", name="3D map", contents=contents_3d, **view_3d_kwargs),
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

    def set_frame(self, frame_idx: int, timestamp: float) -> None:
        """Advance the frame/time timelines; call once per input frame."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", duration=timestamp)

    def log_calibration(
        self,
        T_cami_cam0: Float[Tensor, "n_cams 7"],
        intrinsics: Float[Tensor, "n_dirs 8"],
        width: int,
        height: int,
    ) -> None:
        """Log static rig extrinsics and per-camera pinhole intrinsics.

        Args:
            T_cami_cam0: cam_i-from-cam_0 SE3 7-vectors (metric), row 0 = identity.
            intrinsics: per image dir [fx fy cx cy ...] at processed resolution.
            width: processed image width.
            height: processed image height.
        """
        for i in range(len(self.cam_names)):
            translation, quat_xyzw = _w2c_vec_to_world_t_cam(T_cami_cam0[i], self.scale)
            rr.log(
                self._cam_path(i),
                rr.Transform3D(translation=translation, quaternion=rr.Quaternion(xyzw=quat_xyzw)),
                static=True,
            )
            fx, fy, cx, cy = (float(v) for v in intrinsics[self.stream_indices[i], :4])
            rr.log(
                f"{self._cam_path(i)}/pinhole",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[width, height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.4,
                ),
                static=True,
            )

    def log_images(self, images_bgr: UInt8[Tensor, "n_dirs 3 h w"]) -> None:
        """Log per-frame camera images (BGR, in raw stream order)."""
        for i in range(len(self.cam_names)):
            image_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(images_bgr[self.stream_indices[i]].numpy(), "c h w -> h w c")
            rr.log(
                f"{self._cam_path(i)}/pinhole/image",
                rr.Image(image_hw3, color_model=rr.ColorModel.BGR).compress(jpeg_quality=75),
            )

    def log_keyframe(self, video, viz_idx: Tensor) -> None:
        """Log rig pose, trajectory and per-camera depth for the latest keyframe.

        Args:
            video: the DepthVideo holding poses/disps (poses are w2c, metric).
            viz_idx: keyframe indices updated by the frontend in this step.
        """
        # The newest keyframe's upsampled disparity may not be computed yet;
        # log the most recent keyframe that has one.
        idx: int = int(viz_idx.max().item())
        for candidate in sorted((int(i.item()) for i in viz_idx), reverse=True):
            if bool((video.disps_up[candidate] > 0).any()):
                idx = candidate
                break

        translation, quat_xyzw = _w2c_vec_to_world_t_cam(video.poses[idx], self.scale)
        rr.log("world/rig", rr.Transform3D(translation=translation, quaternion=rr.Quaternion(xyzw=quat_xyzw)))

        n_kf: int = video.counter.value
        poses_scaled: Float[Tensor, "n_kf 7"] = video.poses[:n_kf].detach().float().cpu().clone()
        poses_scaled[:, :3] *= self.scale
        centers: Float[np.ndarray, "n_kf 3"] = SE3(poses_scaled).inv().data[:, :3].numpy()
        self._traj_centers: Float[np.ndarray, "n_kf 3"] = centers
        rr.log("world/trajectory", rr.LineStrips3D([centers], colors=[[0, 200, 255]], radii=0.01))

        disps_up_list = video.disps_up_list if video.multi else [video.disps_up]
        for i in range(len(self.cam_names)):
            disp_hw: Float[Tensor, "h w"] = disps_up_list[i][idx].detach().cpu()
            depth_hw: Float[Tensor, "h w"] = torch.where(disp_hw > 0, self.scale / disp_hw.clamp(min=1e-6), torch.zeros(()))
            depth_hw = torch.where(depth_hw < self.max_depth, depth_hw, torch.zeros(()))
            depth_mm_u16: np.ndarray = (depth_hw[::2, ::2].numpy() * 1000.0).clip(0, 65535).astype(np.uint16)
            rr.log(f"{self._cam_path(i)}/pinhole/depth", rr.DepthImage(depth_mm_u16, meter=1000.0))

    def log_gaussians(self, gaussians, force: bool = False, cap: bool = True) -> None:
        """Log a snapshot of the Gaussian map with rr.GaussianSplats3D.

        Args:
            gaussians: the GaussianModel (INRIA parameterization, sh_degree 0).
            force: bypass the splat_every cadence.
            cap: subsample to splat_cap splats; pass False for the final map.
        """
        self._kf_updates += 1
        if not force and (self._kf_updates % self.splat_every) != 0:
            return
        with torch.no_grad():
            centers: Float[np.ndarray, "n 3"] = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
            if centers.shape[0] == 0:
                return
            scales: Float[np.ndarray, "n 3"] = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
            quats_wxyz: Float[np.ndarray, "n 4"] = gaussians.get_rotation.detach().cpu().numpy().astype(np.float32)
            opacity: Float[np.ndarray, "n"] = gaussians.get_opacity.detach().cpu().numpy().astype(np.float32).reshape(-1)
            f_dc: Float[np.ndarray, "n 3"] = (
                gaussians._features_dc.detach().cpu().numpy().astype(np.float32).reshape(centers.shape[0], 3)
            )

        sane = scales.max(axis=1) < self.max_splat_scale
        centers, scales, quats_wxyz, opacity, f_dc = (
            centers[sane], scales[sane], quats_wxyz[sane], opacity[sane], f_dc[sane],
        )
        if centers.shape[0] == 0:
            return
        if cap and centers.shape[0] > self.splat_cap:
            keep = np.random.default_rng(0).choice(centers.shape[0], self.splat_cap, replace=False)
            centers, scales, quats_wxyz, opacity, f_dc = (
                centers[keep], scales[keep], quats_wxyz[keep], opacity[keep], f_dc[keep],
            )
        self._scene_bbox = (
            np.percentile(centers, 2, axis=0).astype(np.float64),
            np.percentile(centers, 98, axis=0).astype(np.float64),
        )

        quats_xyzw: Float[np.ndarray, "n 4"] = quats_wxyz[:, [1, 2, 3, 0]]
        rgb01: Float[np.ndarray, "n 3"] = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
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
            return
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
        if self._scene_bbox is None:
            return
        lo, hi = self._scene_bbox
        center: Float[np.ndarray, "3"] = (lo + hi) / 2.0
        extent: float = float(np.linalg.norm(hi - lo))
        # World is RDF (x right, y down, z forward): up is -y. Prefer a
        # street-level chase view: behind the trajectory start, slightly
        # elevated, looking down the drive corridor at the scene center.
        if self._traj_centers is not None and len(self._traj_centers) >= 2:
            start: Float[np.ndarray, "3"] = self._traj_centers[0].astype(np.float64)
            forward: Float[np.ndarray, "3"] = center - start
            forward[1] = 0.0
            norm: float = float(np.linalg.norm(forward))
            forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            eye: Float[np.ndarray, "3"] = start - 0.45 * extent * forward + np.array([0.0, -0.18 * extent, 0.0])
        else:
            eye = center + np.array([0.0, -0.9 * extent, -0.7 * extent])
        eye_controls = rrb.archetypes.EyeControls3D(
            kind=rrb.components.Eye3DKind.Orbital,
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
