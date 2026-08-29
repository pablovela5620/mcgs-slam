"""Rerun 0.36 logging for MCGS-SLAM.

Entity schema for catalog mode follows simplecv's ``exoego:v2`` rig layout:

    /                                 root ViewCoordinates (RDF by default; catalog uses RFU)
    world/splats                      rr.GaussianSplats3D (map snapshots + final map)
    world/rig_00                      metadata + temporal rr.Transform3D world_T_rig
    world/rig_00/cam_NN               metadata + static rr.Transform3D rig_T_cam
    world/rig_00/cam_NN/pinhole       native KB4 calibration
    world/rig_00/cam_NN/pinhole/video relayed H.264 packets
    world/rig_00/cam_NN/rectified     virtual pinhole + identity transform
    world/rig_00/cam_NN/rectified/*   image, metric depth, render, and GT
    world/trajectory                  rr.LineStrips3D of the rig path, re-logged per keyframe

All 3D quantities use the same ``scale_factor`` as the Gaussian backend. The
legacy Waymo path defaults to 0.2; the metric catalog path passes 1.0. Splats,
frustums, and the trajectory therefore line up without post-hoc alignment.

Timelines include ``frame`` (input frame index), ``video_time`` for catalog
packets, and ``refine_iter`` (final color-refinement iterations).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from lietorch import SE3
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_custom_types import PinholeWithDistortion
from simplecv.rerun_log_utils import log_pinhole
from simplecv.rerun_rig_logger import SCHEMA_VERSION, log_rig_pose_stream
from simplecv.rig import CameraSensor, Rig, RigCalibration, RigPoseStream
from torch import Tensor

from gaussian.utils.sh_utils import SH2RGB
from prior_mask import prior_valid

if TYPE_CHECKING:
    from catalog_stream import CameraCalibration, CatalogKeyframe, RawVideoStream
    from depth_video import DepthVideo
    from gaussian.scene.gaussian_model import GaussianModel

COMPARE_ROOT: str = "render_vs_gt"
"""Entity root for the splat-render vs ground-truth image pairs."""

CATALOG_RIG_PATH: str = "world/rig_00"
"""RoboCap's single moving ego rig in the exoego:v2 layout."""


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
        imagedirs: list[str] | None = None,
        scale_factor: float = 0.2,
        save_path: str | None = None,
        spawn: bool = False,
        splat_every: int = 10,
        splat_cap: int = 75_000,
        max_splat_scale: float = 8.0,
        splat_scale_percentile: float = 99.7,
        max_depth: float = 60.0,
        refine_every: int = 10_000,
        camera_names: list[str] | None = None,
        camera_ids: list[int] | None = None,
        world_coordinates: rr.ViewCoordinates = rr.ViewCoordinates.RDF,
        image_plane_distance: float = 0.4,
        trajectory_radius: float = 0.01,
        catalog_mode: bool = False,
    ) -> None:
        """
        Args:
            imagedirs: the input image directories, in camera order. Used to
                derive Waymo camera names when ``camera_names`` is absent.
            scale_factor: pose scale used by the Gaussian backend; the Waymo
                default is 0.2 and metric catalog callers pass 1.0.
            save_path: write a .rrd recording here.
            spawn: also stream to a spawned live viewer.
            splat_every: log a Gaussian map snapshot every N keyframe updates.
            splat_cap: random-subsample intermediate snapshots to this many splats
                (the final map is always logged in full).
            max_splat_scale: drop splats whose largest axis exceeds this (scaled
                world units); culls the degenerate spike floaters, not the map.
            splat_scale_percentile: also drop splats above this percentile of largest axis.
            max_depth: estimated depth beyond this (in scaled world units) is
                logged as 0 (invalid) instead of saturating the colormap.
            refine_every: snapshot cadence (iterations) during color refinement.
            camera_names: explicit camera names, used by catalog inputs.
            camera_ids: physical camera IDs; defaults to contiguous indices.
            world_coordinates: root Rerun coordinate convention.
            image_plane_distance: metric frustum image-plane distance.
            trajectory_radius: trajectory line radius in map units.
            catalog_mode: enable the exoego:v2 native/rectified catalog layout.
        """
        if camera_names is None:
            if imagedirs is None:
                raise ValueError("RerunLogger needs imagedirs or camera_names")
            camera_names = [Path(imagedir).name for imagedir in imagedirs]
        self.cam_names = camera_names
        self.camera_ids = camera_ids if camera_ids is not None else list(range(len(camera_names)))
        if len(self.camera_ids) != len(self.cam_names):
            raise ValueError("camera_ids and camera_names must have the same length")
        self.scale = scale_factor
        self.splat_every = splat_every
        self.splat_cap = splat_cap
        self.max_splat_scale = max_splat_scale
        self.splat_scale_percentile = splat_scale_percentile
        self.max_depth = max_depth
        self.refine_every = refine_every
        self.world_coordinates = world_coordinates
        self.image_plane_distance = image_plane_distance
        self.trajectory_radius = trajectory_radius
        self.catalog_mode = catalog_mode
        self._kf_updates: int = 0
        self._calibrated: bool = False
        self._scene_bbox: tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]] | None = None
        self._traj_centers: Float[np.ndarray, "n_kf 3"] | None = None
        self._catalog_trajectory: list[Float[np.ndarray, "3"]] = []
        self._catalog_virtual_K_n4: Float[np.ndarray, "n_cams 4"] | None = None
        self._catalog_rig: Rig | None = None

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
        rr.log("/", self.world_coordinates, static=True)

    def _cam_path(self, cam_idx: int) -> str:
        if self.catalog_mode:
            return f"{CATALOG_RIG_PATH}/cam_{self.camera_ids[cam_idx]:02d}"
        return f"world/rig/cam_{self.camera_ids[cam_idx]:02d}_{self.cam_names[cam_idx]}"

    def _pinhole_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/pinhole"

    def _rectified_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/rectified"

    def _image_path(self, cam_idx: int) -> str:
        if self.catalog_mode:
            return f"{self._rectified_path(cam_idx)}/image"
        return f"{self._pinhole_path(cam_idx)}/image"

    def _depth_path(self, cam_idx: int) -> str:
        if self.catalog_mode:
            return f"{self._rectified_path(cam_idx)}/depth"
        return f"{self._pinhole_path(cam_idx)}/depth"

    def _video_path(self, cam_idx: int) -> str:
        if self.catalog_mode:
            return f"{self._pinhole_path(cam_idx)}/video"
        return f"{self._cam_path(cam_idx)}/video"

    def _compare_path(self, cam_idx: int) -> str:
        if self.catalog_mode:
            return self._rectified_path(cam_idx)
        return f"{COMPARE_ROOT}/cam_{cam_idx:02d}_{self.cam_names[cam_idx]}"

    def _blueprint(self, eye_controls: rrb.EyeControls3D | None = None) -> rrb.Blueprint:
        if self.catalog_mode:
            image_views = [
                rrb.Spatial2DView(
                    origin=self._video_path(i),
                    name=self.cam_names[i],
                )
                for i in range(len(self.cam_names))
            ]
        else:
            image_views = [
                rrb.Spatial2DView(
                    origin=self._pinhole_path(i),
                    name=self.cam_names[i],
                    contents=["+ $origin/image"],
                )
                for i in range(len(self.cam_names))
            ]
        depth_views = [
            rrb.Spatial2DView(origin=self._depth_path(i), name=f"{self.cam_names[i]} depth")
            for i in range(len(self.cam_names))
        ]
        # Per camera: Gaussian-map render on top, ground-truth image below.
        render_name: str = "render" if self.catalog_mode else "rendered"
        compare_views = [
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{self._compare_path(i)}/{render_name}", name=f"{self.cam_names[i]} render"),
                rrb.Spatial2DView(origin=f"{self._compare_path(i)}/gt", name=f"{self.cam_names[i]} GT"),
                name=self.cam_names[i],
            )
            for i in range(len(self.cam_names))
        ]
        # Keep estimated depth images out of the 3D views: their automatic
        # backprojection would double up with the Gaussian map. render_vs_gt
        # has no spatial context and doesn't belong in 3D either.
        if self.catalog_mode:
            contents_3d: list[str] = (
                ["+ /**"]
                + [f"- {self._video_path(i)}" for i in range(len(self.cam_names))]
                + [f"- {self._depth_path(i)}" for i in range(len(self.cam_names))]
                + [f"- {self._compare_path(i)}/render" for i in range(len(self.cam_names))]
                + [f"- {self._compare_path(i)}/gt" for i in range(len(self.cam_names))]
            )
        else:
            contents_3d = (
                ["+ /**", f"- {COMPARE_ROOT}/**"]
                + [f"- {self._depth_path(i)}" for i in range(len(self.cam_names))]
                + [f"- {self._video_path(i)}" for i in range(len(self.cam_names))]
            )
        # Follow view (rerun-io/eye_control_example pattern, as in the
        # examples-monorepo robocap blueprint): the view's origin IS the rig
        # entity, so a fixed chase eye expressed in RoboCap's rig frame
        # (X-left/Y-forward/Z-down) rides the rig as its world transform updates.
        follow_eye: rrb.EyeControls3D
        if self.catalog_mode:
            follow_eye = rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                # Indoor rig: stay close (0.6 m behind, 0.35 m above) so the chase
                # eye is not behind a wall of the room being mapped.
                position=(0.0, -0.6, -0.35),
                look_target=(0.0, 2.0, 0.1),
                eye_up=(0.0, 0.0, -1.0),
                spin_speed=0.0,
            )
        else:
            follow_eye = rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(0.0, -1.2, -2.5),
                look_target=(0.0, 0.0, 1.5),
                eye_up=(0.0, -1.0, 0.0),
                spin_speed=0.0,
            )
        follow_view = rrb.Spatial3DView(
            name="Follow",
            origin=CATALOG_RIG_PATH if self.catalog_mode else "world/rig",
            contents=contents_3d if self.catalog_mode else contents_3d + ["- world/keyframes/**"],
            eye_controls=follow_eye,
        )
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(origin="/", name="3D map", contents=contents_3d, eye_controls=eye_controls),
                    follow_view,
                    row_shares=[3.0, 2.0],
                ),
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

    def log_catalog_calibration(
        self,
        calibrations: Sequence["CameraCalibration"],
        virtual_K_n4: Float[Tensor, "n_cams 4"],
        image_hw: tuple[int, int] = (360, 640),
    ) -> None:
        """Log catalog rig extrinsics and virtual pinholes as static data.

        Args:
            calibrations: Chosen catalog camera calibrations in mapper order.
            virtual_K_n4: Float32 virtual ``[fx, fy, cx, cy]`` values with
                shape ``[n_cams, 4]``.
            image_hw: Virtual image size as ``(height, width)``.
        """
        height: int = image_hw[0]
        width: int = image_hw[1]
        self._catalog_virtual_K_n4 = virtual_K_n4.detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        camera_sensors: list[CameraSensor] = []
        for cam_idx, calibration in enumerate(calibrations):
            camera_from_rig_44: Float[np.ndarray, "4 4"] = calibration.camera_from_rig_44
            native_intrinsics: Intrinsics = Intrinsics.from_k_matrix(
                camera_conventions="RDF",
                k_matrix=calibration.intrinsic_33,
                height=calibration.resolution_wh[1],
                width=calibration.resolution_wh[0],
            )
            native_extrinsics: Extrinsics = Extrinsics(
                cam_R_world=camera_from_rig_44[:3, :3],
                cam_t_world=camera_from_rig_44[:3, 3],
            )
            native_camera: Fisheye62Parameters = Fisheye62Parameters(
                name=calibration.name,
                extrinsics=native_extrinsics,
                intrinsics=native_intrinsics,
                distortion=KannalaBrandtDistortion(
                    k1=float(calibration.distortion_4[0]),
                    k2=float(calibration.distortion_4[1]),
                    k3=float(calibration.distortion_4[2]),
                    k4=float(calibration.distortion_4[3]),
                ),
            )
            camera_sensors.append(
                CameraSensor(
                    index=calibration.camera_id,
                    name=calibration.name,
                    kind="rgb",
                    pinhole=native_camera,
                )
            )
            log_pinhole(
                native_camera,
                cam_log_path=Path(self._cam_path(cam_idx)),
                image_plane_distance=self.image_plane_distance,
                static=True,
                include_distortion=True,
            )
            rr.log(
                self._cam_path(cam_idx),
                rr.AnyValues(name=calibration.name, kind="rgb"),
                static=True,
            )
            fx: float = float(virtual_K_n4[cam_idx, 0])
            fy: float = float(virtual_K_n4[cam_idx, 1])
            cx: float = float(virtual_K_n4[cam_idx, 2])
            cy: float = float(virtual_K_n4[cam_idx, 3])
            rectified_camera: PinholeParameters = PinholeParameters(
                name=f"{calibration.name}_rectified",
                extrinsics=Extrinsics(
                    cam_R_world=np.eye(3, dtype=np.float64),
                    cam_t_world=np.zeros(3, dtype=np.float64),
                ),
                intrinsics=Intrinsics.from_focal_principal_point(
                    camera_conventions="RDF",
                    fl_x=fx,
                    fl_y=fy,
                    cx=cx,
                    cy=cy,
                    height=height,
                    width=width,
                ),
            )
            rr.log(
                self._rectified_path(cam_idx),
                PinholeWithDistortion.from_camera(
                    rectified_camera,
                    image_plane_distance=self.image_plane_distance,
                    include_distortion=False,
                ),
                static=True,
            )
            rr.log(
                self._rectified_path(cam_idx),
                rr.Transform3D(
                    translation=np.zeros(3, dtype=np.float64),
                    mat3x3=np.eye(3, dtype=np.float64),
                    relation=rr.TransformRelation.ChildFromParent,
                ),
                static=True,
            )
        self._catalog_rig = Rig(
            index=0,
            calibration=RigCalibration(cameras=camera_sensors, reference_index=0),
            image_plane_distance=self.image_plane_distance,
        )
        rr.log(
            CATALOG_RIG_PATH,
            rr.AnyValues(
                schema_version=SCHEMA_VERSION,
                reference="imu_00",
                num_cameras=len(camera_sensors),
                name="robocap",
                kind="ego",
            ),
            static=True,
        )
        self._calibrated = True

    def relay_video_streams(self, streams: Sequence["RawVideoStream"]) -> None:
        """Relay original H.264 packet windows to the recording without re-encoding.

        Args:
            streams: Per-camera packet windows, times, and keyframe flags.
        """
        for cam_idx, stream in enumerate(streams):
            rr.log(
                self._video_path(cam_idx),
                rr.VideoStream(codec=rr.VideoCodec.H264),
                static=True,
            )
            times_delta_n: np.ndarray = stream.times_ns.astype("timedelta64[ns]")
            rr.send_columns(
                self._video_path(cam_idx),
                indexes=[rr.TimeColumn("video_time", duration=times_delta_n)],
                columns=rr.VideoStream.columns(
                    sample=stream.samples,
                    is_keyframe=stream.is_keyframe,
                ),
            )

    def log_catalog_keyframe(
        self,
        frame_idx: int,
        frame: "CatalogKeyframe",
        depth_metres_nhw: Float[Tensor, "n_cams h w"],
        normals_n3hw: Float[Tensor, "n_cams 3 h w"],
    ) -> None:
        """Log one Basalt pose, metric trajectory, and per-camera MoGe depth.

        Args:
            frame_idx: Selected keyframe sequence index.
            frame: Joined catalog keyframe carrying ``T_world_from_rig``.
            depth_metres_nhw: Float32 MoGe metric depth with shape
                ``[n_cams, h, w]``.
            normals_n3hw: Float32 MoGe normals with shape ``[n_cams, 3, h, w]``.
        """
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("video_time", duration=np.timedelta64(frame.timestamp_ns, "ns"))
        world_from_rig_44: Float[np.ndarray, "4 4"] = frame.world_from_rig
        if self._catalog_rig is None:
            raise RuntimeError("log_catalog_calibration must run before keyframes")
        moving_rig: Rig = Rig(
            index=self._catalog_rig.index,
            calibration=self._catalog_rig.calibration,
            pose_stream=RigPoseStream(
                world_t_rig=world_from_rig_44[None, :3, 3],
                world_R_rig=world_from_rig_44[None, :3, :3],
            ),
            image_plane_distance=self._catalog_rig.image_plane_distance,
        )
        log_rig_pose_stream(
            moving_rig,
            timestamps_ns=np.asarray([frame.timestamp_ns], dtype=np.int64),
        )
        center_3: Float[np.ndarray, "3"] = world_from_rig_44[:3, 3].astype(
            np.float32, copy=True
        )
        self._catalog_trajectory.append(center_3)
        centers_n3: Float[np.ndarray, "n 3"] = np.stack(self._catalog_trajectory)
        self._traj_centers = centers_n3
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [centers_n3],
                colors=[[0, 200, 255]],
                radii=self.trajectory_radius,
            ),
        )

        if self._catalog_virtual_K_n4 is None:
            raise RuntimeError("log_catalog_calibration must run before keyframes")
        valid_nhw: Tensor = prior_valid(normals_n3hw.detach().cpu())
        depth_cpu_nhw: Tensor = depth_metres_nhw.detach().cpu()
        valid_nhw &= torch.isfinite(depth_cpu_nhw)
        valid_nhw &= (depth_cpu_nhw > 0.0) & (depth_cpu_nhw <= self.max_depth)
        for cam_idx in range(len(self.cam_names)):
            image_rgb_hwc: UInt8[np.ndarray, "h w 3"] = rearrange(
                frame.images_rgb[cam_idx].detach().cpu().numpy(),
                "c h w -> h w c",
            )
            rr.log(
                self._image_path(cam_idx),
                rr.Image(image_rgb_hwc).compress(jpeg_quality=75),
            )
            depth_hw: Tensor = torch.where(
                valid_nhw[cam_idx],
                depth_cpu_nhw[cam_idx],
                torch.zeros((), dtype=depth_cpu_nhw.dtype),
            )
            depth_mm_u16: np.ndarray = (
                depth_hw.numpy() * 1000.0
            ).round().clip(0, 65535).astype(np.uint16)
            rr.log(
                self._depth_path(cam_idx),
                rr.DepthImage(
                    depth_mm_u16,
                    meter=1000.0,
                    depth_range=[0.0, self.max_depth * 1000.0],
                ),
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
            image_hw3: UInt8[np.ndarray, "h w 3"] = rearrange(images_bgr[i].numpy(), "c h w -> h w c")
            rr.log(
                self._image_path(i),
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
            fx, fy, cx, cy = (float(v) for v in intrinsics[i, :4])
            rr.log(
                self._pinhole_path(i),
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
        rr.log("world/trajectory", rr.LineStrips3D([centers], colors=[[0, 200, 255]], radii=self.trajectory_radius))

        disps_up_list = video.disps_up_list
        for i in range(len(self.cam_names)):
            disp_hw: Float[Tensor, "h w"] = disps_up_list[i][idx].detach().cpu()
            prior_ok_hw = prior_valid(video.normals_list[i][idx][None].cpu())[0]
            valid_hw = (disp_hw > 0) & (self.scale < self.max_depth * disp_hw) & prior_ok_hw
            depth_hw: Float[Tensor, "h w"] = torch.where(valid_hw, self.scale / disp_hw.clamp(min=1e-6), torch.zeros(()))
            depth_mm_u16: np.ndarray = (depth_hw[::2, ::2].numpy() * 1000.0).clip(0, 65535).astype(np.uint16)
            rr.log(self._depth_path(i), rr.DepthImage(depth_mm_u16, meter=1000.0))

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

        # Drop the few giant sky/far splats: they are ~0.2 % of the map but hide
        # everything in a 3D view. The percentile cap is scale-free; the absolute
        # cap stays as a backstop for degenerate maps.
        largest_axis: np.ndarray = scales.max(axis=1)
        cap_scale: float = min(self.max_splat_scale, float(np.percentile(largest_axis, self.splat_scale_percentile)))
        keep: np.ndarray = np.flatnonzero(largest_axis < cap_scale)
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
        render_name: str = "render" if self.catalog_mode else "rendered"
        for name, img_3hw in ((render_name, rendered_3hw), ("gt", gt_3hw)):
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
        start: Float[np.ndarray, "3"] = self._traj_centers[0].astype(np.float64)
        forward: Float[np.ndarray, "3"] = center - start
        vertical_axis: int = 2 if self.catalog_mode else 1
        forward[vertical_axis] = 0.0
        norm: float = float(np.linalg.norm(forward))
        default_forward: np.ndarray = (
            np.array([0.0, 1.0, 0.0])
            if self.catalog_mode
            else np.array([0.0, 0.0, 1.0])
        )
        forward = forward / norm if norm > 1e-6 else default_forward
        elevation: np.ndarray = (
            np.array([0.0, 0.0, 0.22 * extent])
            if self.catalog_mode
            else np.array([0.0, -0.18 * extent, 0.0])
        )
        eye_up: list[float] = [0.0, 0.0, 1.0] if self.catalog_mode else [0.0, -1.0, 0.0]
        eye: Float[np.ndarray, "3"] = start - 0.45 * extent * forward + elevation
        eye_controls = rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=eye.tolist(),
            look_target=center.tolist(),
            eye_up=eye_up,
        )
        rr.send_blueprint(self._blueprint(eye_controls=eye_controls))

    def set_refine_iter(self, iteration: int) -> None:
        """Advance the color-refinement timeline (frame/time stay at their last values)."""
        rr.set_time("refine_iter", sequence=iteration)

    def log_text(self, text: str) -> None:
        """Log a status line to the slam_metrics text log."""
        rr.log("slam_metrics", rr.TextLog(text))

    def flush(self, timeout_seconds: float = 30.0) -> None:
        """Flush all configured sinks before the process exits."""
        rr.get_global_data_recording().flush(timeout_sec=timeout_seconds)
