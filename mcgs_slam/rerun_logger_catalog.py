"""Rerun logging for trusted-pose RoboCap catalog mapping."""

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import log_pinhole
from simplecv.rerun_rig_logger import SCHEMA_VERSION
from simplecv.rig import RigCalibration, entity_id
from torch import Tensor

from prior_mask import prior_valid
from rerun_logger import RerunLogger

if TYPE_CHECKING:
    from catalog_stream import CatalogKeyframe, RawVideoStream


CATALOG_RIG_PATH: str = f"world/{entity_id('rig', 0)}"


class CatalogRerunLogger(RerunLogger):
    """Log the metric RFU exoego:v2 rig, video, depth, and Gaussian map."""

    def __init__(
        self,
        *,
        rig_calibration: RigCalibration,
        rectified_cameras: tuple[PinholeParameters, ...],
        map_scale: float,
        save_path: str | None = None,
        spawn: bool = False,
        splat_every: int = 10,
        splat_cap: int = 75_000,
        max_splat_scale: float = 0.3,
        splat_scale_percentile: float = 99.7,
        max_depth: float = 15.0,
        refine_every: int = 10_000,
        image_plane_distance: float = 0.25,
        trajectory_radius: float = 0.04,
    ) -> None:
        if map_scale != 1.0:
            raise ValueError("CatalogRerunLogger requires map_scale=1.0")
        if len(rectified_cameras) != len(rig_calibration.cameras):
            raise ValueError("rectified_cameras must match rig_calibration.cameras")
        self.rig_calibration: RigCalibration = rig_calibration
        self.rectified_cameras: tuple[PinholeParameters, ...] = rectified_cameras
        self._previous_center: Float[np.ndarray, "3"] | None = None
        super().__init__(
            camera_names=[camera.name for camera in rig_calibration.cameras],
            camera_ids=[camera.index for camera in rig_calibration.cameras],
            map_scale=map_scale,
            world_coordinates=rr.ViewCoordinates.RFU,
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
        self.log_catalog_calibration()

    def _cam_path(self, cam_idx: int) -> str:
        return f"{CATALOG_RIG_PATH}/{entity_id('cam', self.camera_ids[cam_idx])}"

    def _pinhole_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/pinhole"

    def _video_path(self, cam_idx: int) -> str:
        return f"{self._pinhole_path(cam_idx)}/video"

    def _rectified_path(self, cam_idx: int) -> str:
        return f"{self._cam_path(cam_idx)}/rectified"

    def _image_path(self, cam_idx: int) -> str:
        return f"{self._rectified_path(cam_idx)}/image"

    def _depth_path(self, cam_idx: int) -> str:
        return f"{self._rectified_path(cam_idx)}/depth"

    def _render_path(self, cam_idx: int) -> str:
        return f"{self._rectified_path(cam_idx)}/render"

    def _ground_truth_path(self, cam_idx: int) -> None:
        del cam_idx
        return None

    def _blueprint(
        self,
        eye_controls: rrb.EyeControls3D | None = None,
    ) -> rrb.Blueprint:
        image_views: list[rrb.Spatial2DView] = [
            rrb.Spatial2DView(
                origin=self._video_path(index),
                name=self.cam_names[index],
            )
            for index in range(len(self.cam_names))
        ]
        depth_views: list[rrb.Spatial2DView] = [
            rrb.Spatial2DView(
                origin=self._depth_path(index),
                name=f"{self.cam_names[index]} depth",
            )
            for index in range(len(self.cam_names))
        ]
        compare_views: list[object] = [
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=self._render_path(index),
                    name=f"{self.cam_names[index]} render",
                ),
                rrb.Spatial2DView(
                    origin=self._image_path(index),
                    name=f"{self.cam_names[index]} GT",
                ),
                name=self.cam_names[index],
            )
            for index in range(len(self.cam_names))
        ]
        contents_3d: list[str] = (
            ["+ /**"]
            + [f"- {self._pinhole_path(index)}" for index in range(len(self.cam_names))]
            + [f"- {self._video_path(index)}" for index in range(len(self.cam_names))]
            + [f"- {self._image_path(index)}" for index in range(len(self.cam_names))]
            + [f"- {self._depth_path(index)}" for index in range(len(self.cam_names))]
            + [f"- {self._render_path(index)}" for index in range(len(self.cam_names))]
        )
        follow_eye = rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.0, -0.6, -0.35),
            look_target=(0.0, 2.0, 0.1),
            eye_up=(0.0, 0.0, -1.0),
            spin_speed=0.0,
        )
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
                        origin=CATALOG_RIG_PATH,
                        contents=contents_3d,
                        eye_controls=follow_eye,
                    ),
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

    def log_catalog_calibration(self) -> None:
        """Log the native SimpleCV rig cameras and rectified child pinholes."""
        # simplecv.log_rig_static requires its reference to be a camera. RoboCap's
        # source rig is IMU-referenced, so preserve the source metadata by hand.
        rr.log(
            CATALOG_RIG_PATH,
            rr.AnyValues(
                schema_version=SCHEMA_VERSION,
                reference="imu_00",
                num_cameras=len(self.rig_calibration.cameras),
                name="robocap",
                kind="ego",
            ),
            static=True,
        )
        for cam_idx, (camera, rectified_camera) in enumerate(
            zip(
                self.rig_calibration.cameras,
                self.rectified_cameras,
                strict=True,
            )
        ):
            log_pinhole(
                camera.pinhole,
                cam_log_path=Path(self._cam_path(cam_idx)),
                image_plane_distance=self.image_plane_distance,
                static=True,
                include_distortion=True,
            )
            rr.log(
                self._cam_path(cam_idx),
                rr.AnyValues(name=camera.name, kind=camera.kind),
                static=True,
            )
            log_pinhole(
                rectified_camera,
                cam_log_path=Path(self._rectified_path(cam_idx)),
                image_plane_distance=self.image_plane_distance,
                static=True,
                include_distortion=False,
            )

    def relay_video_streams(self, streams: Sequence["RawVideoStream"]) -> None:
        """Relay original H.264 packet windows without re-encoding."""
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
        """Log one trusted rig pose, an incremental path segment, images, and depth."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time(
            "video_time",
            duration=np.timedelta64(frame.timestamp_ns, "ns"),
        )
        world_from_rig_44: Float[np.ndarray, "4 4"] = frame.world_from_rig
        rr.log(
            CATALOG_RIG_PATH,
            rr.Transform3D(
                translation=world_from_rig_44[:3, 3],
                mat3x3=world_from_rig_44[:3, :3],
            ),
        )
        center_3: Float[np.ndarray, "3"] = world_from_rig_44[:3, 3].astype(
            np.float32, copy=True
        )
        segment_23: Float[np.ndarray, "2 3"] = np.stack(
            [
                self._previous_center
                if self._previous_center is not None
                else center_3,
                center_3,
            ]
        )
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [segment_23],
                colors=[[0, 200, 255]],
                radii=self.trajectory_radius,
            ),
        )
        self._previous_center = center_3
        self._record_trajectory(center_3[None])

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
