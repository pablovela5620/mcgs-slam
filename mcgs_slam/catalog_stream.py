"""Catalog-backed RoboCap decoding and virtual-pinhole rectification."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import cv2
import numpy as np
import pyarrow as pa
import torch
from datafusion import col, lit
from einops import rearrange
from jaxtyping import Float32, Float64, Int64, UInt8
from rerun import catalog
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_dataloader import wrap_mp4
from simplecv.rig import CameraSensor, RigCalibration, entity_id
from simplecv.rrd_query_utils import first_valid_value
from torchcodec.decoders import VideoDecoder

from gaussian.utils.slam_utils import to_se3_vec


TIMELINE: str = "video_time"
RIG_PATH: str = f"/world/{entity_id('rig', 0)}"
CAMERA_IDS: tuple[int, ...] = (0, 1, 4, 5)
MAX_JOIN_NS: int = 1_000_000
KEYFRAME_PREROLL_NS: int = 1_100_000_000
DECODE_CHUNK_SIZE: int = 8

DecoderDevice: TypeAlias = Literal["cuda", "cpu"]



def _scalar_str(value: Any) -> str:
    """Return a catalog string component as a plain string.

    Rerun stores single-instance string components as one-element lists, and
    simplecv's ``first_valid_value`` only unwraps nested lists, so ``['left_front']``
    would otherwise become the literal text ``"['left_front']"``.
    """
    while isinstance(value, list | tuple) and len(value) == 1:
        value = value[0]
    return str(value)

def catalog_camera_path(camera_id: int) -> str:
    """Return the canonical exoego:v2 entity path for a camera."""
    return f"{RIG_PATH}/{entity_id('cam', camera_id)}"


def camera_video_path(camera: CameraSensor) -> str:
    """Return one camera's source H.264 VideoStream entity path."""
    return f"{catalog_camera_path(camera.index)}/pinhole/video"


@dataclass(slots=True)
class RawVideoStream:
    """Windowed H.264 packets and the decoder that owns their video session."""

    camera_id: int
    """Physical catalog camera index."""
    camera_name: str
    """Catalog camera name."""
    times_ns: Int64[np.ndarray, "n_samples"]
    """Per-packet ``video_time`` values in nanoseconds, including pre-roll."""
    samples: list[bytes]
    """Original Annex-B H.264 access units; no re-encoding."""
    is_keyframe: list[bool]
    """Source keyframe flag per packet."""
    decoder: VideoDecoder | None = field(repr=False)
    """Exact-seek TorchCodec decoder, released when the window closes."""


@dataclass(slots=True)
class CatalogKeyframe:
    """One motion-selected Basalt rig pose and its rectified camera images."""

    timestamp_ns: int
    """Reference-camera ``video_time`` in nanoseconds."""
    images_rgb: UInt8[torch.Tensor, "n_cams 3 360 640"]
    """Rectified uint8 RGB images on the CPU."""
    world_from_rig: Float64[np.ndarray, "4 4"]
    """Trusted Basalt ``T_world_from_rig`` matrix in metres."""


def _duration_literal(timestamp_ns: int) -> Any:
    """Build a DataFusion duration literal for the catalog timeline."""
    return lit(pa.scalar(timestamp_ns, pa.duration("ns")))


def _nearest_indices(
    reference_times_ns: Int64[np.ndarray, "n_reference"],
    query_times_ns: Int64[np.ndarray, "n_query"],
    max_delta_ns: int = MAX_JOIN_NS,
) -> Int64[np.ndarray, "n_query"]:
    """Return nearest reference indices, or ``-1`` beyond the tolerance."""
    if len(reference_times_ns) == 0:
        return np.full(len(query_times_ns), -1, dtype=np.int64)
    right_n: Int64[np.ndarray, "n_query"] = np.searchsorted(
        reference_times_ns, query_times_ns, side="left"
    ).astype(np.int64)
    left_n: Int64[np.ndarray, "n_query"] = np.clip(
        right_n - 1, 0, len(reference_times_ns) - 1
    )
    right_n = np.clip(right_n, 0, len(reference_times_ns) - 1)
    left_delta_n: Int64[np.ndarray, "n_query"] = np.abs(
        reference_times_ns[left_n] - query_times_ns
    )
    right_delta_n: Int64[np.ndarray, "n_query"] = np.abs(
        reference_times_ns[right_n] - query_times_ns
    )
    nearest_n: Int64[np.ndarray, "n_query"] = np.where(
        right_delta_n < left_delta_n, right_n, left_n
    ).astype(np.int64)
    nearest_delta_n: Int64[np.ndarray, "n_query"] = np.minimum(
        left_delta_n, right_delta_n
    )
    nearest_n[nearest_delta_n > max_delta_ns] = -1
    return nearest_n


def camera_from_world_pose(
    world_from_rig_44: Float64[np.ndarray, "4 4"],
    camera_from_rig_44: Float64[np.ndarray, "4 4"],
) -> Float32[np.ndarray, "7"]:
    """Compose a trusted rig pose and static extrinsic into mapper convention."""
    camera_from_world_44: Float64[np.ndarray, "4 4"] = (
        camera_from_rig_44 @ np.linalg.inv(world_from_rig_44)
    )
    return to_se3_vec(camera_from_world_44).astype(np.float32)


def camera_from_world_poses(
    world_from_rig_44: Float64[np.ndarray, "4 4"],
    rig_calibration: RigCalibration,
) -> Float32[torch.Tensor, "n_cams 7"]:
    """Adapt one trusted rig pose to all mapper camera poses."""
    poses_n7: Float32[np.ndarray, "n_cams 7"] = np.stack(
        [
            camera_from_world_pose(
                world_from_rig_44,
                camera.pinhole.extrinsics.cam_T_world,
            )
            for camera in rig_calibration.cameras
        ]
    ).astype(np.float32, copy=False)
    return torch.from_numpy(poses_n7)


@dataclass(frozen=True, slots=True)
class KeyframeSelector:
    """Select frames from trusted poses using translation or rotation motion."""

    distance_metres: float = 0.3
    """Minimum translation from the last keyframe, in metres."""
    angle_degrees: float = 15.0
    """Minimum rotation from the last keyframe, in degrees."""

    def select(
        self,
        world_from_rig_n44: Float64[np.ndarray, "n 4 4"],
    ) -> Int64[np.ndarray, "n_keyframes"]:
        """Return selected pose indices, always including the first frame."""
        pose_count: int = len(world_from_rig_n44)
        if pose_count == 0:
            return np.empty(0, dtype=np.int64)

        selected: list[int] = [0]
        last_index: int = 0
        for index in range(1, pose_count):
            translation_metres: float = float(
                np.linalg.norm(
                    world_from_rig_n44[index, :3, 3]
                    - world_from_rig_n44[last_index, :3, 3]
                )
            )
            relative_rotation_33: Float64[np.ndarray, "3 3"] = (
                world_from_rig_n44[last_index, :3, :3].T
                @ world_from_rig_n44[index, :3, :3]
            )
            angle_degrees: float = float(
                np.rad2deg(Rotation.from_matrix(relative_rotation_33).magnitude())
            )
            if (
                translation_metres >= self.distance_metres
                or angle_degrees >= self.angle_degrees
            ):
                selected.append(index)
                last_index = index
        return np.asarray(selected, dtype=np.int64)


@dataclass(slots=True)
class FisheyeRectifier:
    """Rectify one SimpleCV fisheye camera into a virtual pinhole."""

    camera: CameraSensor
    """Native catalog camera and its KB4 model."""
    output_size: tuple[int, int] = (640, 360)
    """Virtual image size as ``(width, height)``."""
    horizontal_fov_degrees: float = 110.0
    """Horizontal field of view of the virtual pinhole."""
    virtual_camera: PinholeParameters = field(init=False)
    """Virtual pinhole with an identity transform below the native camera."""
    map1: np.ndarray = field(init=False, repr=False)
    """First fixed-point remap table returned by OpenCV."""
    map2: np.ndarray = field(init=False, repr=False)
    """Second fixed-point remap table returned by OpenCV."""

    def __post_init__(self) -> None:
        """Build the virtual camera and immutable remap tables."""
        native_camera = self.camera.pinhole
        if not isinstance(native_camera, Fisheye62Parameters):
            raise TypeError("FisheyeRectifier requires Fisheye62Parameters")
        if native_camera.distortion is None:
            raise ValueError("fisheye distortion coefficients are required")

        width: int = self.output_size[0]
        height: int = self.output_size[1]
        half_fov_radians: float = np.deg2rad(self.horizontal_fov_degrees / 2.0)
        focal: float = (width / 2.0) / np.tan(half_fov_radians)
        virtual_k_33: Float64[np.ndarray, "3 3"] = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.virtual_camera = PinholeParameters(
            name=f"{self.camera.name}_rectified",
            extrinsics=Extrinsics(
                cam_R_world=np.eye(3, dtype=np.float64),
                cam_t_world=np.zeros(3, dtype=np.float64),
            ),
            intrinsics=Intrinsics.from_k_matrix(
                camera_conventions="RDF",
                k_matrix=virtual_k_33,
                height=height,
                width=width,
            ),
        )
        distortion_4: Float64[np.ndarray, "4"] = np.asarray(
            [
                native_camera.distortion.k1,
                native_camera.distortion.k2,
                native_camera.distortion.k3,
                native_camera.distortion.k4,
            ],
            dtype=np.float64,
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            np.asarray(native_camera.intrinsics.k_matrix, dtype=np.float64),
            distortion_4,
            np.eye(3, dtype=np.float64),
            virtual_k_33,
            self.output_size,
            cv2.CV_16SC2,
        )

    def rectify_points(
        self,
        distorted_xy: Float64[np.ndarray, "n 2"],
    ) -> Float64[np.ndarray, "n 2"]:
        """Map native KB4 pixels to the virtual pinhole."""
        native_camera = self.camera.pinhole
        assert isinstance(native_camera, Fisheye62Parameters)
        assert native_camera.distortion is not None
        distortion_4: Float64[np.ndarray, "4"] = np.asarray(
            [
                native_camera.distortion.k1,
                native_camera.distortion.k2,
                native_camera.distortion.k3,
                native_camera.distortion.k4,
            ],
            dtype=np.float64,
        )
        points_n12: Float64[np.ndarray, "n 1 2"] = np.asarray(
            distorted_xy, dtype=np.float64
        )[:, None, :]
        rectified_n12: Float64[np.ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(
            points_n12,
            np.asarray(native_camera.intrinsics.k_matrix, dtype=np.float64),
            distortion_4,
            R=np.eye(3, dtype=np.float64),
            P=np.asarray(self.virtual_camera.intrinsics.k_matrix),
        )
        return rectified_n12[:, 0, :]

    def rectify(
        self,
        image_rgb_hwc: UInt8[np.ndarray, "native_h native_w 3"],
    ) -> UInt8[np.ndarray, "output_h output_w 3"]:
        """Rectify one native RGB image with the precomputed maps."""
        return cv2.remap(
            image_rgb_hwc,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


class CatalogWindow:
    """Own a bounded catalog query, its decoders, and rectified keyframes."""

    def __init__(
        self,
        segment: "RobocapSegment",
        *,
        start_seconds: float,
        end_seconds: float,
        kf_dist: float,
        kf_angle: float,
    ) -> None:
        if start_seconds < 0.0 or end_seconds <= start_seconds:
            raise ValueError("expected 0 <= start_seconds < end_seconds")
        self.segment: RobocapSegment = segment
        self.rectifiers: tuple[FisheyeRectifier, ...] = segment.rectifiers
        start_ns: int = segment.segment_video_start_ns + round(start_seconds * 1e9)
        end_ns: int = segment.segment_video_start_ns + round(end_seconds * 1e9)
        self.relay_streams: tuple[RawVideoStream, ...] = tuple(
            segment._read_video_window(camera, start_ns, end_ns)
            for camera in segment.rig_calibration.cameras
        )

        pose_times_ns, world_from_rig_n44 = segment._read_poses(start_ns, end_ns)
        reference_times_ns: Int64[np.ndarray, "n_reference"] = self.relay_streams[0].times_ns
        # The first cam_00 packet is blown out. Drop it at the window boundary
        # before pose joining or motion selection so no downstream consumer sees it.
        requested_mask_n: np.ndarray = (
            (reference_times_ns >= start_ns)
            & (reference_times_ns < end_ns)
            & (reference_times_ns != segment.cam00_frame0_ns)
        )
        target_times_ns: Int64[np.ndarray, "n_targets"] = reference_times_ns[requested_mask_n]
        stream_indices: list[Int64[np.ndarray, "n_targets"]] = [
            _nearest_indices(stream.times_ns, target_times_ns)
            for stream in self.relay_streams
        ]
        pose_indices_n: Int64[np.ndarray, "n_targets"] = _nearest_indices(
            pose_times_ns, target_times_ns
        )
        matched_mask_n: np.ndarray = pose_indices_n >= 0
        for indices_n in stream_indices:
            matched_mask_n &= indices_n >= 0
        target_times_ns = target_times_ns[matched_mask_n]
        pose_indices_n = pose_indices_n[matched_mask_n]
        stream_indices = [indices_n[matched_mask_n] for indices_n in stream_indices]
        if len(target_times_ns) == 0:
            raise RuntimeError("no four-camera frames matched a Basalt pose within 1 ms")

        aligned_world_from_rig_n44: Float64[np.ndarray, "n_targets 4 4"] = world_from_rig_n44[
            pose_indices_n
        ]
        selected_n: Int64[np.ndarray, "n_keyframes"] = KeyframeSelector(
            distance_metres=kf_dist,
            angle_degrees=kf_angle,
        ).select(aligned_world_from_rig_n44)
        self._times_ns: Int64[np.ndarray, "n_keyframes"] = target_times_ns[selected_n]
        self._world_from_rig_n44: Float64[np.ndarray, "n_keyframes 4 4"] = (
            aligned_world_from_rig_n44[selected_n]
        )
        self._stream_indices: tuple[Int64[np.ndarray, "n_keyframes"], ...] = tuple(
            indices_n[selected_n] for indices_n in stream_indices
        )
        self._closed: bool = False

    def __enter__(self) -> "CatalogWindow":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release every TorchCodec decoder before mapper refinement."""
        if self._closed:
            return
        for stream in self.relay_streams:
            stream.decoder = None
        self._closed = True

    def keyframes(self) -> Iterator[CatalogKeyframe]:
        """Decode and rectify bounded four-camera chunks, then yield rig frames."""
        if self._closed:
            raise RuntimeError("catalog window is closed")

        keyframe_count: int = len(self._times_ns)
        for chunk_start in range(0, keyframe_count, DECODE_CHUNK_SIZE):
            chunk_end: int = min(chunk_start + DECODE_CHUNK_SIZE, keyframe_count)
            rectified_by_camera: list[
                UInt8[torch.Tensor, "chunk n_channels=3 rectified_h rectified_w"]
            ] = []
            for camera_index, stream in enumerate(self.relay_streams):
                decoder = stream.decoder
                if decoder is None:
                    raise RuntimeError(
                        "catalog decoder was released before keyframe decoding"
                    )
                decoder_indices: list[int] = [
                    int(index)
                    for index in self._stream_indices[camera_index][chunk_start:chunk_end]
                ]
                decoded_n3hw: UInt8[
                    torch.Tensor, "chunk n_channels=3 native_h native_w"
                ] = decoder.get_frames_at(decoder_indices).data.detach().to(device="cpu")
                rectified_frames: list[np.ndarray] = []
                for decoded_rgb_chw in decoded_n3hw:
                    decoded_rgb_hwc: UInt8[
                        np.ndarray, "native_h native_w n_channels=3"
                    ] = rearrange(decoded_rgb_chw.numpy(), "c h w -> h w c")
                    rectified_rgb_hwc: UInt8[
                        np.ndarray, "rectified_h rectified_w n_channels=3"
                    ] = self.rectifiers[camera_index].rectify(decoded_rgb_hwc)
                    rectified_frames.append(
                        rearrange(rectified_rgb_hwc, "h w c -> c h w").copy()
                    )
                rectified_n3hw: UInt8[
                    np.ndarray, "chunk n_channels=3 rectified_h rectified_w"
                ] = np.stack(rectified_frames)
                rectified_by_camera.append(torch.from_numpy(rectified_n3hw))

            for chunk_index, keyframe_index in enumerate(
                range(chunk_start, chunk_end)
            ):
                images_rgb_n3hw: UInt8[
                    torch.Tensor,
                    "n_cams n_channels=3 rectified_h rectified_w",
                ] = torch.stack(
                    [frames[chunk_index] for frames in rectified_by_camera]
                )
                yield CatalogKeyframe(
                    timestamp_ns=int(self._times_ns[keyframe_index]),
                    images_rgb=images_rgb_n3hw,
                    world_from_rig=self._world_from_rig_n44[keyframe_index].copy(),
                )


class RobocapSegment:
    """Read one RoboCap catalog segment as trusted-pose mapping windows."""

    def __init__(
        self,
        *,
        catalog_url: str,
        dataset_name: str = "robocap",
        dataset_id: str | None = None,
        segment_id: str,
        camera_ids: tuple[int, ...] = CAMERA_IDS,
        decoder: DecoderDevice = "cuda",
        fps: int = 30,
    ) -> None:
        self.catalog_url: str = catalog_url
        self.dataset_name: str = dataset_name
        self.dataset_id: str | None = dataset_id
        self.segment_id: str = segment_id
        self.camera_ids: tuple[int, ...] = camera_ids
        self.decoder: DecoderDevice = decoder
        self.fps: int = fps
        self.client: catalog.CatalogClient = catalog.CatalogClient(catalog_url)
        self.dataset: Any = (
            self.client.get_dataset(id=dataset_id)
            if dataset_id is not None
            else self.client.get_dataset(name=dataset_name)
        )
        self.rig_calibration: RigCalibration = self._read_calibrations()

        self.rectifiers: tuple[FisheyeRectifier, ...] = tuple(
            FisheyeRectifier(camera) for camera in self.rig_calibration.cameras
        )
        self.rectified_cameras: tuple[PinholeParameters, ...] = tuple(
            rectifier.virtual_camera for rectifier in self.rectifiers
        )
        virtual_intrinsics_n4: Float32[np.ndarray, "n_cams 4"] = np.asarray(
            [
                [
                    camera.intrinsics.fl_x,
                    camera.intrinsics.fl_y,
                    camera.intrinsics.cx,
                    camera.intrinsics.cy,
                ]
                for camera in self.rectified_cameras
            ],
            dtype=np.float32,
        )
        self.virtual_intrinsics: Float32[torch.Tensor, "n_cams 4"] = torch.from_numpy(
            virtual_intrinsics_n4
        )
        video_paths: list[str] = [
            camera_video_path(camera) for camera in self.rig_calibration.cameras
        ]
        self.segment_video_start_ns: int = self._first_packet_ns(video_paths)
        self.cam00_frame0_ns: int = self._first_packet_ns([video_paths[0]])

    @property
    def camera_names(self) -> list[str]:
        """Return chosen catalog camera names in mapper order."""
        return [camera.name for camera in self.rig_calibration.cameras]

    def _view(self, entities: list[str]) -> Any:
        """Return this segment filtered to the requested entities."""
        return self.dataset.filter_contents(entities).filter_segments([self.segment_id])

    def _read_calibrations(self) -> RigCalibration:
        """Read native cameras directly into SimpleCV rig types."""
        entities: list[str] = []
        for camera_id in self.camera_ids:
            camera_path: str = catalog_camera_path(camera_id)
            entities.extend([camera_path, f"{camera_path}/pinhole"])
        table: pa.Table = self._view(entities).reader(index=None).to_arrow_table()
        if table.num_rows == 0:
            raise RuntimeError(f"segment {self.segment_id} has no static camera calibration")

        cameras: list[CameraSensor] = []
        for camera_id in self.camera_ids:
            camera_path = catalog_camera_path(camera_id)
            pinhole_path: str = f"{camera_path}/pinhole"
            name_component: str = f"{camera_path}:name"
            name: str = _scalar_str(
                first_valid_value(table.column(name_component), component_name=name_component)
            )
            matrix_component: str = f"{camera_path}:Transform3D:mat3x3"
            matrix_raw: Any = first_valid_value(
                table.column(matrix_component), component_name=matrix_component
            )
            rotation_camera_from_rig_33: Float64[np.ndarray, "3 3"] = np.asarray(
                matrix_raw, dtype=np.float64
            ).reshape(3, 3, order="F")
            translation_component: str = f"{camera_path}:Transform3D:translation"
            translation_camera_from_rig_3: Float64[np.ndarray, "3"] = np.asarray(
                first_valid_value(
                    table.column(translation_component), component_name=translation_component
                ),
                dtype=np.float64,
            )
            intrinsic_component: str = f"{pinhole_path}:Pinhole:image_from_camera"
            intrinsic_33: Float64[np.ndarray, "3 3"] = np.asarray(
                first_valid_value(
                    table.column(intrinsic_component), component_name=intrinsic_component
                ),
                dtype=np.float64,
            ).reshape(3, 3, order="F")
            distortion_component: str = f"{pinhole_path}:simplecv.components.DistortionCoefficients"
            distortion_values: Float64[np.ndarray, "coefficients"] = np.asarray(
                first_valid_value(
                    table.column(distortion_component), component_name=distortion_component
                ),
                dtype=np.float64,
            )
            padded_distortion_8: Float64[np.ndarray, "8"] = np.pad(
                distortion_values[:8], (0, max(0, 8 - len(distortion_values[:8])))
            )
            resolution_component: str = f"{pinhole_path}:Pinhole:resolution"
            resolution_wh: list[int] = [
                int(value)
                for value in first_valid_value(
                    table.column(resolution_component), component_name=resolution_component
                )
            ]
            kind_component: str = f"{camera_path}:kind"
            kind_value: str = (
                _scalar_str(
                    first_valid_value(
                        table.column(kind_component),
                        allow_none=True,
                        component_name=kind_component,
                    )
                )
                if kind_component in table.column_names
                else "rgb"
            )
            extrinsics = Extrinsics(
                cam_R_world=rotation_camera_from_rig_33,
                cam_t_world=translation_camera_from_rig_3,
            )
            intrinsics = Intrinsics.from_k_matrix(
                camera_conventions="RDF",
                k_matrix=intrinsic_33,
                height=resolution_wh[1],
                width=resolution_wh[0],
            )
            distortion = KannalaBrandtDistortion(
                k1=float(padded_distortion_8[0]),
                k2=float(padded_distortion_8[1]),
                k3=float(padded_distortion_8[2]),
                k4=float(padded_distortion_8[3]),
                k5=float(padded_distortion_8[4]),
                k6=float(padded_distortion_8[5]),
                p1=float(padded_distortion_8[6]),
                p2=float(padded_distortion_8[7]),
            )
            fisheye = Fisheye62Parameters(
                name=name,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                distortion=distortion,
            )
            cameras.append(
                CameraSensor(
                    index=camera_id,
                    name=name,
                    kind="grayscale" if kind_value == "grayscale" else "rgb",
                    pinhole=fisheye,
                )
            )
        return RigCalibration(cameras=cameras, reference_index=self.camera_ids[0])

    def _first_packet_ns(self, video_paths: list[str]) -> int:
        """Read the earliest packet timestamp among the requested video entities."""
        table: pa.Table = (
            self._view(video_paths)
            .reader(index=TIMELINE)
            .sort(col(TIMELINE))
            .limit(1)
            .to_arrow_table()
        )
        if table.num_rows == 0:
            raise RuntimeError(f"segment {self.segment_id} contains no video samples")
        times_ns: Int64[np.ndarray, "n"] = np.asarray(
            table.column(TIMELINE).cast(pa.int64()).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        return int(times_ns[0])

    def _read_video_window(
        self,
        camera: CameraSensor,
        start_ns: int,
        end_ns: int,
    ) -> RawVideoStream:
        """Read one packet window, rounded back to its previous H.264 keyframe."""
        query_start_ns: int = max(0, start_ns - KEYFRAME_PREROLL_NS)
        video_path: str = camera_video_path(camera)
        table: pa.Table = (
            self._view([video_path])
            .reader(index=TIMELINE)
            .filter(
                (col(TIMELINE) >= _duration_literal(query_start_ns))
                & (col(TIMELINE) < _duration_literal(end_ns))
            )
            .sort(col(TIMELINE))
            .to_arrow_table()
        )
        times_ns: Int64[np.ndarray, "n_samples"] = np.asarray(
            table.column(TIMELINE).cast(pa.int64()).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        sample_instances = table.column(f"{video_path}:VideoStream:sample").combine_chunks()
        blobs = sample_instances.flatten()
        byte_values = blobs.flatten()
        byte_buffer = byte_values.buffers()[1]
        if byte_buffer is None:
            raise RuntimeError(f"{camera.name} has no packet bytes in the requested window")
        data = memoryview(byte_buffer)
        offsets: list[int] = blobs.offsets.to_pylist()
        samples: list[bytes] = [
            bytes(data[start:end])
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ]
        is_keyframe: list[bool] = [
            bool(flag)
            for flag in table.column(f"{video_path}:VideoStream:is_keyframe")
            .combine_chunks()
            .flatten()
            .to_pylist()
        ]
        if not samples:
            raise RuntimeError(f"{camera.name} has no packets in [{start_ns}, {end_ns})")

        pre_roll_keyframes: list[int] = [
            index
            for index, (timestamp_ns, keyframe) in enumerate(
                zip(times_ns, is_keyframe, strict=True)
            )
            if keyframe and timestamp_ns <= start_ns
        ]
        first_index: int = pre_roll_keyframes[-1] if pre_roll_keyframes else 0
        times_ns = times_ns[first_index:]
        samples = samples[first_index:]
        is_keyframe = is_keyframe[first_index:]
        if not is_keyframe[0]:
            raise RuntimeError(f"{camera.name} window does not start on an H.264 keyframe")
        decoder = VideoDecoder(
            wrap_mp4(samples, is_keyframe, self.fps, codec="h264"),
            device=self.decoder,
            seek_mode="exact",
            num_ffmpeg_threads=0,
        )
        return RawVideoStream(
            camera_id=camera.index,
            camera_name=camera.name,
            times_ns=times_ns,
            samples=samples,
            is_keyframe=is_keyframe,
            decoder=decoder,
        )

    def _read_poses(
        self,
        start_ns: int,
        end_ns: int,
    ) -> tuple[Int64[np.ndarray, "n_poses"], Float64[np.ndarray, "n_poses 4 4"]]:
        """Read trusted Basalt ``T_world_from_rig`` poses around the window."""
        table: pa.Table = (
            self._view([RIG_PATH])
            .reader(index=TIMELINE)
            .filter(
                (col(TIMELINE) >= _duration_literal(start_ns - MAX_JOIN_NS))
                & (col(TIMELINE) < _duration_literal(end_ns + MAX_JOIN_NS))
            )
            .sort(col(TIMELINE))
            .to_arrow_table()
        )
        times_ns: Int64[np.ndarray, "n_rows"] = np.asarray(
            table.column(TIMELINE).cast(pa.int64()).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        translation_instances: list[Any] = table.column(
            f"{RIG_PATH}:Transform3D:translation"
        ).to_pylist()
        quaternion_instances: list[Any] = table.column(
            f"{RIG_PATH}:Transform3D:quaternion"
        ).to_pylist()
        pose_times: list[int] = []
        poses: list[np.ndarray] = []
        for timestamp_ns, translation_row, quaternion_row in zip(
            times_ns, translation_instances, quaternion_instances, strict=True
        ):
            if translation_row is None or quaternion_row is None:
                continue
            translation_3: Float64[np.ndarray, "3"] = np.asarray(
                translation_row[0], dtype=np.float64
            )
            quaternion_xyzw: Float64[np.ndarray, "4"] = np.asarray(
                quaternion_row[0], dtype=np.float64
            )
            world_from_rig_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
            world_from_rig_44[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
            world_from_rig_44[:3, 3] = translation_3
            pose_times.append(int(timestamp_ns))
            poses.append(world_from_rig_44)
        if not poses:
            raise RuntimeError(f"segment {self.segment_id} has no Basalt poses in the window")
        return np.asarray(pose_times, dtype=np.int64), np.stack(poses)

    def open_window(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
        kf_dist: float = 0.3,
        kf_angle: float = 15.0,
    ) -> CatalogWindow:
        """Open a bounded decoder window that must close before refinement."""
        return CatalogWindow(
            self,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            kf_dist=kf_dist,
            kf_angle=kf_angle,
        )
