"""Catalog-backed robocap decoding and virtual-pinhole rectification."""

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, TypeAlias

import cv2
import numpy as np
import pyarrow as pa
import torch
from datafusion import col, lit
from einops import rearrange
from jaxtyping import Float32, Float64, Int64, UInt8
from rerun import catalog
from scipy.spatial.transform import Rotation
from simplecv.rerun_dataloader import wrap_mp4
from torchcodec.decoders import VideoDecoder

CATALOG_URL: str = "rerun+http://pablo-dl-server.ilish-ruler.ts.net:51235"
DATASET_ID: str = "18CFB19109CFDB071d88fb8b48ef65e9"
SEGMENT_ID: str = "robocap__f408193e6447b3b0__s00000021"
TIMELINE: str = "video_time"
RIG_PATH: str = "/world/rig_00"
CAMERA_IDS: tuple[int, ...] = (0, 1, 4, 5)
MAX_JOIN_NS: int = 1_000_000
KEYFRAME_PREROLL_NS: int = 1_100_000_000

DecoderDevice: TypeAlias = Literal["cuda", "cpu"]


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """One chosen catalog camera's native KB4 calibration and rig extrinsic."""

    camera_id: int
    """Physical catalog index, such as 0, 1, 4, or 5."""
    name: str
    """Catalog camera name, such as ``left_front``."""
    intrinsic_33: Float64[np.ndarray, "3 3"]
    """Native fisheye calibration matrix in pixels."""
    distortion_4: Float64[np.ndarray, "4"]
    """Kannala-Brandt coefficients ``k1..k4``."""
    camera_from_rig_44: Float64[np.ndarray, "4 4"]
    """Static catalog ``T_camera_from_rig`` matrix in metres."""
    resolution_wh: tuple[int, int]
    """Native encoded resolution as ``(width, height)``."""

    @property
    def entity_path(self) -> str:
        """Return the camera entity path in the source catalog."""
        return f"{RIG_PATH}/cam_0{self.camera_id}"

    @property
    def video_path(self) -> str:
        """Return the source H.264 VideoStream entity path."""
        return f"{self.entity_path}/pinhole/video"


@dataclass(slots=True)
class RawVideoStream:
    """Windowed H.264 packets plus their exact decoder and catalog times."""

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
    decoder: VideoDecoder = field(repr=False)
    """Exact-seek TorchCodec decoder built from the packet window."""


@dataclass(slots=True)
class CatalogKeyframe:
    """One motion-selected Basalt pose with all four rectified camera views."""

    timestamp_ns: int
    """cam_00 ``video_time`` in nanoseconds for this joined rig frame."""
    images_rgb: UInt8[torch.Tensor, "n_cams 3 360 640"]
    """Rectified uint8 RGB images on the CPU."""
    virtual_K: Float32[torch.Tensor, "n_cams 4"]
    """Per-camera virtual ``[fx, fy, cx, cy]`` calibration in pixels."""
    camera_from_world: Float32[torch.Tensor, "n_cams 7"]
    """Lietorch ``T_camera_from_world`` vectors in ``xyzw`` quaternion order."""
    world_from_rig: Float64[np.ndarray, "4 4"]
    """Trusted Basalt ``T_world_from_rig`` matrix in metres."""
    relay_streams: tuple[RawVideoStream, ...]
    """Raw packet windows and keyframe flags for one-time Rerun relay."""


def _duration_literal(timestamp_ns: int) -> Any:
    """Build a DataFusion duration literal for the catalog timeline."""
    return lit(pa.scalar(timestamp_ns, pa.duration("ns")))


def _unwrap_component(value: Any) -> Any:
    """Remove Rerun's one-instance list wrappers from one Arrow row value."""
    unwrapped: Any = value
    while isinstance(unwrapped, list) and len(unwrapped) == 1:
        unwrapped = unwrapped[0]
    return unwrapped


def _nearest_indices(
    reference_times_ns: Int64[np.ndarray, "n_reference"],
    query_times_ns: Int64[np.ndarray, "n_query"],
    max_delta_ns: int = MAX_JOIN_NS,
) -> Int64[np.ndarray, "n_query"]:
    """Return nearest reference indices, or ``-1`` beyond the tolerance.

    Args:
        reference_times_ns: Sorted Int64 reference times with shape
            ``[n_reference]``.
        query_times_ns: Int64 query times with shape ``[n_query]``.
        max_delta_ns: Maximum accepted absolute timestamp error.

    Returns:
        Int64 reference indices with shape ``[n_query]``; unmatched values are
        ``-1``.
    """
    if len(reference_times_ns) == 0:
        return np.full(len(query_times_ns), -1, dtype=np.int64)
    right_n: Int64[np.ndarray, "n_query"] = np.searchsorted(
        reference_times_ns, query_times_ns, side="left"
    ).astype(np.int64)
    left_n: Int64[np.ndarray, "n_query"] = np.clip(right_n - 1, 0, len(reference_times_ns) - 1)
    right_n = np.clip(right_n, 0, len(reference_times_ns) - 1)
    left_delta_n: Int64[np.ndarray, "n_query"] = np.abs(
        reference_times_ns[left_n] - query_times_ns
    )
    right_delta_n: Int64[np.ndarray, "n_query"] = np.abs(
        reference_times_ns[right_n] - query_times_ns
    )
    choose_right_n: np.ndarray = right_delta_n < left_delta_n
    nearest_n: Int64[np.ndarray, "n_query"] = np.where(
        choose_right_n, right_n, left_n
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
    """Compose a catalog rig pose and camera extrinsic into mapper convention.

    Args:
        world_from_rig_44: Float64 Basalt ``T_world_from_rig`` matrix with
            shape ``[4, 4]``.
        camera_from_rig_44: Float64 static catalog ``T_camera_from_rig`` matrix
            with shape ``[4, 4]``.

    Returns:
        Float32 lietorch vector ``[tx, ty, tz, qx, qy, qz, qw]`` for
        ``T_camera_from_world`` with shape ``[7]``.
    """
    camera_from_world_44: Float64[np.ndarray, "4 4"] = (
        camera_from_rig_44 @ np.linalg.inv(world_from_rig_44)
    )
    quaternion_xyzw: Float64[np.ndarray, "4"] = Rotation.from_matrix(
        camera_from_world_44[:3, :3]
    ).as_quat()
    pose_7: Float32[np.ndarray, "7"] = np.concatenate(
        [camera_from_world_44[:3, 3], quaternion_xyzw]
    ).astype(np.float32)
    return pose_7


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
        """Return selected pose indices, always including the first frame.

        Args:
            world_from_rig_n44: Float64 Basalt ``T_world_from_rig`` matrices
                with shape ``[n, 4, 4]``.

        Returns:
            Int64 selected indices with shape ``[n_keyframes]``.
        """
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
            cosine: float = float(
                np.clip((np.trace(relative_rotation_33) - 1.0) / 2.0, -1.0, 1.0)
            )
            angle_degrees: float = float(np.rad2deg(np.arccos(cosine)))
            if (
                translation_metres >= self.distance_metres
                or angle_degrees >= self.angle_degrees
            ):
                selected.append(index)
                last_index = index

        return np.asarray(selected, dtype=np.int64)


@dataclass(slots=True)
class FisheyeRectifier:
    """Rectify one OpenCV/Kannala-Brandt fisheye into a virtual pinhole."""

    intrinsic_33: Float64[np.ndarray, "3 3"]
    """Native fisheye calibration matrix in pixels."""
    distortion_4: Float64[np.ndarray, "4"]
    """KB4 coefficients ``k1..k4`` in OpenCV fisheye order."""
    output_size: tuple[int, int] = (640, 360)
    """Virtual image size as ``(width, height)``."""
    horizontal_fov_degrees: float = 110.0
    """Horizontal field of view of the virtual pinhole."""
    virtual_K: Float64[np.ndarray, "3 3"] = field(init=False)
    """Virtual-pinhole calibration matrix in output pixels."""
    map1: np.ndarray = field(init=False, repr=False)
    """First fixed-point remap table returned by OpenCV."""
    map2: np.ndarray = field(init=False, repr=False)
    """Second fixed-point remap table returned by OpenCV."""

    def __post_init__(self) -> None:
        """Build the immutable virtual calibration and remap tables."""
        width: int = self.output_size[0]
        height: int = self.output_size[1]
        half_fov_radians: float = np.deg2rad(self.horizontal_fov_degrees / 2.0)
        focal: float = (width / 2.0) / np.tan(half_fov_radians)
        self.virtual_K = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        remap: tuple[np.ndarray, np.ndarray] = cv2.fisheye.initUndistortRectifyMap(
            np.asarray(self.intrinsic_33, dtype=np.float64),
            np.asarray(self.distortion_4, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            self.virtual_K,
            self.output_size,
            cv2.CV_16SC2,
        )
        self.map1 = remap[0]
        self.map2 = remap[1]

    def rectify_points(
        self,
        distorted_xy: Float64[np.ndarray, "n 2"],
    ) -> Float64[np.ndarray, "n 2"]:
        """Map native KB4 pixels to the virtual pinhole.

        Args:
            distorted_xy: Float64 native-image pixels with shape ``[n, 2]``.

        Returns:
            Float64 virtual-image pixels with shape ``[n, 2]``.
        """
        points_n12: Float64[np.ndarray, "n 1 2"] = np.asarray(
            distorted_xy, dtype=np.float64
        )[:, None, :]
        rectified_n12: Float64[np.ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(
            points_n12,
            np.asarray(self.intrinsic_33, dtype=np.float64),
            np.asarray(self.distortion_4, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            P=self.virtual_K,
        )
        return rectified_n12[:, 0, :]

    def rectify(
        self,
        image_rgb_hwc: UInt8[np.ndarray, "native_h native_w 3"],
    ) -> UInt8[np.ndarray, "output_h output_w 3"]:
        """Rectify one native RGB image with the precomputed maps.

        Args:
            image_rgb_hwc: UInt8 native RGB image with shape ``[H, W, 3]``.

        Returns:
            UInt8 virtual RGB image with shape ``[output_h, output_w, 3]``.
        """
        rectified_rgb_hwc: UInt8[np.ndarray, "output_h output_w 3"] = cv2.remap(
            image_rgb_hwc,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return rectified_rgb_hwc


class RobocapSegment:
    """Read one robocap catalog segment as trusted-pose mapping keyframes."""

    def __init__(
        self,
        catalog_url: str = CATALOG_URL,
        dataset_id: str = DATASET_ID,
        segment_id: str = SEGMENT_ID,
        camera_ids: tuple[int, ...] = CAMERA_IDS,
        decoder: DecoderDevice = "cuda",
        fps: int = 30,
    ) -> None:
        """Connect to the catalog and read static camera calibration.

        Args:
            catalog_url: Rerun catalog endpoint.
            dataset_id: Catalog dataset UUID.
            segment_id: Exact Rerun segment identifier.
            camera_ids: Physical camera indices to include.
            decoder: TorchCodec device; ``cuda`` uses NVDEC and ``cpu`` is the
                explicit fallback.
            fps: Nominal packet rate used by ``wrap_mp4`` timescale metadata.
        """
        self.catalog_url: str = catalog_url
        self.dataset_id: str = dataset_id
        self.segment_id: str = segment_id
        self.camera_ids: tuple[int, ...] = camera_ids
        self.decoder: DecoderDevice = decoder
        self.fps: int = fps
        self.client: catalog.CatalogClient = catalog.CatalogClient(catalog_url)
        self.dataset: Any = self.client.get_dataset(id=dataset_id)
        self.calibrations: tuple[CameraCalibration, ...] = self._read_calibrations()
        self.rectifiers: tuple[FisheyeRectifier, ...] = tuple(
            FisheyeRectifier(calibration.intrinsic_33, calibration.distortion_4)
            for calibration in self.calibrations
        )
        self.segment_video_start_ns: int = self._read_segment_video_start_ns()
        self.cam00_frame0_ns: int = self._read_first_packet_ns(
            self.calibrations[0].video_path
        )

    @property
    def camera_names(self) -> list[str]:
        """Return chosen catalog camera names in mapper order."""
        return [calibration.name for calibration in self.calibrations]

    def _view(self, entities: list[str]) -> Any:
        """Return this segment filtered to the requested entities."""
        return self.dataset.filter_contents(entities).filter_segments([self.segment_id])

    def _read_calibrations(self) -> tuple[CameraCalibration, ...]:
        """Read KB4 intrinsics and ``T_camera_from_rig`` for chosen cameras."""
        entities: list[str] = []
        for camera_id in self.camera_ids:
            camera_path: str = f"{RIG_PATH}/cam_0{camera_id}"
            entities.extend([camera_path, f"{camera_path}/pinhole"])
        table: pa.Table = self._view(entities).reader(index=None).to_arrow_table()
        if table.num_rows == 0:
            raise RuntimeError(f"segment {self.segment_id} has no static camera calibration")
        row: dict[str, Any] = table.to_pylist()[0]
        calibrations: list[CameraCalibration] = []
        for camera_id in self.camera_ids:
            camera_path: str = f"{RIG_PATH}/cam_0{camera_id}"
            pinhole_path: str = f"{camera_path}/pinhole"
            name: str = str(_unwrap_component(row[f"{camera_path}:name"]))
            matrix_raw: Any = _unwrap_component(
                row[f"{camera_path}:Transform3D:mat3x3"]
            )
            rotation_camera_from_rig_33: Float64[np.ndarray, "3 3"] = np.asarray(
                matrix_raw, dtype=np.float64
            ).reshape(3, 3, order="F")
            translation_camera_from_rig_3: Float64[np.ndarray, "3"] = np.asarray(
                _unwrap_component(row[f"{camera_path}:Transform3D:translation"]),
                dtype=np.float64,
            )
            camera_from_rig_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
            camera_from_rig_44[:3, :3] = rotation_camera_from_rig_33
            camera_from_rig_44[:3, 3] = translation_camera_from_rig_3
            intrinsic_33: Float64[np.ndarray, "3 3"] = np.asarray(
                _unwrap_component(row[f"{pinhole_path}:Pinhole:image_from_camera"]),
                dtype=np.float64,
            ).reshape(3, 3, order="F")
            distortion_8: Float64[np.ndarray, "8"] = np.asarray(
                _unwrap_component(
                    row[f"{pinhole_path}:simplecv.components.DistortionCoefficients"]
                ),
                dtype=np.float64,
            )
            resolution_raw: Any = _unwrap_component(
                row[f"{pinhole_path}:Pinhole:resolution"]
            )
            resolution_values: list[int] = [int(value) for value in resolution_raw]
            calibrations.append(
                CameraCalibration(
                    camera_id=camera_id,
                    name=name,
                    intrinsic_33=intrinsic_33,
                    distortion_4=distortion_8[:4].copy(),
                    camera_from_rig_44=camera_from_rig_44,
                    resolution_wh=(resolution_values[0], resolution_values[1]),
                )
            )
        return tuple(calibrations)

    def _read_first_packet_ns(self, video_path: str) -> int:
        """Read the first packet timestamp for one selected video entity."""
        table: pa.Table = (
            self._view([video_path])
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

    def _read_segment_video_start_ns(self) -> int:
        """Read the earliest packet time across the four selected cameras."""
        video_paths: list[str] = [
            calibration.video_path for calibration in self.calibrations
        ]
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
        calibration: CameraCalibration,
        start_ns: int,
        end_ns: int,
    ) -> RawVideoStream:
        """Read one packet window, rounded back to its previous H.264 keyframe."""
        query_start_ns: int = max(0, start_ns - KEYFRAME_PREROLL_NS)
        video_path: str = calibration.video_path
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
        time_column: pa.ChunkedArray = table.column(TIMELINE).cast(pa.int64())
        sample_column: pa.ChunkedArray = table.column(
            f"{video_path}:VideoStream:sample"
        )
        keyframe_values: list[Any] = table.column(
            f"{video_path}:VideoStream:is_keyframe"
        ).to_pylist()
        packet_times: list[int] = []
        samples: list[bytes] = []
        is_keyframe: list[bool] = []
        for row_index in range(table.num_rows):
            sample_raw: Any = sample_column[row_index].as_py()
            if sample_raw is None:
                continue
            packet_times.append(int(time_column[row_index].as_py()))
            samples.append(bytes(_unwrap_component(sample_raw)))
            is_keyframe.append(bool(_unwrap_component(keyframe_values[row_index])))
        if not samples:
            raise RuntimeError(
                f"{calibration.name} has no packets in [{start_ns}, {end_ns})"
            )

        times_ns: Int64[np.ndarray, "n_samples"] = np.asarray(packet_times, dtype=np.int64)
        pre_roll_keyframes: list[int] = [
            index
            for index, (timestamp_ns, keyframe) in enumerate(zip(packet_times, is_keyframe))
            if keyframe and timestamp_ns <= start_ns
        ]
        first_index: int = pre_roll_keyframes[-1] if pre_roll_keyframes else 0
        times_ns = times_ns[first_index:]
        samples = samples[first_index:]
        is_keyframe = is_keyframe[first_index:]
        if not is_keyframe[0]:
            raise RuntimeError(
                f"{calibration.name} window does not start on an H.264 keyframe"
            )
        mp4: bytes = wrap_mp4(samples, is_keyframe, self.fps, codec="h264")
        decoder: VideoDecoder = VideoDecoder(
            mp4,
            device=self.decoder,
            seek_mode="exact",
            num_ffmpeg_threads=0,
        )
        return RawVideoStream(
            camera_id=calibration.camera_id,
            camera_name=calibration.name,
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
        times_column: pa.ChunkedArray = table.column(TIMELINE).cast(pa.int64())
        translations: list[Any] = table.column(
            f"{RIG_PATH}:Transform3D:translation"
        ).to_pylist()
        quaternions: list[Any] = table.column(
            f"{RIG_PATH}:Transform3D:quaternion"
        ).to_pylist()
        pose_times: list[int] = []
        poses: list[np.ndarray] = []
        for row_index in range(table.num_rows):
            if translations[row_index] is None or quaternions[row_index] is None:
                continue
            translation_3: Float64[np.ndarray, "3"] = np.asarray(
                _unwrap_component(translations[row_index]), dtype=np.float64
            )
            quaternion_xyzw: Float64[np.ndarray, "4"] = np.asarray(
                _unwrap_component(quaternions[row_index]), dtype=np.float64
            )
            world_from_rig_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
            world_from_rig_44[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
            world_from_rig_44[:3, 3] = translation_3
            pose_times.append(int(times_column[row_index].as_py()))
            poses.append(world_from_rig_44)
        if not poses:
            raise RuntimeError(f"segment {self.segment_id} has no Basalt poses in the window")
        pose_times_ns: Int64[np.ndarray, "n_poses"] = np.asarray(pose_times, dtype=np.int64)
        world_from_rig_n44: Float64[np.ndarray, "n_poses 4 4"] = np.stack(poses)
        return pose_times_ns, world_from_rig_n44

    def iter_keyframes(
        self,
        start_seconds: float = 0.0,
        end_seconds: float = 10.0,
        kf_dist: float = 0.3,
        kf_angle: float = 15.0,
    ) -> Iterator[CatalogKeyframe]:
        """Yield timestamp-joined, rectified, motion-selected catalog keyframes.

        Args:
            start_seconds: Window start relative to the segment's first video
                packet.
            end_seconds: Exclusive window end relative to the same origin.
            kf_dist: Translation threshold from the last keyframe in metres.
            kf_angle: Rotation threshold from the last keyframe in degrees.

        Yields:
            :class:`CatalogKeyframe` batches containing all four cameras.
        """
        if start_seconds < 0.0 or end_seconds <= start_seconds:
            raise ValueError("expected 0 <= start_seconds < end_seconds")
        start_ns: int = self.segment_video_start_ns + round(start_seconds * 1e9)
        end_ns: int = self.segment_video_start_ns + round(end_seconds * 1e9)
        relay_streams: tuple[RawVideoStream, ...] = tuple(
            self._read_video_window(calibration, start_ns, end_ns)
            for calibration in self.calibrations
        )
        pose_result: tuple[
            Int64[np.ndarray, "n_poses"],
            Float64[np.ndarray, "n_poses 4 4"],
        ] = self._read_poses(start_ns, end_ns)
        pose_times_ns: Int64[np.ndarray, "n_poses"] = pose_result[0]
        world_from_rig_n44: Float64[np.ndarray, "n_poses 4 4"] = pose_result[1]

        reference_times_ns: Int64[np.ndarray, "n_reference"] = relay_streams[0].times_ns
        requested_mask_n: np.ndarray = (
            (reference_times_ns >= start_ns)
            & (reference_times_ns < end_ns)
            & (reference_times_ns != self.cam00_frame0_ns)
        )
        target_times_ns: Int64[np.ndarray, "n_targets"] = reference_times_ns[
            requested_mask_n
        ]
        stream_indices: list[Int64[np.ndarray, "n_targets"]] = [
            _nearest_indices(stream.times_ns, target_times_ns)
            for stream in relay_streams
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

        aligned_world_from_rig_n44: Float64[np.ndarray, "n_targets 4 4"] = (
            world_from_rig_n44[pose_indices_n]
        )
        selected_n: Int64[np.ndarray, "n_keyframes"] = KeyframeSelector(
            distance_metres=kf_dist,
            angle_degrees=kf_angle,
        ).select(aligned_world_from_rig_n44)
        virtual_K_n4: Float32[np.ndarray, "n_cams 4"] = np.asarray(
            [
                [
                    rectifier.virtual_K[0, 0],
                    rectifier.virtual_K[1, 1],
                    rectifier.virtual_K[0, 2],
                    rectifier.virtual_K[1, 2],
                ]
                for rectifier in self.rectifiers
            ],
            dtype=np.float32,
        )
        for aligned_index in selected_n:
            images_rgb_chw: list[np.ndarray] = []
            camera_from_world_7: list[np.ndarray] = []
            world_from_rig_44: Float64[np.ndarray, "4 4"] = aligned_world_from_rig_n44[
                aligned_index
            ]
            for camera_index, stream in enumerate(relay_streams):
                decoder_index: int = int(stream_indices[camera_index][aligned_index])
                decoded_rgb_chw: UInt8[torch.Tensor, "3 native_h native_w"] = (
                    stream.decoder.get_frame_at(decoder_index).data
                )
                decoded_rgb_hwc: UInt8[np.ndarray, "native_h native_w 3"] = rearrange(
                    decoded_rgb_chw.detach().to(device="cpu").numpy(),
                    "c h w -> h w c",
                )
                rectified_rgb_hwc: UInt8[np.ndarray, "360 640 3"] = self.rectifiers[
                    camera_index
                ].rectify(decoded_rgb_hwc)
                rectified_rgb_chw: UInt8[np.ndarray, "3 360 640"] = rearrange(
                    rectified_rgb_hwc, "h w c -> c h w"
                ).copy()
                images_rgb_chw.append(rectified_rgb_chw)
                camera_from_world_7.append(
                    camera_from_world_pose(
                        world_from_rig_44,
                        self.calibrations[camera_index].camera_from_rig_44,
                    )
                )
            images_rgb_n3hw: UInt8[np.ndarray, "n_cams 3 360 640"] = np.stack(
                images_rgb_chw
            )
            camera_from_world_n7: Float32[np.ndarray, "n_cams 7"] = np.stack(
                camera_from_world_7
            ).astype(np.float32, copy=False)
            yield CatalogKeyframe(
                timestamp_ns=int(target_times_ns[aligned_index]),
                images_rgb=torch.from_numpy(images_rgb_n3hw),
                virtual_K=torch.from_numpy(virtual_K_n4.copy()),
                camera_from_world=torch.from_numpy(camera_from_world_n7),
                world_from_rig=world_from_rig_44.copy(),
                relay_streams=relay_streams,
            )
