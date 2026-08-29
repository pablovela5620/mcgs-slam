"""Serialized Rerun schema contracts for catalog mode."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr
import rerun.experimental as rrx
import torch
from jaxtyping import Float32, Float64
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rig import CameraSensor, RigCalibration

from catalog_stream import CatalogKeyframe, FisheyeRectifier
from rerun_logger_catalog import CatalogRerunLogger


def _camera_calibrations() -> tuple[RigCalibration, tuple[PinholeParameters, ...]]:
    """Build one literal four-camera SimpleCV rig and virtual pinholes."""
    cameras: list[CameraSensor] = []
    for camera_id, name in zip((0, 1, 4, 5), ("left_front", "right_front", "left", "right"), strict=True):
        camera_from_rig_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        camera_from_rig_44[0, 3] = camera_id / 100.0
        camera = CameraSensor(
            index=camera_id,
            name=name,
            kind="rgb",
            pinhole=Fisheye62Parameters(
                name=name,
                extrinsics=Extrinsics(
                    cam_R_world=camera_from_rig_44[:3, :3],
                    cam_t_world=camera_from_rig_44[:3, 3],
                ),
                intrinsics=Intrinsics.from_k_matrix(
                    camera_conventions="RDF",
                    k_matrix=np.array(
                        [[600.0, 0.0, 960.0], [0.0, 601.0, 540.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    ),
                    height=1080,
                    width=1920,
                ),
                distortion=KannalaBrandtDistortion(
                    k1=0.1,
                    k2=-0.02,
                    k3=0.003,
                    k4=-0.0004,
                ),
            ),
        )
        cameras.append(camera)
    rig_calibration = RigCalibration(cameras=cameras, reference_index=0)
    rectified_cameras: tuple[PinholeParameters, ...] = tuple(
        FisheyeRectifier(camera).virtual_camera for camera in cameras
    )
    return rig_calibration, rectified_cameras


def _entity_path(chunk: rrx.Chunk) -> str:
    """Normalize a serialized entity path to its absolute spelling."""
    return f"/{str(chunk.entity_path).lstrip('/')}"


def test_catalog_recording_uses_exoego_v2_paths_and_transform_relations(tmp_path: Path) -> None:
    rrd_path: Path = tmp_path / "catalog-schema.rrd"
    rig_calibration, rectified_cameras = _camera_calibrations()
    logger = CatalogRerunLogger(
        rig_calibration=rig_calibration,
        rectified_cameras=rectified_cameras,
        map_scale=1.0,
        save_path=str(rrd_path),
    )
    streams: list[SimpleNamespace] = [
        SimpleNamespace(
            times_ns=np.array([1_000_000_000], dtype=np.int64),
            samples=[b"\x00\x00\x00\x01"],
            is_keyframe=[True],
        )
        for _ in rig_calibration.cameras
    ]
    logger.relay_video_streams(streams)
    world_from_rig_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    frame = CatalogKeyframe(
        timestamp_ns=1_000_000_000,
        world_from_rig=world_from_rig_44,
        images_rgb=torch.zeros((4, 3, 8, 8), dtype=torch.uint8),
    )
    logger.log_catalog_keyframe(
        0,
        frame,
        depth_metres_nhw=torch.ones((4, 8, 8), dtype=torch.float32),
        normals_n3hw=torch.ones((4, 3, 8, 8), dtype=torch.float32),
    )
    logger.log_render_comparison(
        0,
        rendered_3hw=torch.zeros((3, 8, 8), dtype=torch.float32),
        gt_3hw=torch.ones((3, 8, 8), dtype=torch.float32),
    )
    logger.flush()
    rr.get_global_data_recording().disconnect()

    chunks: list[rrx.Chunk] = rrx.RrdReader(rrd_path).stream().to_chunks()
    entity_paths: set[str] = {_entity_path(chunk) for chunk in chunks}
    expected_paths: set[str] = {
        "/world/rig_00",
        "/world/rig_00/cam_00/rectified/render",
        "/world/trajectory",
    }
    for camera_id in (0, 1, 4, 5):
        camera_path: str = f"/world/rig_00/cam_{camera_id:02d}"
        expected_paths.update(
            {
                camera_path,
                f"{camera_path}/pinhole",
                f"{camera_path}/pinhole/video",
                f"{camera_path}/rectified",
                f"{camera_path}/rectified/pinhole",
                f"{camera_path}/rectified/image",
                f"{camera_path}/rectified/depth",
            }
        )
    assert expected_paths <= entity_paths
    assert not any(path.startswith("/world/keyframes/") for path in entity_paths)
    assert not any("virtual_pinhole" in path for path in entity_paths)
    assert not any(path.startswith("/render_vs_gt/") for path in entity_paths)
    assert not any(path.endswith("/rectified/gt") for path in entity_paths)

    root_chunk: rrx.Chunk = next(chunk for chunk in chunks if _entity_path(chunk) == "/")
    assert root_chunk.to_record_batch()["ViewCoordinates:xyz"].to_pylist() == [[[3, 5, 1]]]

    rig_metadata_chunk: rrx.Chunk = next(
        chunk
        for chunk in chunks
        if _entity_path(chunk) == "/world/rig_00" and chunk.is_static
    )
    rig_metadata: dict[str, list[object]] = rig_metadata_chunk.to_record_batch().to_pydict()
    assert rig_metadata["schema_version"] == [["exoego:v2"]]
    assert rig_metadata["reference"] == [["imu_00"]]
    assert rig_metadata["num_cameras"] == [[4]]
    assert rig_metadata["name"] == [["robocap"]]
    assert rig_metadata["kind"] == [["ego"]]

    rig_chunks: list[rrx.Chunk] = [
        chunk for chunk in chunks if _entity_path(chunk) == "/world/rig_00" and not chunk.is_static
    ]
    assert len(rig_chunks) == 1
    assert "video_time" in rig_chunks[0].timeline_names
    assert "Transform3D:relation" not in rig_chunks[0].to_record_batch().schema.names

    camera_transform_chunks: list[rrx.Chunk] = [
        chunk
        for chunk in chunks
        if _entity_path(chunk) == "/world/rig_00/cam_00"
        and "Transform3D:relation" in chunk.to_record_batch().schema.names
    ]
    assert len(camera_transform_chunks) == 1
    relation_values: list[list[int]] = camera_transform_chunks[0].to_record_batch()["Transform3D:relation"].to_pylist()
    assert relation_values == [[rr.TransformRelation.ChildFromParent.value]]

    rectified_transform_chunk: rrx.Chunk = next(
        chunk
        for chunk in chunks
        if _entity_path(chunk) == "/world/rig_00/cam_00/rectified"
        and "Transform3D:relation" in chunk.to_record_batch().schema.names
    )
    rectified_transform: dict[str, list[object]] = rectified_transform_chunk.to_record_batch().to_pydict()
    assert rectified_transform["Transform3D:relation"] == [[rr.TransformRelation.ChildFromParent.value]]
    assert rectified_transform["Transform3D:translation"] == [[[0.0, 0.0, 0.0]]]

    pinhole_columns: set[str] = {
        column
        for chunk in chunks
        if _entity_path(chunk) == "/world/rig_00/cam_00/pinhole"
        for column in chunk.to_record_batch().schema.names
    }
    assert "simplecv.components.DistortionModel" in pinhole_columns
    assert "simplecv.components.DistortionCoefficients" in pinhole_columns
    pinhole_chunk: rrx.Chunk = next(
        chunk
        for chunk in chunks
        if _entity_path(chunk) == "/world/rig_00/cam_00/pinhole"
        and "simplecv.components.DistortionModel" in chunk.to_record_batch().schema.names
    )
    pinhole_components: dict[str, list[object]] = pinhole_chunk.to_record_batch().to_pydict()
    assert pinhole_components["simplecv.components.DistortionModel"] == [["kannala_brandt"]]
    coefficients: list[float] = pinhole_components["simplecv.components.DistortionCoefficients"][0][0]
    assert np.allclose(coefficients, [0.1, -0.02, 0.003, -0.0004, 0.0, 0.0, 0.0, 0.0])


def test_catalog_logger_rejects_nonmetric_map_scale(tmp_path: Path) -> None:
    rig_calibration, rectified_cameras = _camera_calibrations()

    with pytest.raises(ValueError, match="map_scale.*1.0"):
        CatalogRerunLogger(
            rig_calibration=rig_calibration,
            rectified_cameras=rectified_cameras,
            map_scale=0.2,
            save_path=str(tmp_path / "invalid.rrd"),
        )
