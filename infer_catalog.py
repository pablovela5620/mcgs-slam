#!/usr/bin/env python3
"""Map one robocap catalog window with Basalt poses and MoGe-2 geometry."""

from mcgs_slam._paths import configure_import_paths  # nopep8

configure_import_paths()  # nopep8

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import torch
import tyro

from catalog_stream import (
    CatalogKeyframe,
    RobocapSegment,
    camera_from_world_poses,
)
from gs_backend import GSBackEnd
from camera_packet import CameraPacket, build_camera_packet
from prior import MoGePrior, PriorPrediction
from rerun_logger_catalog import CatalogRerunLogger
from utils.utils import load_config


CATALOG_URL: str = "rerun+http://pablo-dl-server.ilish-ruler.ts.net:51235"
DATASET_ID: str = "18CFB19109CFDB071d88fb8b48ef65e9"
SEGMENT_ID: str = "robocap__f408193e6447b3b0__s00000021"
MAP_SCALE: float = 1.0


@dataclass
class CatalogConfig:
    """Trusted-pose robocap catalog mapping options."""

    catalog_url: str = CATALOG_URL
    """Rerun catalog endpoint."""
    dataset_id: str = DATASET_ID
    """Robocap catalog dataset UUID."""
    segment_id: str = SEGMENT_ID
    """Exact robocap segment identifier."""
    start: float = 0.0
    """Window start in seconds relative to the segment's first video packet."""
    end: float = 10.0
    """Exclusive window end in seconds relative to the first video packet."""
    output: Path = Path("output/robocap")
    """Output directory for the RRD, PLY, renders, and PSNR JSON."""
    decoder: Literal["cuda", "cpu"] = "cuda"
    """TorchCodec device. CUDA uses NVDEC; CPU is the explicit fallback."""
    kf_dist: float = 0.3
    """Basalt translation needed for a new keyframe, in metres."""
    kf_angle: float = 15.0
    """Basalt rotation needed for a new keyframe, in degrees."""
    refine_iters: int = 2000
    """Final Gaussian color-refinement iterations."""
    config: Path = Path("config/config_robocap.yaml")
    """Metric Gaussian mapper configuration."""


def run(config: CatalogConfig) -> dict[str, float | int | str]:
    """Run trusted-pose catalog mapping and return its summary.

    Args:
        config: Parsed catalog CLI configuration.

    Returns:
        JSON-compatible wall-time, keyframe, Gaussian, and quality summary.
    """
    cv2.setNumThreads(1)
    torch.multiprocessing.set_start_method("spawn", force=True)
    if not torch.cuda.is_available():
        raise RuntimeError("the Gaussian mapper and MoGe-2 require CUDA")
    if config.end <= config.start:
        raise ValueError("--end must be greater than --start")
    if config.refine_iters < 1:
        raise ValueError("--refine-iters must be positive")

    config.output.mkdir(parents=True, exist_ok=True)
    started: float = time.perf_counter()
    segment = RobocapSegment(
        catalog_url=config.catalog_url,
        dataset_id=config.dataset_id,
        segment_id=config.segment_id,
        decoder=config.decoder,
    )
    mapper_config: dict = load_config(str(config.config))
    mapper_config["opt_params"]["position_lr_max_steps"] = config.refine_iters
    rrd_path: Path = config.output / "mcgs_catalog.rrd"
    rr_logger = CatalogRerunLogger(
        rig_calibration=segment.rig_calibration,
        rectified_cameras=segment.rectified_cameras,
        map_scale=MAP_SCALE,
        save_path=str(rrd_path),
        max_splat_scale=0.3,
        max_depth=15.0,
        image_plane_distance=0.25,
        trajectory_radius=0.04,
        splat_every=4,
    )

    prior = MoGePrior("vits", resolution_level=3, device="cuda")
    backend = GSBackEnd(mapper_config, str(config.output), use_gui=False, rr_logger=rr_logger)
    keyframe_count: int = 0
    with segment.open_window(
        start_seconds=config.start,
        end_seconds=config.end,
        kf_dist=config.kf_dist,
        kf_angle=config.kf_angle,
    ) as window:
        rr_logger.relay_video_streams(window.relay_streams)
        frame: CatalogKeyframe
        for frame in window.keyframes():
            prediction: PriorPrediction = prior(
                frame.images_rgb,
                segment.virtual_intrinsics[:, 0],
            )
            rr_logger.log_catalog_keyframe(
                keyframe_count,
                frame,
                prediction.depth,
                prediction.normal,
            )
            camera_from_world_n7: torch.Tensor = camera_from_world_poses(
                frame.world_from_rig,
                segment.rig_calibration,
            )
            frame_ids: torch.Tensor = torch.tensor(
                [keyframe_count], dtype=torch.long
            )
            camera_count: int = len(segment.rig_calibration.cameras)
            for cam_idx in range(camera_count):
                packet: CameraPacket = build_camera_packet(
                    frame_ids=frame_ids,
                    cam_idx=cam_idx,
                    n_cameras=camera_count,
                    poses_camera_from_world_n7=camera_from_world_n7[
                        cam_idx : cam_idx + 1
                    ],
                    images_rgb_n3hw=frame.images_rgb[cam_idx : cam_idx + 1],
                    depth_metres_nhw=prediction.depth[cam_idx : cam_idx + 1],
                    normals_n3hw=prediction.normal[cam_idx : cam_idx + 1],
                    intrinsics_n4=segment.virtual_intrinsics[cam_idx : cam_idx + 1],
                    map_scale=MAP_SCALE,
                )
                backend.process_track_data(packet)
            rr_logger.log_gaussians(backend.gaussians)
            keyframe_count += 1
            print(
                f"keyframe {keyframe_count:03d} "
                f"video_time={frame.timestamp_ns / 1e9:.6f}s "
                f"gaussians={backend.gaussians.get_xyz.shape[0]}"
            )

    backend.refine_existing_viewpoints(iters=10)
    backend.finalize()
    quality: dict[str, float] = backend.eval_rendering_kf()
    rr_logger.send_final_blueprint()
    rr_logger.flush()

    elapsed_seconds: float = time.perf_counter() - started
    summary: dict[str, float | int | str] = {
        "wall_time_seconds": elapsed_seconds,
        "keyframes": keyframe_count,
        "gaussians": int(backend.gaussians.get_xyz.shape[0]),
        "mean_psnr": float(quality["mean_psnr"]),
        "mean_ssim": float(quality["mean_ssim"]),
        "mean_lpips": float(quality["mean_lpips"]),
        "rrd": str(rrd_path),
        "ply": str(config.output / "3dgs_final.ply"),
        "psnr_json": str(config.output / "psnr" / "after_opt" / "final_result_kf.json"),
    }
    print("RUN_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    """Parse the tyro CLI and run catalog mapping."""
    run(tyro.cli(CatalogConfig))


if __name__ == "__main__":
    main()
