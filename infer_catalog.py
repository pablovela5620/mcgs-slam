#!/usr/bin/env python3
"""Map one robocap catalog window with Basalt poses and MoGe-2 geometry."""

import os  # nopep8
import sys  # nopep8

_ROOT = os.path.dirname(os.path.abspath(__file__))  # nopep8
sys.path.append(os.path.join(_ROOT, "mcgs_slam"))  # nopep8
# CUDA extensions are built in-place by `pixi run build` (see pixi.toml).
sys.path.append(os.path.join(_ROOT, "thirdparty/lietorch"))  # nopep8
sys.path.append(os.path.join(_ROOT, "thirdparty/simple-knn"))  # nopep8
sys.path.append(os.path.join(_ROOT, "thirdparty/diff-gaussian-rasterization"))  # nopep8

import json
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Literal

import cv2
import rerun as rr
import torch
import tyro

from catalog_stream import CATALOG_URL, DATASET_ID, SEGMENT_ID, CatalogKeyframe, RobocapSegment
from gs_backend import GSBackEnd
from mcgs import CameraPacket, build_camera_packet
from prior import MoGePrior, PriorPrediction
from rerun_logger import RerunLogger
from utils.utils import load_config


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


def _merge_packets(packets: list[CameraPacket]) -> dict[str, object]:
    """Stack per-view packets for the backend's final re-registration pass."""
    tensor_keys: tuple[str, ...] = (
        "viz_idx",
        "tstamp",
        "poses",
        "images",
        "normals",
        "depths",
        "intrinsics",
    )
    merged: dict[str, object] = {
        key: torch.cat([packet[key] for packet in packets], dim=0)
        for key in tensor_keys
    }
    merged["cam_idx"] = torch.cat(
        [
            torch.full_like(packet["viz_idx"], packet["cam_idx"], dtype=torch.long)
            for packet in packets
        ]
    )
    merged["pose_updates"] = None
    merged["scale_updates"] = None
    return merged


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
    keyframe_iterator = segment.iter_keyframes(
        start_seconds=config.start,
        end_seconds=config.end,
        kf_dist=config.kf_dist,
        kf_angle=config.kf_angle,
    )
    first_frame: CatalogKeyframe = next(keyframe_iterator)

    mapper_config: dict = load_config(str(config.config))
    mapper_config["opt_params"]["position_lr_max_steps"] = config.refine_iters
    rrd_path: Path = config.output / "mcgs_catalog.rrd"
    rr_logger = RerunLogger(
        camera_names=segment.camera_names,
        camera_ids=list(segment.camera_ids),
        scale_factor=1.0,
        save_path=str(rrd_path),
        max_splat_scale=0.3,
        max_depth=15.0,
        image_plane_distance=0.25,
        trajectory_radius=0.04,
        splat_every=4,
        catalog_mode=True,
        world_coordinates=rr.ViewCoordinates.RFU,
    )
    rr_logger.log_catalog_calibration(
        segment.calibrations,
        first_frame.virtual_K,
    )
    rr_logger.relay_video_streams(first_frame.relay_streams)

    prior = MoGePrior("vits", resolution_level=3, device="cuda")
    backend = GSBackEnd(mapper_config, str(config.output), use_gui=False, rr_logger=rr_logger)
    packets: list[CameraPacket] = []
    keyframe_count: int = 0
    frame: CatalogKeyframe
    for frame in chain((first_frame,), keyframe_iterator):
        prediction: PriorPrediction = prior(
            frame.images_rgb,
            frame.virtual_K[:, 0],
        )
        rr_logger.log_catalog_keyframe(
            keyframe_count,
            frame,
            prediction.depth,
            prediction.normal,
        )
        viz_idx: torch.Tensor = torch.tensor([keyframe_count], dtype=torch.long)
        for cam_idx in range(len(segment.calibrations)):
            packet: CameraPacket = build_camera_packet(
                viz_idx=viz_idx,
                cam_idx=cam_idx,
                poses_camera_from_world_n7=frame.camera_from_world[cam_idx : cam_idx + 1],
                images_rgb_n3hw=frame.images_rgb[cam_idx : cam_idx + 1],
                depth_metres_nhw=prediction.depth[cam_idx : cam_idx + 1],
                normals_n3hw=prediction.normal[cam_idx : cam_idx + 1],
                intrinsics_n4=frame.virtual_K[cam_idx : cam_idx + 1],
                scale_factor=1.0,
            )
            backend.process_track_data(packet)
            packets.append(packet)
        rr_logger.log_gaussians(backend.gaussians)
        keyframe_count += 1
        print(
            f"keyframe {keyframe_count:03d} "
            f"video_time={frame.timestamp_ns / 1e9:.6f}s "
            f"gaussians={backend.gaussians.get_xyz.shape[0]}"
        )

    backend.process_global_track_data(_merge_packets(packets), len(segment.calibrations))
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
