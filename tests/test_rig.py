"""Rig calibration and factor-graph contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT: Path = Path(__file__).resolve().parents[1]

from factor_graph import FactorGraph
from depth_video import DepthVideo
import options
from options import load_configs


def _write_calibration(path: Path, camera_count: int, extrinsic_count: int) -> None:
    """Write a minimal calibration with independent camera and rig row counts."""
    identity: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    calibration: dict[str, object] = {
        "intrinsic": [[100.0, 100.0, 50.0, 40.0] for _ in range(camera_count)],
        "camera": "pinhole",
        "T_cami_cam0": [identity for _ in range(extrinsic_count)],
        "timescale": 1,
        "ht": 80,
        "wd": 104,
    }
    path.write_text(yaml.safe_dump(calibration), encoding="utf-8")


def test_calibration_loads_one_rig_transform_per_directory(tmp_path: Path) -> None:
    calibration_path: Path = tmp_path / "six-camera.yml"
    camera_count: int = 6
    _write_calibration(calibration_path, camera_count, camera_count)
    args: SimpleNamespace = SimpleNamespace(
        calib=str(calibration_path),
        imagedir=[f"camera-{index}" for index in range(camera_count)],
    )

    loaded: SimpleNamespace = load_configs(args)

    assert loaded.multi == len(args.imagedir) == camera_count
    assert loaded.T_cami_cam0.shape == (camera_count, 7)


def test_calibration_rejects_mismatched_rig_row_count(tmp_path: Path) -> None:
    calibration_path: Path = tmp_path / "bad-rig.yml"
    _write_calibration(calibration_path, camera_count=3, extrinsic_count=2)
    args: SimpleNamespace = SimpleNamespace(
        calib=str(calibration_path),
        imagedir=["front", "front-left", "front-right"],
    )

    with pytest.raises(ValueError, match=r"T_cami_cam0.*2.*3"):
        load_configs(args)


def test_calibration_rejects_nonidentity_camera_zero_transform(tmp_path: Path) -> None:
    calibration_path: Path = tmp_path / "bad-camera-zero.yml"
    _write_calibration(calibration_path, camera_count=2, extrinsic_count=2)
    calibration: dict[str, object] = yaml.safe_load(
        calibration_path.read_text(encoding="utf-8")
    )
    rig_rows: list[list[float]] = calibration["T_cami_cam0"]  # type: ignore[assignment]
    rig_rows[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    calibration_path.write_text(yaml.safe_dump(calibration), encoding="utf-8")
    args: SimpleNamespace = SimpleNamespace(
        calib=str(calibration_path),
        imagedir=["camera-zero", "camera-one"],
    )

    with pytest.raises(ValueError, match=r"T_cami_cam0\[0\].*identity"):
        load_configs(args)


def test_static_intrinsics_row_has_no_batch_dimension() -> None:
    video: DepthVideo = DepthVideo.__new__(DepthVideo)
    video.intrinsics = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)

    intrinsics: torch.Tensor = video.K_row(1)

    assert intrinsics.shape == (8,)
    assert torch.equal(intrinsics, video.intrinsics[0, 1])


@pytest.mark.parametrize(
    ("calibration_name", "camera_names"),
    [
        ("100613.yml", ["front", "front_left", "front_right"]),
        (
            "drones.yml",
            ["front_left", "front_center", "front_right", "left_center", "right_center"],
        ),
    ],
)
def test_example_calibrations_match_directory_order(
    calibration_name: str,
    camera_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(options.os, "listdir", lambda _: ["frame.png"])
    monkeypatch.setattr(
        options.cv2,
        "imread",
        lambda _: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    args: SimpleNamespace = SimpleNamespace(
        calib=str(ROOT / "calib" / calibration_name),
        imagedir=camera_names,
    )

    loaded: SimpleNamespace = load_configs(args)

    assert loaded.multi == len(camera_names)
    assert loaded.T_cami_cam0.shape == (len(camera_names), 7)


class _SyntheticVideo:
    """Small CPU video seam for testing temporal edge construction."""

    def __init__(self, frame_count: int) -> None:
        self.counter: SimpleNamespace = SimpleNamespace(value=frame_count)
        self.ds: torch.Tensor | None = None
        self.nets: torch.Tensor = torch.zeros((frame_count, 1, 2), dtype=torch.float32)

    def distance(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Return finite distances for all candidate frame pairs."""
        del jj, beta
        return torch.zeros_like(ii, dtype=torch.float32)

    def reproject(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        camera_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero flow at the mapper's expected factor shape."""
        del jj, camera_index
        target: torch.Tensor = torch.zeros((1, len(ii), 1, 1, 2), dtype=torch.float32)
        return target, torch.ones_like(target)


def _edge_builder(frame_count: int = 6) -> tuple[FactorGraph, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Create a factor graph shell that records edges instead of building correlations."""
    graph: FactorGraph = FactorGraph.__new__(FactorGraph)
    graph.video = _SyntheticVideo(frame_count)
    graph.device = "cpu"
    graph.index = 0
    graph.max_factors = 1_000
    graph.ii = torch.empty(0, dtype=torch.long)
    graph.jj = torch.empty(0, dtype=torch.long)
    graph.ii_inac = torch.empty(0, dtype=torch.long)
    graph.jj_inac = torch.empty(0, dtype=torch.long)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(ii: torch.Tensor, jj: torch.Tensor, remove: bool = False) -> None:
        del remove
        captured.append((ii.cpu(), jj.cpu()))

    graph.add_factors = capture
    return graph, captured


def _real_edge_graph(frame_count: int = 6) -> FactorGraph:
    """Create a CPU factor graph that stores edges without a correlation volume."""
    graph: FactorGraph = FactorGraph.__new__(FactorGraph)
    graph.video = _SyntheticVideo(frame_count)
    graph.device = "cpu"
    graph.index = 0
    graph.max_factors = 1_000
    graph.corr_impl = "none"
    graph.ii = torch.empty(0, dtype=torch.long)
    graph.jj = torch.empty(0, dtype=torch.long)
    graph.age = torch.empty(0, dtype=torch.long)
    graph.ii_inac = torch.empty(0, dtype=torch.long)
    graph.jj_inac = torch.empty(0, dtype=torch.long)
    graph.eset = set()
    graph.corr = None
    graph.net = None
    graph.inp = None
    graph.target = torch.zeros((1, 0, 1, 1, 2), dtype=torch.float32)
    graph.weight = torch.zeros((1, 0, 1, 1, 2), dtype=torch.float32)
    return graph


def test_factor_builders_never_emit_self_edges() -> None:
    graph, captured = _edge_builder()
    graph.add_neighborhood_factors(0, 6, r=2)
    graph.add_proximity_factors(rad=2, nms=1, thresh=16.0)
    graph.add_proximity_factors_backend(
        t=6,
        rad=2,
        nms=1,
        beta=0.25,
        thresh=16.0,
        framedist=2,
        relset=set(),
    )

    assert captured
    for ii, jj in captured:
        assert torch.all(ii != jj), (ii, jj)


def test_every_factor_builder_honors_minimum_temporal_gap() -> None:
    neighborhood: FactorGraph = _real_edge_graph()
    neighborhood.add_neighborhood_factors(0, 6, r=3)

    frontend: FactorGraph = _real_edge_graph()
    frontend.add_proximity_factors(rad=2, nms=1, thresh=16.0)

    backend: FactorGraph = _real_edge_graph()
    backend.add_proximity_factors_backend(
        t=6,
        rad=2,
        nms=1,
        beta=0.25,
        thresh=16.0,
        framedist=1,
        relset=set(),
    )

    for graph in (neighborhood, frontend, backend):
        temporal_gaps: torch.Tensor = (graph.ii - graph.jj).abs()
        assert temporal_gaps.numel() > 0
        assert torch.all(temporal_gaps >= FactorGraph.MIN_TEMPORAL_GAP), (
            graph.ii,
            graph.jj,
        )


def test_add_factors_rejects_a_self_edge() -> None:
    graph: FactorGraph = FactorGraph.__new__(FactorGraph)
    graph.device = "cpu"
    graph.ii = torch.empty(0, dtype=torch.long)
    graph.jj = torch.empty(0, dtype=torch.long)
    graph.ii_bad = torch.empty(0, dtype=torch.long)
    graph.jj_bad = torch.empty(0, dtype=torch.long)
    graph.ii_inac = torch.empty(0, dtype=torch.long)
    graph.jj_inac = torch.empty(0, dtype=torch.long)

    with pytest.raises(AssertionError, match="self-edge reached FactorGraph.add_factors"):
        graph.add_factors(torch.tensor([2]), torch.tensor([2]))


def test_removing_middle_keyframe_drops_newly_adjacent_active_and_inactive_edges() -> None:
    graph: FactorGraph = _real_edge_graph(frame_count=5)
    graph.index = 1
    graph.ii = torch.tensor([0, 0], dtype=torch.long)
    graph.jj = torch.tensor([2, 3], dtype=torch.long)
    graph.age = torch.tensor([10, 20], dtype=torch.long)
    graph.net = torch.tensor([[[10.0], [20.0]]])
    graph.inp = torch.tensor([[[[10.0]], [[20.0]]]])
    graph.target = torch.tensor([[[[[10.0, 10.0]]], [[[20.0, 20.0]]]]])
    graph.weight = torch.tensor([[[[[1.0, 1.0]]], [[[2.0, 2.0]]]]])
    graph.ii_inac = torch.tensor([2, 3], dtype=torch.long)
    graph.jj_inac = torch.tensor([0, 0], dtype=torch.long)
    graph.target_inac = torch.tensor([[[[[30.0, 30.0]]], [[[40.0, 40.0]]]]])
    graph.weight_inac = torch.tensor([[[[[3.0, 3.0]]], [[[4.0, 4.0]]]]])
    graph.eset = {(0, 2), (0, 3), (2, 0), (3, 0)}

    graph.rm_keyframe(1)

    assert (graph.ii.tolist(), graph.jj.tolist()) == ([0], [2])
    assert graph.age.tolist() == [20]
    assert graph.net.flatten().tolist() == [20.0]
    assert graph.inp.flatten().tolist() == [20.0]
    assert graph.target.flatten().tolist() == [20.0, 20.0]
    assert graph.weight.flatten().tolist() == [2.0, 2.0]
    assert (graph.ii_inac.tolist(), graph.jj_inac.tolist()) == ([2], [0])
    assert graph.target_inac.flatten().tolist() == [40.0, 40.0]
    assert graph.weight_inac.flatten().tolist() == [4.0, 4.0]
    assert graph.eset == {(0, 2), (2, 0)}
