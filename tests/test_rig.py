"""Rig calibration and factor-graph contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT: Path = Path(__file__).resolve().parents[1]

from factor_graph import FactorGraph
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

    def distance(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Return finite distances for all candidate frame pairs."""
        del jj, beta
        return torch.zeros_like(ii, dtype=torch.float32)


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
