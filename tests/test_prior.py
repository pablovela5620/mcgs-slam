"""MoGePrior contract on a real Waymo frame: shapes, devices, masking, focal handling."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from prior import MoGePrior, PriorPrediction

ROOT: Path = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data/100613"
pytestmark = [
    pytest.mark.skipif(not SEQ.is_dir(), reason="example sequence not downloaded"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.fixture(scope="module")
def prior() -> MoGePrior:
    return MoGePrior("vits", resolution_level=3)


@pytest.fixture(scope="module")
def frame() -> tuple[torch.Tensor, torch.Tensor]:
    with (ROOT / "calib/100613.yml").open(encoding="utf-8") as calibration_file:
        calib = np.array(yaml.safe_load(calibration_file)["intrinsic"])
    dirs = ["front", "front_left", "front_right"]
    name = sorted(os.listdir(SEQ / dirs[0]))[10]
    imgs = [cv2.resize(cv2.imread(str(SEQ / directory / name)), (536, 360))[:, :, ::-1].copy() for directory in dirs]
    images = torch.stack([torch.as_tensor(i).permute(2, 0, 1) for i in imgs])           # uint8 RGB [3,3,360,536]
    fx = torch.tensor([calib[i][0] * 536 / 1920 for i in range(3)])
    return images, fx


@pytest.fixture(scope="module")
def prediction(prior: MoGePrior, frame: tuple[torch.Tensor, torch.Tensor]) -> PriorPrediction:
    """Run the shared three-camera MoGe prediction once for contract tests."""
    images, fx = frame
    return prior(images, fx)


def test_output_contract(prediction: PriorPrediction) -> None:
    p = prediction
    assert p.depth.shape == (3, 360, 536) and p.depth.dtype == torch.float32 and p.depth.device.type == "cpu"
    assert p.normal.shape == (3, 3, 360, 536) and p.normal.dtype == torch.float32 and p.normal.device.type == "cpu"
    assert p.valid.shape == (3, 360, 536) and p.valid.dtype == torch.bool
    assert torch.isfinite(p.depth).all() and torch.isfinite(p.normal).all()


def test_invalid_pixels_are_zeroed_and_valid_are_metric(prediction: PriorPrediction) -> None:
    p = prediction
    assert (p.depth[~p.valid] == 0).all()
    assert (p.normal.permute(0, 2, 3, 1)[~p.valid] == 0).all()
    assert (p.depth[p.valid] > 0).all()
    # driving scene: road plane at a few meters, buildings tens of meters
    assert 3.0 < p.depth[0][p.valid[0]].median() < 60.0
    norms = p.normal.permute(0, 2, 3, 1)[p.valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_sky_is_masked_out(prediction: PriorPrediction) -> None:
    p = prediction
    valid = p.valid[0].float()
    assert 0.5 < valid.mean() < 1.0
    assert valid[:60].mean() < valid[-60:].mean()      # top band (sky) less valid than bottom band (road)


def test_normals_point_toward_camera_on_the_road(prediction: PriorPrediction) -> None:
    """Road plane normal is 'up' = -y in the RDF camera frame (matches Metric3D's convention)."""
    p = prediction
    road = p.normal[0, :, -40:, 200:340]               # bottom-center patch
    assert road[1].mean() < -0.8


def test_focal_length_sets_the_field_of_view(prior: MoGePrior, frame: tuple[torch.Tensor, torch.Tensor]) -> None:
    images, fx = frame
    a = prior(images[:1], fx[:1])
    b = prior(images[:1], fx[:1] * 2.0)                # narrower FoV → scene geometrically further away
    v = a.valid[0] & b.valid[0]
    assert (b.depth[0][v] / a.depth[0][v]).median() > 1.3
