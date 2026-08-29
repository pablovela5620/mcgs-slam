"""image_stream must reproduce the original decode→undistort→resize pipeline on real frames."""
import os
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from streams import image_stream

ROOT: Path = Path(__file__).resolve().parents[1]

SEQ: Path = ROOT / "data/100613"
DIRS = [str(SEQ / directory) for directory in ("front", "front_left", "front_right")]
pytestmark = pytest.mark.skipif(not SEQ.is_dir(), reason="example sequence not downloaded")


def _args(reduced: bool = False) -> SimpleNamespace:
    return SimpleNamespace(stride=1, ht=360, wd=536, timescale=1.0, decode_reduced=reduced)


def _calib() -> np.ndarray:
    with (ROOT / "calib/100613.yml").open(encoding="utf-8") as calibration_file:
        return np.array(yaml.safe_load(calibration_file)["intrinsic"])


def _reference_frame(t: int, calib: np.ndarray, a: SimpleNamespace):
    """The upstream implementation: full-res decode, cv2.undistort per call, resize."""
    images, intrinsics = [], torch.zeros((len(DIRS), 8))
    for i, d in enumerate(DIRS):
        names = sorted(os.listdir(d), key=lambda x: float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]))
        img = cv2.imread(os.path.join(d, names[t]))
        K = np.array([[calib[i][0], 0, calib[i][2]], [0, calib[i][1], calib[i][3]], [0, 0, 1]])
        img = cv2.undistort(img, K, calib[i][4:])
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (a.wd, a.ht))
        images.append(torch.as_tensor(img).permute(2, 0, 1))
    intrinsics[:, :4] = torch.tensor(calib[:, :4])
    intrinsics[:, [0, 2]] *= a.wd / w0
    intrinsics[:, [1, 3]] *= a.ht / h0
    return torch.stack(images), intrinsics


def _frame(t: int, a: SimpleNamespace, calib: np.ndarray):
    gen = image_stream(DIRS, calib, a)
    for _ in range(t + 1):
        tt, images, intrinsics, stamp = next(gen)
    assert tt == t
    return images, intrinsics


@pytest.mark.parametrize("t", [0, 7])
def test_default_is_bit_exact_with_reference_pipeline(t: int) -> None:
    a, calib = _args(), _calib()
    images, intrinsics = _frame(t, a, calib)
    ref_images, ref_intr = _reference_frame(t, calib, a)
    assert images.shape == ref_images.shape == (3, 3, 360, 536) and images.dtype == torch.uint8
    assert torch.allclose(intrinsics, ref_intr, atol=1e-3)
    assert torch.equal(images, ref_images)


def test_reduced_decode_is_close_to_reference_pipeline() -> None:
    """Reduced decode stays below the documented <2/255 mean and <=16/255 p99 bounds."""
    a, calib = _args(reduced=True), _calib()
    images, intrinsics = _frame(7, a, calib)
    ref_images, ref_intr = _reference_frame(7, calib, a)
    assert torch.allclose(intrinsics, ref_intr, atol=1e-3)
    diff = (images[:, :, 8:-8, 8:-8].float() - ref_images[:, :, 8:-8, 8:-8].float()).abs()
    assert diff.mean().item() < 2.0, diff.mean()
    assert torch.quantile(diff.flatten(), 0.99).item() <= 16, torch.quantile(diff.flatten(), 0.99)


def test_intrinsics_use_each_cameras_native_resolution(tmp_path: Path) -> None:
    camera_dirs: list[str] = []
    for camera_index, image_hw in enumerate(((100, 200), (200, 400))):
        camera_dir = tmp_path / f"camera-{camera_index}"
        camera_dir.mkdir()
        cv2.imwrite(str(camera_dir / "1.0.png"), np.zeros((*image_hw, 3), dtype=np.uint8))
        camera_dirs.append(str(camera_dir))

    calibration = np.array([
        [100.0, 100.0, 50.0, 50.0],
        [200.0, 200.0, 100.0, 100.0],
    ])
    args = SimpleNamespace(stride=1, ht=50, wd=100, timescale=1.0, decode_reduced=False)

    _, _, intrinsics, _ = next(image_stream(camera_dirs, calibration, args))

    expected = torch.tensor([[50.0, 50.0, 25.0, 25.0], [50.0, 50.0, 25.0, 25.0]])
    assert torch.equal(intrinsics[:, :4], expected)


def test_timestamp_from_filename() -> None:
    _, _, _, stamp = next(image_stream(DIRS, _calib(), _args()))
    assert stamp == pytest.approx(1552933152.129883)
