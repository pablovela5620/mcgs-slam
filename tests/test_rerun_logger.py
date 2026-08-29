"""Schema-neutral Rerun logger contracts."""

from types import SimpleNamespace

import pytest
import rerun as rr
import torch

from rerun_logger import RerunLogger


@pytest.mark.parametrize("scales", [[[0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1]] * 3])
def test_gaussian_logging_keeps_single_and_equal_scale_splats(
    scales: list[list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splat_count: int = len(scales)
    gaussians = SimpleNamespace(
        get_xyz=torch.zeros((splat_count, 3), dtype=torch.float32),
        get_scaling=torch.tensor(scales, dtype=torch.float32),
        get_rotation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * splat_count,
            dtype=torch.float32,
        ),
        get_opacity=torch.ones((splat_count, 1), dtype=torch.float32),
        get_features=torch.zeros((splat_count, 1, 3), dtype=torch.float32),
    )
    logger: RerunLogger = RerunLogger.__new__(RerunLogger)
    logger._kf_updates = 0
    logger.splat_every = 1
    logger.splat_cap = 75_000
    logger.max_splat_scale = 8.0
    logger.splat_scale_percentile = 99.7
    logger._scene_bbox = None
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(rr, "log", lambda path, value: logged.append((path, value)))

    logger.log_gaussians(gaussians)

    assert [path for path, _ in logged] == ["world/splats"]
