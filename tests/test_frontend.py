"""Frontend scale-alignment contracts."""

import torch

from droid_frontend import disparity_scale


def test_unpopulated_sensor_disparity_uses_neutral_scale() -> None:
    estimated = torch.full((4, 4), 2.0)
    unpopulated_sensor = torch.zeros((4, 4))

    scale = disparity_scale(estimated, unpopulated_sensor)

    assert scale.item() == 1.0


def test_disparity_scale_uses_shared_positive_finite_support() -> None:
    estimated: torch.Tensor = torch.tensor(
        [[4.0, 8.0], [float("nan"), 40.0]],
        dtype=torch.float32,
    )
    half_invalid_sensor: torch.Tensor = torch.tensor(
        [[2.0, 0.0], [0.0, 20.0]],
        dtype=torch.float32,
    )

    scale: torch.Tensor = disparity_scale(estimated, half_invalid_sensor)

    assert scale.item() == 2.0
