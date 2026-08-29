"""Frontend scale-alignment contracts."""

import torch

from droid_frontend import disparity_scale


def test_unpopulated_sensor_disparity_uses_neutral_scale() -> None:
    estimated = torch.full((4, 4), 2.0)
    unpopulated_sensor = torch.zeros((4, 4))

    scale = disparity_scale(estimated, unpopulated_sensor)

    assert scale.item() == 1.0
