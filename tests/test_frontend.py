"""Frontend scale-alignment contracts."""

from types import SimpleNamespace

import torch

from droid_frontend import disparity_scale
from mcgs import Mcgs


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


def test_small_tracker_buffer_releases_to_a_bounded_window() -> None:
    released_windows: list[int] = []

    class StubFrontend:
        def __call__(self) -> torch.Tensor:
            return torch.empty(0, dtype=torch.long)

        def release_buffer(self, window: int) -> None:
            released_windows.append(window)

    slam: Mcgs = Mcgs.__new__(Mcgs)
    slam.filterx = SimpleNamespace(track=lambda *args: None)
    slam.frontend = StubFrontend()
    slam.video = SimpleNamespace(
        counter=SimpleNamespace(value=30),
        buffer=30,
        release_buffer=lambda window: released_windows.append(window),
    )

    slam.track(0, 0.0, None, None)

    assert released_windows == [15, 15]
