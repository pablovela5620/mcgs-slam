"""Finite Gaussian mapping losses for masked metric depth."""

import pytest
import torch

from gaussian.utils.slam_utils import masked_inverse_depth_l1


def test_masked_inverse_depth_loss_never_divides_invalid_zero_depth() -> None:
    rendered_depth: torch.Tensor = torch.tensor([[[0.0, 2.0, 5.0]]])
    prior_depth: torch.Tensor = torch.tensor([[[0.0, 4.0, 0.0]]])
    valid: torch.Tensor = (rendered_depth > 0.01) & (prior_depth > 0.01)

    loss: torch.Tensor = masked_inverse_depth_l1(
        rendered_depth,
        prior_depth,
        valid,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.0 / 12.0)
