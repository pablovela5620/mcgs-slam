"""Prior-validity mask: pixels MoGe marks invalid (zero normal) carry no depth into the mapper."""

import torch

from prior_mask import prior_valid, mask_invalid_depth


def test_prior_valid_is_true_only_where_a_unit_normal_exists() -> None:
    normals = torch.zeros(2, 3, 4, 5)
    normals[0, :, 1, 2] = torch.tensor([0.0, -1.0, 0.0])
    normals[1, :, 3, 4] = torch.tensor([0.6, 0.0, 0.8])

    valid = prior_valid(normals)

    assert valid.shape == (2, 4, 5) and valid.dtype == torch.bool
    assert valid.sum() == 2 and valid[0, 1, 2] and valid[1, 3, 4]


def test_mask_invalid_depth_zeroes_only_invalid_pixels() -> None:
    depths = torch.full((1, 4, 5), 7.5)
    normals = torch.zeros(1, 3, 4, 5)
    normals[0, 1, :2] = -1.0  # rows 0-1 valid

    masked = mask_invalid_depth(depths, normals)

    assert torch.equal(masked[0, :2], depths[0, :2])
    assert torch.all(masked[0, 2:] == 0)
    assert torch.all(depths == 7.5)  # input untouched
