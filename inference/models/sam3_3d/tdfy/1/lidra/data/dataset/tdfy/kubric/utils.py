import torch


def normalize_points_to_unit_cube(
    points: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    """
    Normalize points to be in the range [-1, 1]
    """
    orig_scale = bounds[:, 1] - bounds[:, 0]
    orig_scale = orig_scale.max(dim=1).values
    points = points / orig_scale
    return points
