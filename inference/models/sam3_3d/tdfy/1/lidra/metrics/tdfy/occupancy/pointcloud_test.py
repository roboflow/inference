import unittest
import torch

from lidra.metrics.tdfy.occupancy.pointcloud import (
    create_occupancy_volume,
    occupancy_grid_to_local_points,
)
from lidra.test.util import OverwriteTensorEquality


class TestPointCloudOccupancy(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_round_trip_conversion(self):
        n_voxels = 64
        test_volume = (
            torch.zeros((1, 1, n_voxels, n_voxels, n_voxels), device=self.device) - 1.0
        )
        test_volume[0, 0, 20:40, 20:40, 20:40] = 1.0

        test_points = occupancy_grid_to_local_points(test_volume, threshold=0.0)

        self.assertEqual(test_points.shape[0], 20**3, "Should have 20^3 points")
        self.assertEqual(test_points.shape[1], 3, "Points should be 3-dimensional")

        reconstructed_volume = create_occupancy_volume(test_points, n_voxels=n_voxels)

        with OverwriteTensorEquality(torch, check_shape=True):
            self.assertEqual(test_volume.squeeze() > 0, reconstructed_volume)
