import unittest
import os
import json
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import tempfile
import shutil

from lidra.test.util import run_unittest, run_only_if_path_exists
from lidra.data.dataset.tdfy.preference.dataset import (
    PreferenceMeshCandidate,
    PreferenceJobSample,
    PreferenceJobDataset,
    PreferenceJobDatasetForSFT,
    PerSubsetSampleID,
)
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor
from lidra.data.dataset.tdfy.metadata_filter import custom_metadata_filter
from lidra.data.dataset.tdfy.trellis.pose_loader import identity_pose


class MockLatentLoaderDataset:
    """Mock class for latent_loader_dataset"""

    def __init__(self):
        self.mesh_loader = Mock(return_value={"mesh": torch.rand(100, 3)})

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        return {
            "image": rgb_image,
            "mask": rgb_image_mask,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

    def _load_latent(self, sha256):
        return {
            "mean": torch.randn(4096, 8),
            "logvar": torch.randn(4096, 8),
        }

    def _ensure_mask_binary(self, mask_tensor):
        return (mask_tensor > 0.5).float()

    def _dummy_pointmap_moments(self):
        return {
            "pointmap_scale": torch.tensor(1.0),
            "pointmap_shift": torch.zeros(1, 3),
        }


class TestPreferenceMeshCandidate(unittest.TestCase):
    def test_init(self):
        candidate = PreferenceMeshCandidate(
            sha256="test_sha", local_path="/path/to/mesh", num_voxels=1000, is_best=True
        )
        self.assertEqual(candidate.sha256, "test_sha")
        self.assertEqual(candidate.local_path, "/path/to/mesh")
        self.assertEqual(candidate.num_voxels, 1000)
        self.assertTrue(candidate.is_best)

    def test_default_is_best(self):
        candidate = PreferenceMeshCandidate(
            sha256="test_sha", local_path="/path/to/mesh", num_voxels=1000
        )
        self.assertFalse(candidate.is_best)


class TestPreferenceJobSample(unittest.TestCase):
    def setUp(self):
        self.sample_row = pd.Series(
            {
                "job_id": "12345",
                "image_path": "/path/to/image.jpg",
                "mask": "encoded_mask_data",
                "text_prompt": "A 3D object",
                "quality": "high",
                "timestamp": 1234567890,
                "duration": 120,
                "local_paths": '["path1", "path2", "path3"]',
                "sha256s": '["sha1", "sha2", "sha3"]',
                "num_voxels": "[1000, 2000, 1500]",
                "preference_best_path": "path2",
            }
        )

    def test_from_metadata_row(self):
        job_sample = PreferenceJobSample.from_metadata_row(self.sample_row)

        self.assertEqual(job_sample.job_id, "12345")
        self.assertEqual(job_sample.image_path, "/path/to/image.jpg")
        self.assertEqual(job_sample.mask, "encoded_mask_data")
        self.assertEqual(job_sample.text_prompt, "A 3D object")
        self.assertEqual(job_sample.quality, "high")
        self.assertEqual(job_sample.timestamp, 1234567890)
        self.assertEqual(job_sample.duration, 120)
        self.assertEqual(len(job_sample.candidates), 3)

        # Check best candidate
        self.assertIsNotNone(job_sample.best_candidate)
        self.assertEqual(job_sample.best_candidate.local_path, "path2")
        self.assertEqual(job_sample.best_candidate.sha256, "sha2")
        self.assertEqual(job_sample.best_candidate.num_voxels, 2000)
        self.assertTrue(job_sample.best_candidate.is_best)

        # Check other candidates
        non_best = [c for c in job_sample.candidates if not c.is_best]
        self.assertEqual(len(non_best), 2)

    def test_from_metadata_row_no_best(self):
        self.sample_row["preference_best_path"] = "non_existent_path"
        with self.assertRaises(ValueError):
            PreferenceJobSample.from_metadata_row(self.sample_row)

    def test_from_metadata_row_json_parsing(self):
        # Test with double quotes in JSON
        self.sample_row["local_paths"] = '["path1"", ""path2"]'
        self.sample_row["sha256s"] = '["sha1"", ""sha2"]'
        self.sample_row["preference_best_path"] = "path1"

        job_sample = PreferenceJobSample.from_metadata_row(self.sample_row)
        self.assertEqual(len(job_sample.candidates), 2)


class TestPreferenceJobDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.test_dir, "metadata_pref_0606.csv")

        # Create test metadata
        self.test_metadata = pd.DataFrame(
            {
                "job_id": ["job1", "job2", "job3"],
                "image_path": [
                    "/datasets01/segment_everything/032322_resized/1118169308.jpg",
                    "/datasets01/segment_everything/032322_resized/1118169308.jpg",
                    "/datasets01/segment_everything/032322_resized/1118169308.jpg",
                ],
                "mask": ["mask1", "mask2", "mask3"],
                "text_prompt": ["prompt1", "prompt2", "prompt3"],
                "quality": ["high", "medium", "low"],
                "timestamp": [1000, 2000, 3000],
                "duration": [100, 200, 300],
                "local_paths": ['["p1", "p2"]', '["p3", "p4", "p5"]', '["p6"]'],
                "sha256s": ['["s1", "s2"]', '["s3", "s4", "s5"]', '["s6"]'],
                "num_voxels": ["[100, 200]", "[300, 400, 500]", "[600]"],
                "preference_best_path": ["p2", "p4", "p6"],
            }
        )
        self.test_metadata.to_csv(self.metadata_path, index=False)

        # Mock latent loader dataset
        self.mock_latent_loader = MockLatentLoaderDataset()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
            return_mesh=False,
            return_job_metadata=True,
        )

        self.assertEqual(len(dataset), 2)  # Only jobs with >=2 candidates
        self.assertEqual(dataset.split, "train")
        self.assertTrue(dataset.return_job_metadata)
        self.assertFalse(dataset.return_mesh)

    def test_invalid_split(self):
        with self.assertRaises(AssertionError):
            PreferenceJobDataset(
                path=self.test_dir,
                split="test",  # Invalid split
                metadata_fname="metadata_pref_0606.csv",
                latent_loader_dataset=self.mock_latent_loader,
            )

    def test_parse_preference_jobs(self):
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        # Should have filtered out job3 (only 1 candidate)
        self.assertEqual(len(dataset.preference_jobs), 2)
        self.assertEqual(dataset.preference_jobs[0].job_id, "job1")
        self.assertEqual(dataset.preference_jobs[1].job_id, "job2")

    def test_sample_lose_candidate(self):
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        job = dataset.preference_jobs[0]
        lose_candidate = dataset._sample_lose_candidate(job)

        self.assertFalse(lose_candidate.is_best)
        self.assertIn(lose_candidate, job.candidates)

    @patch("os.path.exists")
    @patch("matplotlib.pyplot.imread")
    def test_load_img(self, mock_plt_imread, mock_exists):
        mock_exists.return_value = True
        # Mock plt.imread to return a float32 image (0-1 range) as matplotlib typically does
        mock_img = np.random.rand(100, 100, 3).astype(np.float32)
        mock_plt_imread.return_value = mock_img

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        # Test load_rgb function directly since the dataset uses it
        from lidra.data.dataset.tdfy.img_and_mask_transforms import load_rgb

        img_tensor = load_rgb("/fake/path.jpg")

        self.assertEqual(img_tensor.shape, (3, 100, 100))
        self.assertTrue(torch.is_tensor(img_tensor))
        self.assertTrue((img_tensor >= 0).all() and (img_tensor <= 1).all())

    @patch("lidra.data.dataset.tdfy.preference.dataset.decode_mask")
    def test_read_mask(self, mock_decode_mask):
        mock_mask = np.random.rand(100, 100) > 0.5
        mock_decode_mask.return_value = mock_mask

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        job = dataset.preference_jobs[0]
        rgb_image = torch.rand(3, 100, 100)

        mask_tensor = dataset._read_mask(job, rgb_image)

        self.assertEqual(mask_tensor.shape, (1, 100, 100))
        self.assertTrue(torch.is_tensor(mask_tensor))
        mock_decode_mask.assert_called_once_with(job.mask, 100, 100)

    @patch.object(PreferenceJobDataset, "compute_item")
    def test_getitem(self, mock_compute_item):
        # Setup mock returns - compute_item returns items with tags already applied
        win_item = {"data_win": "win", "num_voxels_win": 200}
        lose_item = {"data_lose": "lose", "num_voxels_lose": 100}
        mock_compute_item.side_effect = [win_item, lose_item]

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        sample_uuid, result = dataset[0]

        self.assertEqual(result["data_win"], "win")
        self.assertEqual(result["num_voxels_win"], 200)
        self.assertEqual(result["data_lose"], "lose")
        self.assertEqual(result["num_voxels_lose"], 100)

        # Check that compute_item was called twice with correct tags
        self.assertEqual(mock_compute_item.call_count, 2)

    @patch("os.path.exists")
    @patch("matplotlib.pyplot.imread")
    @patch("lidra.data.dataset.tdfy.preference.dataset.decode_mask")
    def test_compute_item(self, mock_decode_mask, mock_plt_imread, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        # Mock plt.imread to return a float32 image (0-1 range) as matplotlib typically does
        mock_img = np.random.rand(100, 100, 3).astype(np.float32)
        mock_plt_imread.return_value = mock_img
        mock_decode_mask.return_value = np.ones((100, 100))

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
            return_mesh=True,
            return_job_metadata=True,
        )

        job = dataset.preference_jobs[0]
        candidate = job.candidates[0]

        with patch.object(
            dataset, "_load_mesh", return_value={"mesh": torch.rand(10, 3)}
        ):
            item = dataset.compute_item(job, candidate, 0, tag="test")

        # Check that all expected keys are present with tag
        expected_keys = [
            "mean_test",
            "logvar_test",
            "image_test",
            "mask_test",
            "rgb_image_test",
            "rgb_image_mask_test",
            "mesh_test",
        ]
        for key in expected_keys:
            self.assertIn(key, item)

        # Check job metadata
        self.assertEqual(item["job_id_test"], "job1")
        self.assertEqual(item["text_prompt_test"], "prompt1")
        self.assertEqual(item["quality_test"], "high")

    @patch("lidra.data.dataset.tdfy.preference.dataset.load_rgb")
    @patch("os.path.exists")
    def test_compute_item_no_tag(self, mock_exists, mock_load_rgb):
        mock_exists.return_value = True
        mock_load_rgb.return_value = torch.rand(3, 100, 100)

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
            return_mesh=False,
            return_job_metadata=False,
        )

        job = dataset.preference_jobs[0]
        candidate = job.candidates[0]

        with patch.object(dataset, "_read_mask", return_value=torch.ones(1, 100, 100)):
            item = dataset.compute_item(job, candidate, 0)

        # Check that keys don't have tags
        self.assertIn("mean", item)
        self.assertIn("logvar", item)
        self.assertNotIn("job_id", item)  # return_job_metadata=False

    def test_load_mesh_error_no_mesh_loader(self):
        """Test that _load_mesh raises error when mesh_loader is None"""
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
            return_mesh=False,  # This should cause error
        )

        candidate = PreferenceMeshCandidate(
            sha256="test", local_path="/fake/path", num_voxels=100
        )

        with self.assertRaises(ValueError):
            dataset._load_mesh(candidate)

    @patch("os.path.exists")
    def test_load_mesh_error_file_not_found(self, mock_exists):
        """Test that _load_mesh raises error when mesh file doesn't exist"""
        mock_exists.return_value = False

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
            return_mesh=True,
        )

        candidate = PreferenceMeshCandidate(
            sha256="test", local_path="/fake/path", num_voxels=100
        )

        with self.assertRaises(ValueError):
            dataset._load_mesh(candidate)

    @patch("matplotlib.pyplot.imread")
    def test_load_img_error_imread_fails(self, mock_plt_imread):
        """Test that load_rgb raises error when plt.imread fails"""
        # Mock plt.imread to raise an exception (file not found, permission error, etc.)
        mock_plt_imread.side_effect = FileNotFoundError("No such file or directory")

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        from lidra.data.dataset.tdfy.img_and_mask_transforms import load_rgb

        with self.assertRaises(FileNotFoundError):
            load_rgb("/fake/path.jpg")

    def test_decode_mask_real_data(self):
        """Test decoding a real mask string"""
        from lidra.data.dataset.tdfy.preference.decode_mask import decode_mask

        # Real mask string from user
        real_mask_string = "JoAw2gFgjAHAenWBJAbAKXQVgLIGZsAs2ATAHJQDyADBcRVObafnZVVZQ7o5XTZ+3odyg3jyFt2UtrWqypzcgBkCaAOwiGUYZybbJ26u0McjhkwepttNY1funpt689umLo1-zOCOUn242Vvy+foGhQcYRogGxdnYW9GZJfkm6CSnmaSnuQt4C+YJkQbpkxcJlJExltCSkZLhK3I34BEqYSAAqAJYAagCqUEA"

        # Test with different dimensions
        width, height = 2250, 1500

        try:
            decoded_mask = decode_mask(real_mask_string, width, height)

            # Basic sanity checks
            self.assertIsInstance(decoded_mask, np.ndarray)
            self.assertEqual(decoded_mask.shape, (height, width))
            self.assertEqual(decoded_mask.dtype, np.uint8)

            # Check that mask contains binary values (0 or 1)
            unique_values = np.unique(decoded_mask)
            self.assertTrue(all(val in [0, 1] for val in unique_values))

            # Check that mask is not all zeros or all ones (should have some content)
            self.assertGreater(np.sum(decoded_mask), 0)
            self.assertLess(np.sum(decoded_mask), decoded_mask.size)

            print(f"Successfully decoded mask with shape {decoded_mask.shape}")
            print(
                f"Mask coverage: {np.sum(decoded_mask) / decoded_mask.size * 100:.2f}%"
            )

        except Exception as e:
            self.fail(f"Failed to decode real mask string: {e}")


class TestPreferenceJobDatasetForSFT(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.test_dir, "metadata_pref_0606.csv")

        # Create test metadata
        self.test_metadata = pd.DataFrame(
            {
                "job_id": ["job1", "job2"],
                "image_path": ["/img1.jpg", "/img2.jpg"],
                "mask": ["mask1", "mask2"],
                "text_prompt": ["prompt1", "prompt2"],
                "quality": ["high", "medium"],
                "timestamp": [1000, 2000],
                "duration": [100, 200],
                "local_paths": ['["p1", "p2"]', '["p3", "p4"]'],
                "sha256s": ['["s1", "s2"]', '["s3", "s4"]'],
                "num_voxels": ["[100, 200]", "[300, 400]"],
                "preference_best_path": ["p2", "p4"],
            }
        )
        self.test_metadata.to_csv(self.metadata_path, index=False)

        # Mock latent loader dataset
        self.mock_latent_loader = MockLatentLoaderDataset()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    @patch.object(PreferenceJobDatasetForSFT, "compute_item")
    def test_getitem_sft(self, mock_compute_item):
        mock_compute_item.return_value = {"data": "best", "num_voxels": 200}

        dataset = PreferenceJobDatasetForSFT(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=self.mock_latent_loader,
        )

        sample_uuid, result = dataset[0]

        self.assertEqual(result["data"], "best")
        self.assertEqual(result["num_voxels"], 200)

        # Check that compute_item was called once (only for best candidate)
        mock_compute_item.assert_called_once()

        # Check that best candidate was used
        call_args = mock_compute_item.call_args[0]
        self.assertTrue(call_args[1].is_best)  # Second arg is candidate


class TestIntegration(unittest.TestCase):
    """Integration tests with real data structures"""

    def setUp(self):
        # Create a more realistic test environment
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.test_dir, "metadata_pref_0606.csv")

        # Create metadata with edge cases
        self.test_metadata = pd.DataFrame(
            {
                "job_id": ["job1", "job2", "job3", "job4", "job5"],
                "image_path": [
                    "/img1.jpg",
                    "/img2.jpg",
                    "/img3.jpg",
                    "/img4.jpg",
                    "/img5.jpg",
                ],
                "mask": ["mask1", "mask2", "mask3", "mask4", "mask5"],
                "text_prompt": [
                    "A very long prompt " * 10,
                    "Short",
                    "",
                    "Normal prompt",
                    "",
                ],
                "quality": ["high", "medium", "low", "high", "medium"],
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "duration": [100, 200, 300, 400, 500],
                "local_paths": [
                    '["p1", "p2", "p3", "p4", "p5"]',  # Many candidates
                    '["p6", "p7"]',  # Minimum candidates
                    '["p8"]',  # Single candidate (should be filtered)
                    '["p9", "p10"]',  # Normal case
                    '["p11", "p12"]',  # Empty prompt with 2 candidates
                ],
                "sha256s": [
                    '["s1", "s2", "s3", "s4", "s5"]',
                    '["s6", "s7"]',
                    '["s8"]',
                    '["s9", "s10"]',
                    '["s11", "s12"]',
                ],
                "num_voxels": [
                    "[100, 200, 300, 400, 500]",
                    "[600, 700]",
                    "[800]",
                    "[900, 1000]",
                    "[1100, 1200]",
                ],
                "preference_best_path": ["p3", "p7", "p8", "p9", "p12"],
            }
        )
        # Save CSV with proper handling of empty strings to avoid NaN conversion
        self.test_metadata.to_csv(self.metadata_path, index=False, na_rep="")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_filtering(self):
        """Test that single-candidate jobs are filtered out"""
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=MockLatentLoaderDataset(),
        )

        # Should have 4 jobs (job3 filtered out)
        self.assertEqual(len(dataset), 4)

        # Check job IDs
        job_ids = [job.job_id for job in dataset.preference_jobs]
        self.assertNotIn("job3", job_ids)

    def test_multiple_candidates_handling(self):
        """Test handling of jobs with many candidates"""
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=MockLatentLoaderDataset(),
        )

        # Find job with 5 candidates
        job_with_many = next(j for j in dataset.preference_jobs if j.job_id == "job1")
        self.assertEqual(len(job_with_many.candidates), 5)

        # Test that we can sample different lose candidates
        lose_candidates = set()
        for _ in range(10):
            lose = dataset._sample_lose_candidate(job_with_many)
            lose_candidates.add(lose.sha256)

        # Should have sampled multiple different candidates
        self.assertGreater(len(lose_candidates), 1)

    def test_edge_case_prompts(self):
        """Test handling of edge case text prompts"""
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=MockLatentLoaderDataset(),
        )

        # Check long prompt
        long_prompt_job = next(j for j in dataset.preference_jobs if j.job_id == "job1")
        self.assertGreater(len(long_prompt_job.text_prompt), 100)

        # Check short prompt (job2 has 'Short')
        short_prompt_job = next(
            j for j in dataset.preference_jobs if j.job_id == "job2"
        )
        self.assertEqual(short_prompt_job.text_prompt, "Short")

        # Check empty prompt (job5 has empty string and 2 candidates)
        empty_prompt_job = next(
            j for j in dataset.preference_jobs if j.job_id == "job5"
        )
        self.assertEqual(empty_prompt_job.text_prompt, "")

    def test_metadata_filter_integration(self):
        """Test that metadata filters work correctly"""

        # Custom filter that removes high quality jobs
        def quality_filter(df):
            return df[df["quality"] != "high"]

        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            metadata_filter=quality_filter,
            latent_loader_dataset=MockLatentLoaderDataset(),
        )

        # Should only have medium/low quality jobs (and filtered for >=2 candidates)
        job_qualities = [job.quality for job in dataset.preference_jobs]
        self.assertNotIn("high", job_qualities)

    def test_preference_job_sample_best_candidate_identification(self):
        """Test that best candidate is correctly identified"""
        dataset = PreferenceJobDataset(
            path=self.test_dir,
            split="train",
            metadata_fname="metadata_pref_0606.csv",
            latent_loader_dataset=MockLatentLoaderDataset(),
        )

        for job in dataset.preference_jobs:
            # Check that exactly one candidate is marked as best
            best_candidates = [c for c in job.candidates if c.is_best]
            self.assertEqual(len(best_candidates), 1)

            # Check that the best candidate matches the job's best_candidate
            self.assertEqual(best_candidates[0], job.best_candidate)

            # Check that non-best candidates exist
            non_best = [c for c in job.candidates if not c.is_best]
            self.assertGreater(len(non_best), 0)


# Test with real paths if available
class TestDecodeMask(unittest.TestCase):
    """Dedicated tests for the decode_mask functionality"""

    def setUp(self):
        self.real_mask_string = "JoAw2gFgjAHAenWBJAbAKXQVgLIGZsAs2ATAHJQDyADBcRVObafnZVVZQ7o5XTZ+3odyg3jyFt2UtrWqypzcgBkCaAOwiGUYZybbJ26u0McjhkwepttNY1funpt689umLo1-zOCOUn242Vvy+foGhQcYRogGxdnYW9GZJfkm6CSnmaSnuQt4C+YJkQbpkxcJlJExltCSkZLhK3I34BEqYSAAqAJYAagCqUEA"

    def test_decode_mask_basic_functionality(self):
        """Test basic decode_mask functionality"""
        from lidra.data.dataset.tdfy.preference.decode_mask import decode_mask

        decoded_mask = decode_mask(self.real_mask_string, 2250, 1500)

        # Shape and type checks
        self.assertEqual(decoded_mask.shape, (1500, 2250))
        self.assertEqual(decoded_mask.dtype, np.uint8)

        # Binary mask check
        unique_values = np.unique(decoded_mask)
        self.assertTrue(all(val in [0, 1] for val in unique_values))

        # Content check
        self.assertGreater(np.sum(decoded_mask), 0)
        self.assertLess(np.sum(decoded_mask), decoded_mask.size)

    def test_decode_mask_statistical_properties(self):
        """Test statistical properties of the decoded mask"""
        from lidra.data.dataset.tdfy.preference.decode_mask import decode_mask

        decoded_mask = decode_mask(self.real_mask_string, 2250, 1500)

        # Calculate statistics
        total_pixels = decoded_mask.size
        foreground_pixels = np.sum(decoded_mask)
        background_pixels = total_pixels - foreground_pixels
        coverage_ratio = foreground_pixels / total_pixels

        # Basic sanity checks
        self.assertGreater(
            foreground_pixels, 0, "Mask should have some foreground pixels"
        )
        self.assertGreater(
            background_pixels, 0, "Mask should have some background pixels"
        )
        self.assertGreater(coverage_ratio, 0.001, "Coverage should be at least 0.1%")
        self.assertLess(coverage_ratio, 0.999, "Coverage should be at most 99.9%")

        print(f"Mask statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Foreground pixels: {foreground_pixels:,}")
        print(f"  Background pixels: {background_pixels:,}")
        print(f"  Coverage ratio: {coverage_ratio:.4f} ({coverage_ratio*100:.2f}%)")

    def test_decode_mask_compression_functions(self):
        """Test the underlying compression functions"""
        from lidra.data.dataset.tdfy.preference.decode_mask import (
            decompress_from_encoded_uri,
            get_base_value,
            KEYSTRURISAFE,
        )

        # Test base value function
        for i, char in enumerate(KEYSTRURISAFE[:10]):  # Test first 10 characters
            self.assertEqual(get_base_value(KEYSTRURISAFE, char), i)

        # Test decompression
        decompressed = decompress_from_encoded_uri(self.real_mask_string)
        self.assertIsInstance(decompressed, str)
        self.assertGreater(len(decompressed), 0)

        print(f"Decompressed string length: {len(decompressed)}")
        print(f"First 100 chars: {decompressed[:100]}")


class TestRealDataPaths(unittest.TestCase):
    """Tests that run only if real data paths are available"""

    @run_only_if_path_exists("/fsx-3dfy-v2/shared/datasets/task2_generated_mesh")
    def test_real_metadata_loading(self):
        """Test loading with real metadata file"""
        from lidra.data.dataset.tdfy.trellis.latent_loader import load_structure_latents
        from lidra.data.dataset.tdfy.trellis.mesh_loader import load_trellis_mesh
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        # Load preprocessor config
        config_path = "/fsx-3dfy-v2/ryanxiangli/lidra/etc/lidra/data/dataset/tdfy/trellis500k/preprocessor/train_default.yaml"
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
            preprocessor = instantiate(config)

            latent_loader_dataset = PerSubsetDataset(
                path="/fsx-3dfy-v2/ryanxiangli/lidra/lidra/data/dataset/tdfy/preference",
                split="train",
                preprocessor=preprocessor,
                metadata_fname="/fsx-3dfy-v2/shared/datasets/task2_generated_mesh/metadata_dummy.csv",
                latent_dir="/fsx-3dfy-v2/shared/datasets/task2_generated_mesh/ss_latents/ss_enc_conv3d_16l8_fp16",
                latent_loader=load_structure_latents,
                mesh_loader=load_trellis_mesh,
            )

            dataset = PreferenceJobDataset(
                path="/fsx-3dfy-v2/ryanxiangli/lidra/lidra/data/dataset/tdfy/preference",
                split="train",
                metadata_fname="/fsx-3dfy-v2/shared/datasets/task2_generated_mesh/metadata_pref_0606.csv",
                latent_loader_dataset=latent_loader_dataset,
                return_mesh=False,
                return_job_metadata=True,
            )

            # Basic sanity checks
            self.assertGreater(len(dataset), 0)

            # Test that we can get first item without errors
            try:
                sample_uuid, first_item = dataset[0]
                self.assertIsInstance(first_item, dict)

                # Check for expected keys
                expected_suffixes = ["_win", "_lose"]
                for suffix in expected_suffixes:
                    self.assertTrue(
                        any(key.endswith(suffix) for key in first_item.keys())
                    )

            except Exception as e:
                # If we can't load the first item, that's still valuable info
                self.fail(f"Could not load first dataset item: {e}")


if __name__ == "__main__":
    run_unittest(TestPreferenceMeshCandidate)
    run_unittest(TestPreferenceJobSample)
    run_unittest(TestPreferenceJobDataset)
    run_unittest(TestPreferenceJobDatasetForSFT)
    run_unittest(TestIntegration)
    run_unittest(TestDecodeMask)
    run_unittest(TestRealDataPaths)
