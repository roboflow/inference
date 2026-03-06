from collections import namedtuple
import json
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import os
from loguru import logger
from dataclasses import dataclass
from typing import Callable, List
from ..metadata_filter import custom_metadata_filter

from lidra.utils.decorators.counter import garbage_collect
from lidra.data.dataset.tdfy.trellis.pose_loader import identity_pose

from ..trellis.dataset import PerSubsetDataset
from .decode_mask import decode_mask
from ..img_and_mask_transforms import load_rgb


PerSubsetSampleID = namedtuple("PerSubsetSampleID", ["sha256", "image_fname"])


# New data structures for preference-based dataset
@dataclass
class PreferenceMeshCandidate:
    """Represents a single mesh candidate in a preference job"""

    sha256: str
    local_path: str
    num_voxels: int
    is_best: bool = False


@dataclass
class PreferenceJobSample:
    """Represents a complete preference job with multiple mesh candidates"""

    job_id: str
    image_path: str
    mask: str
    text_prompt: str
    quality: str
    timestamp: int
    duration: int
    candidates: List[PreferenceMeshCandidate]
    best_candidate: PreferenceMeshCandidate

    @classmethod
    def from_metadata_row(cls, row: pd.Series) -> "PreferenceJobSample":
        """Create PreferenceJobSample from a metadata DataFrame row"""
        # Parse JSON fields
        local_paths = json.loads(row["local_paths"].replace('""', '"'))
        sha256s = json.loads(row["sha256s"].replace('""', '"'))
        num_voxels = json.loads(row["num_voxels"])
        preference_best_path = row["preference_best_path"]

        # Create candidates
        candidates = []
        best_candidate = None

        for i, (path, sha256, voxels) in enumerate(
            zip(local_paths, sha256s, num_voxels)
        ):
            is_best = path == preference_best_path
            candidate = PreferenceMeshCandidate(
                sha256=sha256, local_path=path, num_voxels=voxels, is_best=is_best
            )
            candidates.append(candidate)
            if is_best:
                best_candidate = candidate

        if best_candidate is None:
            raise ValueError(f"No best candidate found for job {row['job_id']}")

        # Handle NaN values for text_prompt (pandas converts empty strings to NaN)
        text_prompt = row["text_prompt"]
        if pd.isna(text_prompt):
            text_prompt = ""

        return cls(
            job_id=str(row["job_id"]),
            image_path=row["image_path"],
            mask=row["mask"],
            text_prompt=text_prompt,
            quality=row["quality"],
            timestamp=int(row["timestamp"]),
            duration=int(row["duration"]),
            candidates=candidates,
            best_candidate=best_candidate,
        )


class PreferenceJobDataset(Dataset):
    """Dataset for preference-based 3D generation where each job contains multiple candidates"""

    VALID_SPLITS = {"train", "val"}

    def __init__(
        self,
        path: str,
        split: str,
        metadata_fname: str = "metadata_pref_0618_filtered_catA.csv",
        metadata_filter: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = custom_metadata_filter(None),
        latent_loader_dataset: PerSubsetDataset = None,
        return_mesh: bool = False,
        # New parameters for preference dataset
        return_job_metadata: bool = True,  # Whether to return job-level metadata
    ):
        self.path = path
        self.split = split
        self.latent_loader_dataset = latent_loader_dataset

        assert (
            split in PreferenceJobDataset.VALID_SPLITS
        ), f"split should be in {PreferenceJobDataset.VALID_SPLITS}"

        self.metadata_fname = metadata_fname
        self.metadata_filter = metadata_filter
        self.metadata = pd.read_csv(os.path.join(self.path, metadata_fname))
        self.metadata = metadata_filter(self.metadata)

        # Parse preference jobs from metadata
        self.preference_jobs = self._parse_preference_jobs()

        # Filter jobs that have at least 2 candidates (1 best + 1 other)
        self.preference_jobs = [
            job for job in self.preference_jobs if len(job.candidates) >= 2
        ]

        logger.info(
            f"Loaded {len(self.preference_jobs)} preference jobs with at least 2 candidates"
        )

        self.return_mesh = return_mesh
        self.return_job_metadata = return_job_metadata

    def _parse_preference_jobs(self) -> List[PreferenceJobSample]:
        """Parse metadata into preference job samples"""
        preference_jobs = []
        for _, row in self.metadata.iterrows():
            try:
                job_sample = PreferenceJobSample.from_metadata_row(row)
                preference_jobs.append(job_sample)
            except Exception as e:
                logger.error(f"Failed to parse job {row.get('job_id', 'unknown')}: {e}")
                continue
        return preference_jobs

    def __len__(self) -> int:
        return len(self.preference_jobs)

    def _sample_lose_candidate(
        self, job: PreferenceJobSample
    ) -> PreferenceMeshCandidate:
        """Sample a random non-best candidate from the job"""
        non_best_candidates = [
            candidate for candidate in job.candidates if not candidate.is_best
        ]
        return random.choice(non_best_candidates)

    def _load_mesh(self, candidate: PreferenceMeshCandidate):
        """Load mesh from candidate's local_path"""
        if self.latent_loader_dataset.mesh_loader is None or not self.return_mesh:
            raise ValueError("Mesh loader is not set or return_mesh is False")

        mesh_path = candidate.local_path
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file not found: {mesh_path}")

        trellis_mesh = self.latent_loader_dataset.mesh_loader(mesh_path)
        return trellis_mesh

    def _read_mask(self, job: PreferenceJobSample, rgb_image: torch.Tensor):
        """Read mask using decode_mask function from job's encoded mask data"""

        mask_data = job.mask

        # Get image dimensions
        if rgb_image.dim() == 3:  # (C, H, W)
            _, h, w = rgb_image.shape
        else:  # (H, W, C)
            h, w, _ = rgb_image.shape

        # Decode the mask
        decoded_mask = decode_mask(mask_data, w, h)

        # Convert to tensor and ensure binary
        mask_tensor = torch.from_numpy(decoded_mask).float()

        # Add channel dimension to match image format (C, H, W)
        mask_tensor = mask_tensor.unsqueeze(0)

        return self.latent_loader_dataset._ensure_mask_binary(mask_tensor)

    @garbage_collect()
    def __getitem__(self, index):
        job = self.preference_jobs[index]

        # Get best candidate (win)
        best_candidate = job.best_candidate
        if not best_candidate:
            raise ValueError(f"No best candidate found for job {job.job_id}")

        # Get random non-best candidate (lose)
        lose_candidate = self._sample_lose_candidate(job)

        # Compute both samples
        win_sample = self.compute_item(job, best_candidate, index, tag="win")
        lose_sample = self.compute_item(job, lose_candidate, index, tag="lose")

        if win_sample is None or lose_sample is None:
            raise ValueError(f"Failed to compute samples for job {job.job_id}")

        sample_uuid = job.job_id

        return sample_uuid, {**win_sample, **lose_sample}

    def compute_item(
        self,
        job: PreferenceJobSample,
        candidate: PreferenceMeshCandidate,
        index: int,
        tag: str = "",
    ):
        """Compute a single item from job and candidate"""
        uid = job.job_id  # Use job ID instead of candidate sha256

        # For preference dataset, we use the original image from the job
        img_path = job.image_path

        # Try to load the original image
        if os.path.exists(img_path):
            rgb_image = load_rgb(img_path)
        else:
            raise ValueError(f"Could not find image for job {uid} at {img_path}")

        # Load mask using decode_mask function
        rgb_image_mask = self._read_mask(job, rgb_image)
        if rgb_image_mask is None:
            raise ValueError(f"Failed to load mask for job {uid}")

        # How the images are processed (into crops, padded, etc)
        image_dict = self.latent_loader_dataset._process_image_and_mask_mess(
            rgb_image, rgb_image_mask
        )

        latent_dict = self.latent_loader_dataset._load_latent(candidate.sha256)

        # This is a workaround to be compatible with the existing code and config
        pose_dict = identity_pose(None)
        pointmap_dict = self.latent_loader_dataset._dummy_pointmap_moments()

        if self.return_mesh:
            mesh_dict = self._load_mesh(candidate)

        item = {}
        item.update(latent_dict)
        item.update(image_dict)
        item.update(pose_dict)
        item.update(pointmap_dict)

        if self.return_mesh:
            item.update(mesh_dict)

        # Add preference-specific information
        if self.return_job_metadata:
            item.update(
                {
                    "job_id": job.job_id,
                    "candidate_sha256": candidate.sha256,
                    "is_best_candidate": candidate.is_best,
                    "text_prompt": job.text_prompt,
                    "quality": job.quality,
                    "num_voxels": candidate.num_voxels,
                    "num_candidates": len(job.candidates),
                }
            )

        # for each key in item, add tag to the key
        if len(tag) > 0:
            for key in list(item.keys()):
                item[f"{key}_{tag}"] = item[key]
                del item[key]

        return item


class PreferenceJobDatasetForSFT(PreferenceJobDataset):

    @garbage_collect()
    def __getitem__(self, index):
        job = self.preference_jobs[index]

        # Get best candidate (win)
        best_candidate = job.best_candidate
        if not best_candidate:
            raise ValueError(f"No best candidate found for job {job.job_id}")

        sample_uuid = job.job_id

        # Compute sample data
        data = self.compute_item(job, best_candidate, index)

        return sample_uuid, data
