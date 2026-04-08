"""Integration tests for Falcon Perception model.

Tests end-to-end inference pipeline with a small randomly-initialized model.
These tests verify the full pipeline from image input to structured output
without requiring pretrained weights.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.engine import BatchEngine
from inference_models.models.falcon_perception.falcon_perception_torch import (
    FalconPerceptionTorch,
    _to_rgb_numpy_list,
)
from inference_models.models.falcon_perception.model import FalconPerceptionModel
from inference_models.models.falcon_perception.postprocessing import (
    result_to_detections,
    result_to_instance_detections,
)
from inference_models.models.falcon_perception.preprocessing import (
    ImageMetadata,
    preprocess_image,
    tokenize_prompts,
)


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return FalconPerceptionConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=256,
        patch_size=16,
        max_image_size=128,
        coord_bins=64,
        size_bins=64,
        seg_dim=32,
        anyup_levels=2,
        anyup_hidden_dim=32,
        max_generation_tokens=50,
        # Use low IDs for special tokens
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eoq_token_id=3,
        present_token_id=4,
        absent_token_id=5,
        coord_token_id=6,
        size_token_id=7,
        seg_token_id=8,
        image_token_id=9,
    )


@pytest.fixture
def small_model(small_config):
    """Create a small randomly-initialized model for testing."""
    model = FalconPerceptionModel(small_config)
    model.eval()
    return model


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_image():
    """Create a sample BGR image (as from OpenCV)."""
    return np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)


class TestToRGBNumpyList:
    def test_single_numpy_bgr(self):
        """Single BGR numpy array converted correctly."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = _to_rgb_numpy_list(img, "bgr")
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_single_numpy_rgb(self):
        """Single RGB numpy array passed through."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = _to_rgb_numpy_list(img, "rgb")
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_list_of_numpy(self):
        """List of numpy arrays."""
        imgs = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (48, 96, 3), dtype=np.uint8),
        ]
        result = _to_rgb_numpy_list(imgs, "bgr")
        assert len(result) == 2

    def test_torch_tensor_chw(self):
        """3D torch tensor (C, H, W) converted correctly."""
        tensor = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8).float()
        result = _to_rgb_numpy_list(tensor)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_torch_tensor_batch(self):
        """4D torch tensor (B, C, H, W) converted correctly."""
        tensor = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8).float()
        result = _to_rgb_numpy_list(tensor)
        assert len(result) == 2


class TestBatchEngineRun:
    def test_engine_produces_results(self, small_model, small_config, sample_image):
        """Engine should produce QueryResult for each prompt."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)

        # Create simple token sequence (BOS + some tokens + EOQ)
        text_token_ids = torch.tensor(
            [small_config.bos_token_id, 10, 11, small_config.eoq_token_id],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["test"],
            task="detection",
        )

        assert len(result.query_results) == 1
        qr = result.query_results[0]
        assert qr.prompt == "test"
        assert isinstance(qr.present, bool)
        assert 0.0 <= qr.presence_confidence <= 1.0

    def test_engine_multi_prompt(self, small_model, small_config, sample_image):
        """Engine should handle multiple prompts."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)

        text_token_ids = torch.tensor(
            [
                small_config.bos_token_id,
                10, 11, small_config.eoq_token_id,  # prompt 1
                12, 13, small_config.eoq_token_id,  # prompt 2
            ],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["cat", "dog"],
            task="detection",
        )

        assert len(result.query_results) == 2

    def test_segmentation_mode_stores_image_features(
        self, small_model, small_config, sample_image
    ):
        """Segmentation mode should store image features for mask computation."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)
        text_token_ids = torch.tensor(
            [small_config.bos_token_id, 10, small_config.eoq_token_id],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["test"],
            task="segmentation",
        )

        assert result.image_features is not None
        assert result.h_patches > 0
        assert result.w_patches > 0


class TestResultConversions:
    def test_detection_result_to_detections(self, small_model, small_config, sample_image):
        """Detection result should convert to valid Detections object."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)
        text_token_ids = torch.tensor(
            [small_config.bos_token_id, 10, small_config.eoq_token_id],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["cat"],
            task="detection",
        )

        detections = result_to_detections(
            result, metadata, small_config, ["cat"]
        )

        # Should have valid structure regardless of content
        assert hasattr(detections, "xyxy")
        assert hasattr(detections, "class_id")
        assert hasattr(detections, "confidence")
        assert detections.xyxy.shape[1] == 4 or detections.xyxy.shape[0] == 0
        assert detections.image_metadata["class_names"] == ["cat"]

    def test_segmentation_result_to_instance_detections(
        self, small_model, small_config, sample_image
    ):
        """Segmentation result should convert to valid InstanceDetections object."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)
        text_token_ids = torch.tensor(
            [small_config.bos_token_id, 10, small_config.eoq_token_id],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["cat"],
            task="segmentation",
        )

        instance_dets = result_to_instance_detections(
            result, metadata, small_model, small_config, ["cat"]
        )

        assert hasattr(instance_dets, "xyxy")
        assert hasattr(instance_dets, "mask")
        assert instance_dets.image_metadata["class_names"] == ["cat"]

    def test_detections_to_supervision_roundtrip(
        self, small_model, small_config, sample_image
    ):
        """Detections should convert to supervision format successfully."""
        engine = BatchEngine(
            model=small_model, config=small_config, device=torch.device("cpu")
        )

        pixel_values, metadata = preprocess_image(sample_image, small_config)
        text_token_ids = torch.tensor(
            [small_config.bos_token_id, 10, small_config.eoq_token_id],
            dtype=torch.long,
        )

        result = engine.run(
            pixel_values=pixel_values,
            text_token_ids=text_token_ids,
            image_metadata=metadata,
            prompts=["cat"],
            task="detection",
        )

        detections = result_to_detections(
            result, metadata, small_config, ["cat"]
        )
        sv_dets = detections.to_supervision()
        # Should not raise
        assert sv_dets is not None


class TestModelRegistration:
    def test_falcon_perception_in_registry(self):
        """Falcon Perception should be registered in the models registry."""
        from inference_models.models.auto_loaders.entities import BackendType
        from inference_models.models.auto_loaders.models_registry import (
            OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            REGISTERED_MODELS,
            model_implementation_exists,
        )

        assert model_implementation_exists(
            model_architecture="falcon-perception",
            task_type=OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            backend=BackendType.TORCH,
        )

    def test_falcon_perception_class_resolves(self):
        """The lazy class should resolve to FalconPerceptionTorch."""
        from inference_models.models.auto_loaders.entities import BackendType
        from inference_models.models.auto_loaders.models_registry import (
            OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            resolve_model_class,
        )

        cls = resolve_model_class(
            model_architecture="falcon-perception",
            task_type=OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            backend=BackendType.TORCH,
        )
        assert cls.__name__ == "FalconPerceptionTorch"

    def test_falcon_perception_not_available_as_onnx(self):
        """Falcon Perception should NOT be available as ONNX backend."""
        from inference_models.models.auto_loaders.entities import BackendType
        from inference_models.models.auto_loaders.models_registry import (
            OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            model_implementation_exists,
        )

        assert not model_implementation_exists(
            model_architecture="falcon-perception",
            task_type=OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            backend=BackendType.ONNX,
        )

    def test_falcon_perception_not_available_as_trt(self):
        """Falcon Perception should NOT be available as TRT backend."""
        from inference_models.models.auto_loaders.entities import BackendType
        from inference_models.models.auto_loaders.models_registry import (
            OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            model_implementation_exists,
        )

        assert not model_implementation_exists(
            model_architecture="falcon-perception",
            task_type=OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
            backend=BackendType.TRT,
        )
