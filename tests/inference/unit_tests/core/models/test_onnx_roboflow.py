import io
import numpy as np
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Tuple, Any

from inference.core.models.roboflow import OnnxRoboflowInferenceModel


class TestOnnxRoboflowInferenceModelLoadImage:
    @pytest.fixture
    def mock_model(self):
        """Create a mock OnnxRoboflowInferenceModel instance."""
        with patch.object(OnnxRoboflowInferenceModel, "__init__", return_value=None):
            model = OnnxRoboflowInferenceModel("test_model_id")
            model.image_loader_threadpool = ThreadPoolExecutor(max_workers=2)
            model.preproc_image = MagicMock()
            return model

    def test_load_image_single_image(self, mock_model):
        """Test load_image with a single image."""
        # Setup
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(test_image)

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=False,
            disable_preproc_contrast=False,
            disable_preproc_grayscale=False,
            disable_preproc_static_crop=False,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]
        assert result.shape == (1, 3, 224, 224)

    def test_load_image_multiple_images(self, mock_model):
        """Test load_image with multiple images."""
        # Setup
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        
        # Create preprocessed images
        preprocessed_images = [
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        ]
        
        # Setup the mock to return the expected values
        mock_model.image_loader_threadpool.map = MagicMock(return_value=[
            (preprocessed_images[0], (100, 100)),
            (preprocessed_images[1], (200, 200)),
        ])
        
        # Execute
        with patch('numpy.concatenate', return_value=np.zeros((2, 3, 224, 224), dtype=np.float32)) as mock_concat:
            result, dims = mock_model.load_image(test_images)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert dims == ((100, 100), (200, 200))  # Note: dims is a tuple of tuples, not a list
        mock_concat.assert_called_once()

    def test_load_image_with_preprocessing_options(self, mock_model):
        """Test load_image with different preprocessing options."""
        # Setup
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=True,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=True,
        )

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=True,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=True,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]

    @patch("inference.core.models.roboflow.USE_PYTORCH_FOR_PREPROCESSING", True)
    @patch("inference.core.models.roboflow.torch", create=True)
    def test_load_image_multiple_images_with_pytorch(self, mock_torch, mock_model):
        """Test load_image with multiple images when PyTorch is used for preprocessing."""
        # Setup
        # Create a mock torch module
        import sys
        if 'torch' not in sys.modules:
            sys.modules['torch'] = mock_torch
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        
        # Create mock torch tensors for preprocessed images
        mock_tensor1 = MagicMock()
        mock_tensor2 = MagicMock()
        preprocessed_images = [mock_tensor1, mock_tensor2]
        
        # Setup the mock to return the expected values
        mock_model.image_loader_threadpool.map = MagicMock(return_value=[
            (preprocessed_images[0], (100, 100)),
            (preprocessed_images[1], (200, 200)),
        ])
        
        # Execute
        mock_torch.cat.return_value = MagicMock()
        result, dims = mock_model.load_image(test_images)
        
        # Assert
        mock_torch.cat.assert_called_once()
        assert dims == ((100, 100), (200, 200))

    def test_load_image_with_invalid_type(self, mock_model):
        """Test load_image with an invalid image type in the list."""
        # Setup
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        
        # Setup the mock to return invalid types
        class CustomObject:
            pass
        
        mock_model.image_loader_threadpool.map = MagicMock(return_value=[
            (CustomObject(), (100, 100)),
            (CustomObject(), (200, 200)),
        ])
        
        # Execute and Assert
        with pytest.raises(ValueError, match="unknown type"):
            mock_model.load_image(test_images)

    def test_load_image_empty_list(self, mock_model):
        """Test load_image with an empty list of images."""
        # Setup
        test_images = []
        
        # In the actual implementation, when an empty list is passed to zip(*[]),
        # it raises a ValueError because there are not enough values to unpack
        # Setup the mock to simulate this behavior
        def mock_map(*args, **kwargs):
            return []
        mock_model.image_loader_threadpool.map = MagicMock(side_effect=mock_map)
        
        # Execute and Assert
        with pytest.raises(ValueError, match="not enough values to unpack"):
            mock_model.load_image(test_images)
            
    def test_load_image_with_all_preprocessing_flags(self, mock_model):
        """Test load_image with all preprocessing flags set to True."""
        # Setup
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=True,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=True,
        )

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=True,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=True,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]
        
    def test_load_image_with_mixed_preprocessing_flags(self, mock_model):
        """Test load_image with mixed preprocessing flags."""
        # Setup
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=False,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=False,
        )

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=False,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=False,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]
        
    def test_load_image_with_different_image_sizes(self, mock_model):
        """Test load_image with images of different sizes."""
        # Setup
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),  # Small image
            np.zeros((500, 500, 3), dtype=np.uint8),  # Medium image
            np.zeros((1000, 800, 3), dtype=np.uint8),  # Large image
        ]
        
        # Create preprocessed images
        preprocessed_images = [
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        ]
        
        # Setup the mock to return the expected values
        mock_model.image_loader_threadpool.map = MagicMock(return_value=[
            (preprocessed_images[0], (100, 100)),
            (preprocessed_images[1], (500, 500)),
            (preprocessed_images[2], (1000, 800)),
        ])
        
        # Execute
        with patch('numpy.concatenate', return_value=np.zeros((3, 3, 224, 224), dtype=np.float32)) as mock_concat:
            result, dims = mock_model.load_image(test_images)
        
        # Assert
        assert isinstance(result, np.ndarray)
        assert dims == ((100, 100), (500, 500), (1000, 800))
        mock_concat.assert_called_once()
        
    def test_load_image_with_grayscale_image(self, mock_model):
        """Test load_image with a grayscale image."""
        # Setup
        test_image = np.zeros((100, 100, 1), dtype=np.uint8)  # Grayscale image
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),  # Converted to RGB
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(test_image)

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=False,
            disable_preproc_contrast=False,
            disable_preproc_grayscale=False,
            disable_preproc_static_crop=False,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]
        
    def test_load_image_with_rgba_image(self, mock_model):
        """Test load_image with an RGBA image."""
        # Setup
        test_image = np.zeros((100, 100, 4), dtype=np.uint8)  # RGBA image
        mock_model.preproc_image.return_value = (
            np.zeros((1, 3, 224, 224), dtype=np.float32),  # Converted to RGB
            (100, 100),
        )

        # Execute
        result, dims = mock_model.load_image(test_image)

        # Assert
        mock_model.preproc_image.assert_called_once_with(
            test_image,
            disable_preproc_auto_orient=False,
            disable_preproc_contrast=False,
            disable_preproc_grayscale=False,
            disable_preproc_static_crop=False,
        )
        assert isinstance(result, np.ndarray)
        assert dims == [(100, 100)]
        
    def test_load_image_with_different_output_shapes(self, mock_model):
        """Test load_image with preprocessed images of different shapes."""
        # Setup
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        
        # Create preprocessed images with different shapes
        preprocessed_images = [
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.zeros((1, 3, 640, 640), dtype=np.float32),  # Different size
        ]
        
        # Setup the mock to return the expected values
        mock_model.image_loader_threadpool.map = MagicMock(return_value=[
            (preprocessed_images[0], (100, 100)),
            (preprocessed_images[1], (200, 200)),
        ])
        
        # In this case, the actual implementation would fail because the shapes are different
        # and numpy.concatenate would raise an error
        with patch('numpy.concatenate', side_effect=ValueError("all the input arrays must have same shape")) as mock_concat:
            with pytest.raises(ValueError, match="all the input arrays must have same shape"):
                mock_model.load_image(test_images)
                
    def test_load_image_with_threadpool_exception(self, mock_model):
        """Test load_image when the threadpool raises an exception."""
        # Setup
        test_images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        
        # Setup the mock to raise an exception
        mock_model.image_loader_threadpool.map = MagicMock(side_effect=RuntimeError("Threadpool error"))
        
        # Execute and Assert
        with pytest.raises(RuntimeError, match="Threadpool error"):
            mock_model.load_image(test_images)
