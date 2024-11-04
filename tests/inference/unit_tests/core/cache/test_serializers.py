import unittest
from unittest.mock import MagicMock
from inference.core.cache.serializers import (
    to_cachable_inference_item,
    build_condensed_response,
)
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse


class TestSerializers(unittest.TestCase):

    def setUp(self):
        self.mock_request = MagicMock(spec=InferenceRequest)
        self.mock_request.id = "test_id"
        self.mock_request.confidence = 0.9
        self.mock_request.dict.return_value = {
            "api_key": "test_key",
            "confidence": 0.9,
            "model_id": "test_model",
            "model_type": "test_type",
            "source": "test_source",
            "source_info": "test_info",
        }

        self.mock_response = MagicMock(spec=InferenceResponse)
        self.mock_response.predictions = [
            {"class_name": "cat", "confidence": 0.95},
            {"class_name": "dog", "confidence": 0.85},
        ]
        self.mock_response.time = "2023-10-01T12:00:00Z"

    def test_to_cachable_inference_item_no_tiny_cache(self):
        # Assuming TINY_CACHE is False
        result = to_cachable_inference_item(self.mock_request, self.mock_response)
        self.assertIn("inference_id", result)
        self.assertIn("request", result)
        self.assertIn("response", result)

    def test_to_cachable_inference_item_with_tiny_cache(self):
        # TINY_CACHE is True
        result = to_cachable_inference_item(self.mock_request, self.mock_response)
        self.assertIn("inference_id", result)
        self.assertIn("request", result)
        self.assertIn("response", result)

    def test_build_condensed_response_single(self):
        result = build_condensed_response(self.mock_response, self.mock_request)
        self.assertEqual(len(result), 1)
        self.assertIn("predictions", result[0])
        self.assertIn("time", result[0])

    def test_build_condensed_response_multiple(self):
        responses = [self.mock_response, self.mock_response]
        result = build_condensed_response(responses, self.mock_request)
        self.assertEqual(len(result), 2)

    def test_build_condensed_response_no_predictions(self):
        self.mock_response.predictions = None
        result = build_condensed_response(self.mock_response, self.mock_request)
        self.assertEqual(len(result), 0)
