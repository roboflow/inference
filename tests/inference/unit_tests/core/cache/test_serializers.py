import unittest
import os
from unittest.mock import MagicMock
from inference.core.cache.serializers import (
    to_cachable_inference_item,
    build_condensed_response,
)

from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)


class TestSerializersClassification(unittest.TestCase):

    def setUp(self):
        self.mock_classification_request = MagicMock(
            spec=ClassificationInferenceRequest
        )
        self.mock_classification_request.id = "test_id"
        self.mock_classification_request.confidence = 0.9
        self.mock_classification_request.dict.return_value = {
            "api_key": "test_key",
            "confidence": 0.9,
            "model_id": "test_model",
            "model_type": "test_type",
            "source": "test_source",
            "source_info": "test_info",
        }

        self.mock_classification_response = MagicMock(
            spec=ClassificationInferenceResponse
        )
        self.mock_classification_response.predictions = ["cat", "dog"]
        self.mock_classification_response.time = "2023-10-01T12:00:00Z"

    def test_to_cachable_inference_item_no_tiny_cache(self):
        os.environ["TINY_CACHE"] = "False"
        result = to_cachable_inference_item(
            self.mock_classification_request, self.mock_classification_response
        )
        self.assertEqual(result["inference_id"], self.mock_classification_request.id)
        self.assertEqual(
            result["request"]["api_key"],
            self.mock_classification_request.dict.return_value["api_key"],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["class"],
            self.mock_classification_response.predictions[0],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["confidence"],
            self.mock_classification_request.confidence,
        )

    def test_to_cachable_inference_item_with_tiny_cache(self):
        os.environ["TINY_CACHE"] = "True"
        result = to_cachable_inference_item(
            self.mock_classification_request, self.mock_classification_response
        )
        self.assertEqual(result["inference_id"], self.mock_classification_request.id)
        self.assertEqual(
            result["request"]["api_key"],
            self.mock_classification_request.dict.return_value["api_key"],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["class"],
            self.mock_classification_response.predictions[0],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["confidence"],
            self.mock_classification_request.confidence,
        )

    def test_build_condensed_response_single(self):
        result = build_condensed_response(
            self.mock_classification_response, self.mock_classification_request
        )
        self.assertEqual(len(result), 1)
        self.assertIn("predictions", result[0])
        self.assertIn("time", result[0])

    def test_build_condensed_response_multiple(self):
        responses = [
            self.mock_classification_response,
            self.mock_classification_response,
        ]
        result = build_condensed_response(responses, self.mock_classification_request)
        self.assertEqual(len(result), 2)

    def test_build_condensed_response_no_predictions(self):
        self.mock_classification_response.predictions = None
        result = build_condensed_response(
            self.mock_classification_response, self.mock_classification_request
        )
        self.assertEqual(len(result), 0)


class TestSerializersObjectDetection(unittest.TestCase):

    def setUp(self):
        self.mock_object_detection_request = MagicMock(
            spec=ObjectDetectionInferenceRequest
        )
        self.mock_object_detection_request.id = "test_id"
        self.mock_object_detection_request.confidence = 0.85
        self.mock_object_detection_request.dict.return_value = {
            "api_key": "test_key",
            "confidence": 0.85,
            "model_id": "sharks",
            "model_type": "object_detection",
        }

        self.mock_object_detection_response = MagicMock(
            spec=ObjectDetectionInferenceResponse
        )
        self.mock_object_detection_response.predictions = [
            ObjectDetectionPrediction(
                **{
                    "class_name": "tiger-shark",
                    "confidence": 0.95,
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0,
                    "class_confidence": None,
                    "class_id": 1,
                    "class": "tiger-shark",
                }
            ),
            ObjectDetectionPrediction(
                **{
                    "class_name": "hammerhead",
                    "confidence": 0.95,
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0,
                    "class_confidence": None,
                    "class_id": 2,
                    "class": "hammerhead",
                }
            ),
            ObjectDetectionPrediction(
                **{
                    "class_name": "white-shark",
                    "confidence": 0.95,
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0,
                    "class_confidence": None,
                    "class_id": 3,
                    "class": "white-shark",
                }
            ),
        ]
        self.mock_object_detection_response.time = "2023-10-01T12:00:00Z"

    def test_to_cachable_inference_item_no_tiny_cache(self):
        os.environ["TINY_CACHE"] = "False"
        result = to_cachable_inference_item(
            self.mock_object_detection_request, self.mock_object_detection_response
        )
        self.assertEqual(result["inference_id"], self.mock_object_detection_request.id)
        self.assertEqual(
            result["request"]["api_key"],
            self.mock_object_detection_request.dict.return_value["api_key"],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["class"],
            self.mock_object_detection_response.predictions[0].class_name,
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["confidence"],
            self.mock_object_detection_response.predictions[0].confidence,
        )

    def test_to_cachable_inference_item_with_tiny_cache(self):
        os.environ["TINY_CACHE"] = "True"
        result = to_cachable_inference_item(
            self.mock_object_detection_request, self.mock_object_detection_response
        )
        self.assertEqual(result["inference_id"], self.mock_object_detection_request.id)
        self.assertEqual(
            result["request"]["api_key"],
            self.mock_object_detection_request.dict.return_value["api_key"],
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["class"],
            self.mock_object_detection_response.predictions[0].class_name,
        )
        self.assertEqual(
            result["response"][0]["predictions"][0]["confidence"],
            self.mock_object_detection_response.predictions[0].confidence,
        )

    def test_build_condensed_response(self):
        responses = [
            self.mock_object_detection_response,
        ]
        result = build_condensed_response(responses, self.mock_object_detection_request)
        self.assertEqual(len(result), 1)

    def test_build_condensed_response_no_predictions(self):
        self.mock_object_detection_response.predictions = None
        result = build_condensed_response(
            self.mock_object_detection_response, self.mock_object_detection_request
        )
        self.assertEqual(len(result), 0)
