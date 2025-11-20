import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt

class TestSam3Proxy(unittest.TestCase):
    def setUp(self):
        self.model_manager = MagicMock()
        # Mock the HttpInterface initialization to avoid side effects if any
        # But we need the app to be created.
        # We will use patch.dict to set environment variables or patch the module attributes.
        pass

    @patch("inference.core.interfaces.http.http_api.SAM3_EXEC_MODE", "remote")
    @patch("inference.core.interfaces.http.http_api.CORE_MODEL_SAM3_ENABLED", True)
    @patch("inference.core.interfaces.http.http_api.GCP_SERVERLESS", False)
    @patch("inference.core.interfaces.http.http_api.API_BASE_URL", "http://remote-api")
    @patch("inference.core.interfaces.http.http_api.requests.post")
    def test_sam3_concept_segment_proxy(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "prompt_results": [],
            "time": 0.1
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Initialize app
        http_interface = HttpInterface(model_manager=self.model_manager)
        client = TestClient(http_interface.app)

        # payload
        payload = {
            "model_id": "sam3/sam3_interactive",
            "image": {
                "type": "base64",
                "value": "base64string"
            },
            "prompts": [
                {"type": "text", "text": "cat"}
            ],
            "output_prob_thresh": 0.5
        }

        response = client.post("/sam3/concept_segment", json=payload)

        self.assertEqual(response.status_code, 200)
        
        # Verify requests.post called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://remote-api/inferenceproxy/seg-preview?api_key=None")
        self.assertEqual(kwargs["json"]["image"]["value"], "base64string")
        self.assertEqual(kwargs["json"]["prompts"][0]["text"], "cat")

    @patch("inference.core.interfaces.http.http_api.SAM3_EXEC_MODE", "remote")
    @patch("inference.core.interfaces.http.http_api.CORE_MODEL_SAM3_ENABLED", True)
    @patch("inference.core.interfaces.http.http_api.GCP_SERVERLESS", False)
    def test_sam3_embed_image_disabled_in_remote(self):
        http_interface = HttpInterface(model_manager=self.model_manager)
        client = TestClient(http_interface.app)

        payload = {
            "model_id": "sam3/sam3_interactive",
            "image": {"type": "base64", "value": "base64string"}
        }
        # Note: Sam2EmbeddingRequest is used for sam3_embed_image
        
        response = client.post("/sam3/embed_image", json=payload)
        self.assertEqual(response.status_code, 501)
        self.assertIn("not supported in remote execution mode", response.json()["detail"])

    @patch("inference.core.interfaces.http.http_api.SAM3_EXEC_MODE", "remote")
    @patch("inference.core.interfaces.http.http_api.CORE_MODEL_SAM3_ENABLED", True)
    @patch("inference.core.interfaces.http.http_api.GCP_SERVERLESS", False)
    def test_sam3_visual_segment_disabled_in_remote(self):
        http_interface = HttpInterface(model_manager=self.model_manager)
        client = TestClient(http_interface.app)

        payload = {
            "model_id": "sam3/sam3_interactive",
            "image": {"type": "base64", "value": "base64string"},
            "prompts": [{"type": "point", "data": [10, 10], "label": 1}]
        }
        # Note: Sam2SegmentationRequest is used for sam3_visual_segment
        
        response = client.post("/sam3/visual_segment", json=payload)
        self.assertEqual(response.status_code, 501)
        self.assertIn("not supported in remote execution mode", response.json()["detail"])

    @patch("inference.core.interfaces.http.http_api.SAM3_EXEC_MODE", "remote")
    @patch("inference.core.interfaces.http.http_api.CORE_MODEL_SAM3_ENABLED", True)
    @patch("inference.core.interfaces.http.http_api.GCP_SERVERLESS", False)
    def test_sam3_fine_tuned_model_error(self):
        http_interface = HttpInterface(model_manager=self.model_manager)
        client = TestClient(http_interface.app)

        payload = {
            "model_id": "fine-tuned-model",
            "image": {
                "type": "base64",
                "value": "base64string"
            },
            "prompts": [
                {"type": "text", "text": "cat"}
            ],
            "output_prob_thresh": 0.5
        }

        response = client.post("/sam3/concept_segment", json=payload)
        self.assertEqual(response.status_code, 501)
        self.assertIn("Fine-tuned SAM3 models are not supported", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
