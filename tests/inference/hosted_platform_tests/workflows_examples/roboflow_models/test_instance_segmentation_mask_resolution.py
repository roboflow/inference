import numpy as np
import pytest
import supervision as sv

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.entities import VisualisationResponseFormat
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    apply_auth_mode,
)


@pytest.mark.parametrize("version", ["v4", "v5"])
@pytest.mark.parametrize(
    "mode,factor", [("accurate", 1.0), ("tradeoff", 0.5), ("fast", 0.0)]
)
def test_mask_resolution_workflow_visualization_and_serialization(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
    asl_image: np.ndarray,
    auth_mode: str,
    version: str,
    mode: str,
    factor: float,
) -> None:
    client = InferenceHTTPClient(
        api_url=instance_segmentation_service_url, api_key=ROBOFLOW_API_KEY
    ).configure(
        InferenceConfiguration(
            output_visualisation_format=VisualisationResponseFormat.NUMPY
        )
    )
    client = apply_auth_mode(client, auth_mode)
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": f"roboflow_core/roboflow_instance_segmentation_model@{version}",
                "name": "segment",
                "image": "$inputs.image",
                "model_id": segmentation_model_id,
                "mask_decode_mode": mode,
                "tradeoff_factor": factor,
            },
            {
                "type": "roboflow_core/mask_visualization@v1",
                "name": "visualize",
                "image": "$inputs.image",
                "predictions": "$steps.segment.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segment.predictions",
            },
            {
                "type": "JsonField",
                "name": "visualization",
                "selector": "$steps.visualize.image",
            },
        ],
    }

    results = client.run_workflow(
        specification=specification,
        images={"image": [asl_image, asl_image]},
    )

    assert len(results) == 2
    for result in results:
        predictions = result["predictions"]
        detections = sv.Detections.from_inference(predictions)
        assert len(detections) > 0
        assert detections.mask.shape == (len(detections), *asl_image.shape[:2])
        assert detections.mask.any()
        assert all(
            prediction.get("rle_mask") for prediction in predictions["predictions"]
        )
        assert all(
            tuple(prediction["rle_mask"]["size"]) == asl_image.shape[:2]
            for prediction in predictions["predictions"]
        )
        sv.MaskAnnotator().annotate(asl_image.copy(), detections)
        visualization = result["visualization"]
        assert isinstance(visualization, np.ndarray)
        assert visualization.shape == asl_image.shape
        assert np.any(visualization != asl_image)

    first = sv.Detections.from_inference(results[0]["predictions"])
    second = sv.Detections.from_inference(results[1]["predictions"])
    np.testing.assert_allclose(first.xyxy, second.xyxy, atol=1)
    np.testing.assert_array_equal(first.mask, second.mask)
