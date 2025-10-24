from inference.core.models.exp_adapter import InferenceExpObjectDetectionModelAdapter


class RFDetrExperimentalModel(InferenceExpObjectDetectionModelAdapter):
    """Adapter for RF-DETR using inference_exp AutoModel backend.

    This class wraps an inference_exp AutoModel to present the same interface
    as legacy models in the inference server.
    """

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return {
            "threshold": kwargs.get("confidence"),
        }
