from inference.core.models.exp_adapter import InferenceExpObjectDetectionModelAdapter


class Yolo8ODExperimentalModel(InferenceExpObjectDetectionModelAdapter):
    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return {
            "conf_thresh": kwargs.get("confidence"),
            "iou_thresh": kwargs.get("iou_threshold"),
            "max_detections": kwargs.get("max_detections"),
            "class_agnostic": kwargs.get("class_agnostic"),
        }
