from time import perf_counter
from typing import Any

from ultralytics import YOLO

from inference.core.cache import cache
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.models.defaults import DEFAULT_CONFIDENCE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.hash import get_string_list_hash
from inference.core.utils.image_utils import load_image_rgb


class YOLOWorld(RoboflowCoreModel):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        model: The GroundingDINO model.
    """

    def __init__(self, *args, model_id="yolo_world/l", **kwargs):
        """Initializes the YOLO-World model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, model_id=model_id, **kwargs)

        self.model = YOLO(self.cache_file("yolo-world.pt"))
        self.class_names = None

    def preproc_image(self, image: Any):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_rgb(image)
        return np_image[:, :, ::-1]

    def infer_from_request(
        self,
        request: YOLOWorldInferenceRequest,
    ) -> ObjectDetectionInferenceResponse:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        """
        result = self.infer(**request.dict())
        return result

    def infer(
        self,
        image: Any = None,
        text: list = None,
        confidence: float = DEFAULT_CONFIDENCE,
        **kwargs,
    ):
        """
        Run inference on a provided image.

        Args:
            request (CVInferenceRequest): The inference request.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            GroundingDINOInferenceRequest: The inference response.
        """
        t1 = perf_counter()
        image = self.preproc_image(image)
        img_dims = image.shape

        if text is not None and text != self.class_names:
            self.set_classes(text)
        if self.class_names is None:
            raise ValueError(
                "Class names not set and not provided in the request. Must set class names before inference or provide them via the argument `text`."
            )
        results = self.model.predict(
            image,
            conf=confidence,
            verbose=False,
        )[0]

        t2 = perf_counter() - t1

        predictions = []
        for i, box in enumerate(results.boxes):
            x, y, w, h = box.xywh.tolist()[0]
            class_id = int(box.cls)
            predictions.append(
                ObjectDetectionPrediction(
                    **{
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": float(box.conf),
                        "class": self.class_names[class_id],
                        "class_id": class_id,
                    }
                )
            )

        responses = ObjectDetectionInferenceResponse(
            predictions=predictions,
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2,
        )
        return responses

    def set_classes(self, text: list):
        """Set the class names for the model.

        Args:
            text (list): The class names.
        """
        text_hash = get_string_list_hash(text)
        cached_embeddings = cache.get_numpy(text_hash)
        if cached_embeddings is not None:
            self.model.model.txt_feats = cached_embeddings
            self.model.model.model[-1].nc = len(text)
        else:
            self.model.set_classes(text)
            cache.set_numpy(text_hash, self.model.model.txt_feats, expire=300)
        self.class_names = text

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["yolo-world.pt"]
