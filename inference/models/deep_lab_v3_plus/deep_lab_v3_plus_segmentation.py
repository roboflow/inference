from inference.core.models.semantic_segmentation_base import (
    SemanticSegmentationBaseOnnxRoboflowInferenceModel,
)


class DeepLabV3PlusSemanticSegmentation(
    SemanticSegmentationBaseOnnxRoboflowInferenceModel
):
    """DeepLabV3Plus Semantic Segmentation ONNX Inference Model.

    This class is responsible for performing semantic segmentation using the DeepLabV3Plus model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs inference on the given image using the ONNX session.
    """

    # match train params
    preprocess_means = [0.485, 0.456, 0.406]
    preprocess_stds = [0.229, 0.224, 0.225]

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the DeepLabV3Plus model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"
