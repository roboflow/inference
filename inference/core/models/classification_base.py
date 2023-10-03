from io import BytesIO
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference.core.data_models import (
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    MultiLabelClassificationInferenceResponse,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_rgb


class ClassificationBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    """Base class for ONNX models for Roboflow classification inference.

    Attributes:
        multiclass (bool): Whether the classification is multi-class or not.

    Methods:
        get_infer_bucket_file_list() -> list: Get the list of required files for inference.
        softmax(x): Compute softmax values for a given set of scores.
        infer(request: ClassificationInferenceRequest) -> Union[List[Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]], Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]]: Perform inference on a given request and return the response.
        draw_predictions(inference_request, inference_response): Draw prediction visuals on an image.
    """

    task_type = "classification"

    def __init__(self, *args, **kwargs):
        """Initialize the model, setting whether it is multiclass or not."""
        super().__init__(*args, **kwargs)
        self.multiclass = self.environment.get("MULTICLASS", False)

    def draw_predictions(self, inference_request, inference_response):
        """Draw prediction visuals on an image.

        This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

        Args:
            inference_request: The request object containing the image and parameters.
            inference_response: The response object containing the predictions and other details.

        Returns:
            bytes: The bytes of the visualized image in JPEG format.
        """
        image = load_image_rgb(inference_request.image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        if isinstance(inference_response.predictions, list):
            prediction = inference_response.predictions[0]
            color = self.colors.get(prediction.class_name, "#4892EA")
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=color,
                width=inference_request.visualization_stroke_width,
            )
            text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, 0))
        else:
            if len(inference_response.predictions) > 0:
                box_color = "#4892EA"
                draw.rectangle(
                    [0, 0, image.size[1], image.size[0]],
                    outline=box_color,
                    width=inference_request.visualization_stroke_width,
                )
            row = 0
            predictions = [
                (cls_name, pred)
                for cls_name, pred in inference_response.predictions.items()
            ]
            predictions = sorted(
                predictions, key=lambda x: x[1].confidence, reverse=True
            )
            for i, (cls_name, pred) in enumerate(predictions):
                color = self.colors.get(cls_name, "#4892EA")
                text = f"{cls_name} {pred.confidence:.2f}"
                text_size = font.getbbox(text)

                # set button size + 10px margins
                button_size = (text_size[2] + 20, text_size[3] + 20)
                button_img = Image.new("RGBA", button_size, color)
                # put text on button with 10px margins
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

                # put button on source image in position (0, 0)
                image.paste(button_img, (0, row))
                row += button_size[1]

        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["environment.json"].
        """
        return ["environment.json"]

    def infer(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        return_image_dims: bool = False,
        **kwargs,
    ):
        """
        Perform inference on the provided image(s) and return the predictions.

        Args:
            image (Any): The image or list of images to be processed.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            return_image_dims (bool, optional): If set to True, the function will also return the dimensions of the image. Defaults to False.
            **kwargs: Additional parameters to customize the inference process.

        Returns:
            Union[List[np.array], np.array, Tuple[List[np.array], List[Tuple[int, int]]], Tuple[np.array, Tuple[int, int]]]:
            If `return_image_dims` is True and a list of images is provided, a tuple containing a list of prediction arrays and a list of image dimensions (width, height) is returned.
            If `return_image_dims` is True and a single image is provided, a tuple containing the prediction array and image dimensions (width, height) is returned.
            If `return_image_dims` is False and a list of images is provided, only the list of prediction arrays is returned.
            If `return_image_dims` is False and a single image is provided, only the prediction array is returned.

        Notes:
            - The input image(s) will be preprocessed (normalized and reshaped) before inference.
            - This function uses an ONNX session to perform inference on the input image(s).
        """
        return super().infer(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            return_image_dims=return_image_dims,
        )

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preprocess_return_metadata: PreprocessReturnMetadata,
        return_image_dims=False,
        **kwargs,
    ) -> Any:
        predictions = predictions[0]
        if return_image_dims:
            return predictions, preprocess_return_metadata["img_dims"]
        else:
            return predictions

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        predictions = self.onnx_session.run(None, {self.input_name: img_in})
        return (predictions,)

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        if isinstance(image, list):
            imgs_with_dims = [
                self.preproc_image(
                    i,
                    disable_preproc_auto_orient=kwargs["disable_preproc_auto_orient"],
                    disable_preproc_contrast=kwargs["disable_preproc_contrast"],
                    disable_preproc_grayscale=kwargs["disable_preproc_grayscale"],
                    disable_preproc_static_crop=kwargs["disable_preproc_static_crop"],
                )
                for i in image
            ]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
        else:
            img_in, img_dims = self.preproc_image(
                image,
                disable_preproc_auto_orient=kwargs["disable_preproc_auto_orient"],
                disable_preproc_contrast=kwargs["disable_preproc_contrast"],
                disable_preproc_grayscale=kwargs["disable_preproc_grayscale"],
                disable_preproc_static_crop=kwargs["disable_preproc_static_crop"],
            )
            img_dims = [img_dims]

        img_in /= 255.0

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        img_in = img_in.astype(np.float32)

        img_in[:, 0, :, :] = (img_in[:, 0, :, :] - mean[0]) / std[0]
        img_in[:, 1, :, :] = (img_in[:, 1, :, :] - mean[1]) / std[1]
        img_in[:, 2, :, :] = (img_in[:, 2, :, :] - mean[2]) / std[2]
        return img_in, PreprocessReturnMetadata({"img_dims": img_dims})

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        predictions_data = self.infer(**request.dict(), return_image_dims=True)
        responses = self.make_response(
            *predictions_data,
            confidence=request.confidence,
        )
        for response in responses:
            response.time = perf_counter() - t1

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses

    def make_response(
        self,
        predictions,
        img_dims,
        confidence: float = 0.5,
        **kwargs,
    ) -> Union[ClassificationInferenceResponse, List[ClassificationInferenceResponse]]:
        """
        Create response objects for the given predictions and image dimensions.

        Args:
            predictions (list): List of prediction arrays from the inference process.
            img_dims (list): List of tuples indicating the dimensions (width, height) of each image.
            confidence (float, optional): Confidence threshold for filtering predictions. Defaults to 0.5.
            **kwargs: Additional parameters to influence the response creation process.

        Returns:
            Union[ClassificationInferenceResponse, List[ClassificationInferenceResponse]]: A response object or a list of response objects encapsulating the prediction details.

        Notes:
            - If the model is multiclass, a `MultiLabelClassificationInferenceResponse` is generated for each image.
            - If the model is not multiclass, a `ClassificationInferenceResponse` is generated for each image.
            - Predictions below the confidence threshold are filtered out.
        """
        responses = []
        confidence_threshold = float(confidence)
        for ind, prediction in enumerate(predictions):
            if self.multiclass:
                preds = prediction[0]
                results = dict()
                predicted_classes = []
                for i, o in enumerate(preds):
                    cls_name = self.class_names[i]
                    score = float(o)
                    results[cls_name] = {"confidence": score, "class_id": i}
                    if score > confidence_threshold:
                        predicted_classes.append(cls_name)
                response = MultiLabelClassificationInferenceResponse(
                    image=InferenceResponseImage(
                        width=img_dims[ind][0], height=img_dims[ind][1]
                    ),
                    predicted_classes=predicted_classes,
                    predictions=results,
                )
            else:
                preds = prediction[0]
                preds = self.softmax(preds)
                results = []
                for i, cls_name in enumerate(self.class_names):
                    score = float(preds[i])
                    pred = {
                        "class_id": i,
                        "class": cls_name,
                        "confidence": round(score, 4),
                    }
                    results.append(pred)
                results = sorted(results, key=lambda x: x["confidence"], reverse=True)

                response = ClassificationInferenceResponse(
                    image=InferenceResponseImage(
                        width=img_dims[ind][1], height=img_dims[ind][0]
                    ),
                    predictions=results,
                    top=results[0]["class"],
                    confidence=results[0]["confidence"],
                )
            responses.append(response)

        return responses

    @staticmethod
    def softmax(x):
        """Compute softmax values for each set of scores in x.

        Args:
            x (np.array): The input array containing the scores.

        Returns:
            np.array: The softmax values for each set of scores.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
