from time import perf_counter
from typing import Any, List, Union

import cv2
import numpy as np

from inference.core.data_models import (
    InferenceResponseImage,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
)
from inference.core.models.mixins import InstanceSegmentationMixin
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import (
    crop_mask,
    mask2poly,
    postprocess_predictions,
    scale_boxes,
    scale_polys,
)


class YOLACT(OnnxRoboflowInferenceModel, InstanceSegmentationMixin):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method)"""

    @property
    def weights_file(self) -> str:
        """Gets the weights file.

        Returns:
            str: Path to the weights file.
        """
        return "weights.onnx"

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = False,
        confidence: float = 0.5,
        iou_threshold: float = 0.5,
        max_candidates: int = 3000,
        max_detections: int = 300,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[List[dict]]:
        """
        Performs instance segmentation inference on a given image, post-processes the results,
        and returns the segmented instances as dictionaries containing their properties.

        Args:
            image (Any): The image or list of images to segment. Can be in various formats (e.g., raw array, PIL image).
            class_agnostic_nms (bool, optional): Whether to perform class-agnostic non-max suppression. Defaults to False.
            confidence (float, optional): Confidence threshold for filtering weak detections. Defaults to 0.5.
            iou_threshold (float, optional): Intersection-over-union threshold for non-max suppression. Defaults to 0.5.
            max_candidates (int, optional): Maximum number of candidate detections to consider. Defaults to 3000.
            max_detections (int, optional): Maximum number of detections to return after non-max suppression. Defaults to 300.
            return_image_dims (bool, optional): Whether to return the dimensions of the input image(s). Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            List[List[dict]]: Each list contains dictionaries of segmented instances for a given image. Each dictionary contains:
                - x, y: Center coordinates of the instance.
                - width, height: Width and height of the bounding box around the instance.
                - class: Name of the detected class.
                - confidence: Confidence score of the detection.
                - points: List of points describing the segmented mask's boundary.
                - class_id: ID corresponding to the detected class.
            If `return_image_dims` is True, the function returns a tuple where the first element is the list of detections and the
            second element is the list of image dimensions.

        Notes:
            - The function supports processing multiple images in a batch.
            - If an input list of images is provided, the function returns a list of lists,
              where each inner list corresponds to the detections for a specific image.
            - The function internally uses an ONNX model for inference.
        """
        t1 = perf_counter()

        if isinstance(image, list):
            imgs_with_dims = [self.preproc_image(i) for i in image]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
            unwrap = False
        else:
            img_in, img_dims = self.preproc_image(image)
            img_dims = [img_dims]
            unwrap = True

        # IN BGR order (for some reason)
        mean = (103.94, 116.78, 123.68)
        std = (57.38, 57.12, 58.40)

        img_in = img_in.astype(np.float32)

        # Our channels are RGB, so apply mean and std accordingly
        img_in[:, 0, :, :] = (img_in[:, 0, :, :] - mean[2]) / std[2]
        img_in[:, 1, :, :] = (img_in[:, 1, :, :] - mean[1]) / std[1]
        img_in[:, 2, :, :] = (img_in[:, 2, :, :] - mean[0]) / std[0]

        predictions = self.onnx_session.run(None, {self.input_name: img_in})

        loc_data = np.float32(predictions[0])
        conf_data = np.float32(predictions[1])
        mask_data = np.float32(predictions[2])
        prior_data = np.float32(predictions[3])
        proto_data = np.float32(predictions[4])

        batch_size = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        boxes = np.zeros((batch_size, num_priors, 4))
        for batch_idx in range(batch_size):
            boxes[batch_idx, :, :] = self.decode_predicted_bboxes(
                loc_data[batch_idx], prior_data
            )

        conf_preds = np.reshape(
            conf_data, (batch_size, num_priors, self.num_classes + 1)
        )
        class_confs = conf_preds[:, :, 1:]  # remove background class
        box_confs = np.expand_dims(
            np.max(class_confs, axis=2), 2
        )  # get max conf for each box

        predictions = np.concatenate((boxes, box_confs, class_confs, mask_data), axis=2)

        predictions[:, :, 0] *= img_in.shape[2]
        predictions[:, :, 1] *= img_in.shape[3]
        predictions[:, :, 2] *= img_in.shape[2]
        predictions[:, :, 3] *= img_in.shape[3]

        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            num_masks=32,
            box_format="xyxy",
        )
        predictions = np.array(predictions)
        batch_preds = []
        for batch_idx, img_dim in zip(range(batch_size), img_dims):
            boxes = predictions[batch_idx, :, :4]
            scores = predictions[batch_idx, :, 4]
            classes = predictions[batch_idx, :, 6]
            masks = predictions[batch_idx, :, 7:]
            proto = proto_data[batch_idx]
            decoded_masks = self.decode_masks(boxes, masks, proto, img_in.shape[2:])
            polys = mask2poly(decoded_masks)
            infer_shape = (self.img_size_w, self.img_size_h)
            boxes = postprocess_predictions(
                [boxes], infer_shape, [img_dim], self.preproc, self.resize_method
            )[0]
            polys = scale_polys(
                img_in.shape[2:],
                polys,
                img_dim,
                self.preproc,
                resize_method=self.resize_method,
            )
            preds = []
            for i, (box, poly, score, cls) in enumerate(
                zip(boxes, polys, scores, classes)
            ):
                confidence = float(score)
                class_name = self.class_names[int(cls)]
                points = [{"x": round(x, 1), "y": round(y, 1)} for (x, y) in poly]
                pred = {
                    "x": round((box[2] + box[0]) / 2, 1),
                    "y": round((box[3] + box[1]) / 2, 1),
                    "width": int(box[2] - box[0]),
                    "height": int(box[3] - box[1]),
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "points": points,
                    "class_id": int(cls),
                }
                preds.append(pred)
            batch_preds.append(preds)

        if return_image_dims:
            return batch_preds, img_dims
        else:
            return batch_preds

    def make_response(
        self,
        predictions,
        img_dims,
        class_filter: List[str] = None,
    ):
        """
        Constructs a list of InstanceSegmentationInferenceResponse objects based on the provided predictions
        and image dimensions, optionally filtering by class name.

        Args:
            predictions (List[List[dict]]): A list containing batch predictions, where each inner list contains
                dictionaries of segmented instances for a given image.
            img_dims (List[Tuple[int, int]]): List of tuples specifying the dimensions of each image in the format
                (height, width).
            class_filter (List[str], optional): A list of class names to filter the predictions by. If not provided,
                all predictions are included.

        Returns:
            List[InstanceSegmentationInferenceResponse]: A list of response objects, each containing the filtered
            predictions and corresponding image dimensions for a given image.

        Examples:
            >>> predictions = [[{"class_name": "cat", ...}, {"class_name": "dog", ...}], ...]
            >>> img_dims = [(300, 400), ...]
            >>> responses = make_response(predictions, img_dims, class_filter=["cat"])
            >>> len(responses[0].predictions)  # Only predictions with "cat" class are included
            1
        """
        responses = [
            InstanceSegmentationInferenceResponse(
                predictions=[
                    InstanceSegmentationPrediction(**p)
                    for p in batch_pred
                    if not class_filter or p["class_name"] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[i][1], height=img_dims[i][0]
                ),
            )
            for i, batch_pred in enumerate(predictions)
        ]
        return responses

    def decode_masks(self, boxes, masks, proto, img_dim):
        """Decodes the masks from the given parameters.

        Args:
            boxes (np.array): Bounding boxes.
            masks (np.array): Masks.
            proto (np.array): Proto data.
            img_dim (tuple): Image dimensions.

        Returns:
            np.array: Decoded masks.
        """
        ret_mask = np.matmul(proto, np.transpose(masks))
        ret_mask = 1 / (1 + np.exp(-ret_mask))
        w, h, _ = ret_mask.shape
        gain = min(h / img_dim[0], w / img_dim[1])  # gain  = old / new
        pad = (w - img_dim[1] * gain) / 2, (h - img_dim[0] * gain) / 2  # wh padding
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(h - pad[1]), int(w - pad[0])
        ret_mask = np.transpose(ret_mask, (2, 0, 1))
        ret_mask = ret_mask[:, top:bottom, left:right]
        if len(ret_mask.shape) == 2:
            ret_mask = np.expand_dims(ret_mask, axis=0)
        ret_mask = ret_mask.transpose((1, 2, 0))
        ret_mask = cv2.resize(ret_mask, img_dim, interpolation=cv2.INTER_LINEAR)
        if len(ret_mask.shape) == 2:
            ret_mask = np.expand_dims(ret_mask, axis=2)
        ret_mask = ret_mask.transpose((2, 0, 1))
        ret_mask = crop_mask(ret_mask, boxes)  # CHW
        ret_mask[ret_mask < 0.5] = 0

        return ret_mask

    def decode_predicted_bboxes(self, loc, priors):
        """Decode predicted bounding box coordinates using the scheme employed by Yolov2.

        Args:
            loc (np.array): The predicted bounding boxes of size [num_priors, 4].
            priors (np.array): The prior box coordinates with size [num_priors, 4].

        Returns:
            np.array: A tensor of decoded relative coordinates in point form with size [num_priors, 4].
        """

        variances = [0.1, 0.2]

        boxes = np.concatenate(
            [
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ],
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes
