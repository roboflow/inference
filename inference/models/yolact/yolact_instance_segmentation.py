from time import perf_counter
from typing import List, Union

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


class YOLACTInstanceSegmentationOnnxRoboflowInferenceModel(
    OnnxRoboflowInferenceModel, InstanceSegmentationMixin
):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method)"""

    @property
    def weights_file(self) -> str:
        """Gets the weights file.

        Returns:
            str: Path to the weights file.
        """
        return "weights.onnx"

    def infer(
        self, request: InstanceSegmentationInferenceRequest
    ) -> Union[
        List[InstanceSegmentationInferenceResponse],
        InstanceSegmentationInferenceResponse,
    ]:
        """Takes an instance segmentation inference request, preprocesses all images, runs inference on all images, and returns the postprocessed detections in the form of inference response objects.

        Args:
            request (InstanceSegmentationInferenceRequest): A request containing 1 to N inference image objects and other inference parameters (confidence, iou threshold, etc.)

        Returns:
            Union[List[InstanceSegmentationInferenceResponse], InstanceSegmentationInferenceResponse]: One to N inference response objects based on the number of inference request images in the inference request object, each inference response object contains a list of predictions.
        """
        t1 = perf_counter()

        if isinstance(request.image, list):
            imgs_with_dims = [self.preproc_image(i) for i in request.image]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
            unwrap = False
        else:
            img_in, img_dims = self.preproc_image(request.image)
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
            conf_thresh=request.confidence,
            iou_thresh=request.iou_threshold,
            class_agnostic=request.class_agnostic_nms,
            max_detections=request.max_detections,
            max_candidate_detections=request.max_candidates,
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
                }
                if not request.class_filter or class_name in request.class_filter:
                    preds.append(pred)
            batch_preds.append(preds)

        responses = [
            InstanceSegmentationInferenceResponse(
                predictions=[InstanceSegmentationPrediction(**p) for p in batch_pred],
                time=perf_counter() - t1,
                image=InferenceResponseImage(
                    width=img_dims[i][1], height=img_dims[i][0]
                ),
            )
            for i, batch_pred in enumerate(batch_preds)
        ]

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if unwrap:
            responses = responses[0]
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
