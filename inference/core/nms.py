from typing import Optional

import numpy as np


def w_np_non_max_suppression(
    prediction,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    class_agnostic: bool = False,
    max_detections: int = 300,
    max_candidate_detections: int = 3000,
    timeout_seconds: Optional[int] = None,
    num_masks: int = 0,
    box_format: str = "xywh",
):
    """Applies non-maximum suppression to predictions.

    Args:
        prediction (np.ndarray): Array of predictions. Format for single prediction is
            [bbox x 4, max_class_confidence, (confidence) x num_of_classes, additional_element x num_masks]
        conf_thresh (float, optional): Confidence threshold. Defaults to 0.25.
        iou_thresh (float, optional): IOU threshold. Defaults to 0.45.
        class_agnostic (bool, optional): Whether to ignore class labels. Defaults to False.
        max_detections (int, optional): Maximum number of detections. Defaults to 300.
        max_candidate_detections (int, optional): Maximum number of candidate detections. Defaults to 3000.
        timeout_seconds (Optional[int], optional): Timeout in seconds. Defaults to None.
        num_masks (int, optional): Number of masks. Defaults to 0.
        box_format (str, optional): Format of bounding boxes. Either 'xywh' or 'xyxy'. Defaults to 'xywh'.

    Returns:
        list: List of filtered predictions after non-maximum suppression. Format of a single result is:
            [bbox x 4, max_class_confidence, max_class_confidence, id_of_class_with_max_confidence,
            additional_element x num_masks]
    """
    num_classes = prediction.shape[2] - 5 - num_masks

    np_box_corner = np.zeros(prediction.shape)
    if box_format == "xywh":
        np_box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        np_box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        np_box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        np_box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = np_box_corner[:, :, :4]
    elif box_format == "xyxy":
        pass
    else:
        raise ValueError(
            "box_format must be either 'xywh' or 'xyxy', got {}".format(box_format)
        )

    batch_predictions = []
    for np_image_i, np_image_pred in enumerate(prediction):
        filtered_predictions = []
        np_conf_mask = np_image_pred[:, 4] >= conf_thresh

        np_image_pred = np_image_pred[np_conf_mask]
        cls_confs = np_image_pred[:, 5 : num_classes + 5]
        if (
            np_image_pred.shape[0] == 0
            or np_image_pred.shape[1] == 0
            or cls_confs.shape[1] == 0
        ):
            batch_predictions.append(filtered_predictions)
            continue

        np_class_conf = np.max(cls_confs, 1)
        np_class_pred = np.argmax(np_image_pred[:, 5 : num_classes + 5], 1)
        np_class_conf = np.expand_dims(np_class_conf, axis=1)
        np_class_pred = np.expand_dims(np_class_pred, axis=1)
        np_mask_pred = np_image_pred[:, 5 + num_classes :]
        np_detections = np.append(
            np.append(
                np.append(np_image_pred[:, :5], np_class_conf, axis=1),
                np_class_pred,
                axis=1,
            ),
            np_mask_pred,
            axis=1,
        )

        np_unique_labels = np.unique(np_detections[:, 6])

        if class_agnostic:
            np_detections_class = sorted(
                np_detections, key=lambda row: row[4], reverse=True
            )
            filtered_predictions.extend(
                non_max_suppression_fast(np.array(np_detections_class), iou_thresh)
            )
        else:
            for c in np_unique_labels:
                np_detections_class = np_detections[np_detections[:, 6] == c]
                np_detections_class = sorted(
                    np_detections_class, key=lambda row: row[4], reverse=True
                )
                filtered_predictions.extend(
                    non_max_suppression_fast(np.array(np_detections_class), iou_thresh)
                )
        filtered_predictions = sorted(
            filtered_predictions, key=lambda row: row[4], reverse=True
        )
        batch_predictions.append(filtered_predictions[:max_detections])
    return batch_predictions


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    """Applies non-maximum suppression to bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes with confidence scores.
        overlapThresh (float): Overlap threshold for suppression.

    Returns:
        list: List of bounding boxes after non-maximum suppression.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    conf = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("float")
