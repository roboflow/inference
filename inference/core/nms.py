from typing import Dict, List, Optional

import numpy as np
from pycocotools import mask as mask_utils


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

    if box_format == "xywh":
        pred_view = prediction[:, :, :4]

        # Calculate all values without allocating a new array
        x1 = pred_view[:, :, 0] - pred_view[:, :, 2] / 2
        y1 = pred_view[:, :, 1] - pred_view[:, :, 3] / 2
        x2 = pred_view[:, :, 0] + pred_view[:, :, 2] / 2
        y2 = pred_view[:, :, 1] + pred_view[:, :, 3] / 2

        # Assign directly to the view
        pred_view[:, :, 0] = x1
        pred_view[:, :, 1] = y1
        pred_view[:, :, 2] = x2
        pred_view[:, :, 3] = y2
    elif box_format != "xyxy":
        raise ValueError(
            "box_format must be either 'xywh' or 'xyxy', got {}".format(box_format)
        )

    batch_predictions = []

    # Pre-allocate space for class confidence and class prediction arrays
    cls_confs_shape = (prediction.shape[1], 1)

    for np_image_i, np_image_pred in enumerate(prediction):
        np_conf_mask = np_image_pred[:, 4] >= conf_thresh
        if not np.any(np_conf_mask):  # Quick check if no boxes pass threshold
            batch_predictions.append([])
            continue

        np_image_pred = np_image_pred[np_conf_mask]

        # Handle empty case after filtering
        if np_image_pred.shape[0] == 0:
            batch_predictions.append([])
            continue

        cls_confs = np_image_pred[:, 5 : num_classes + 5]
        # Check for empty classes after slicing
        if cls_confs.shape[1] == 0:
            batch_predictions.append([])
            continue

        np_class_conf = np.max(cls_confs, axis=1, keepdims=True)
        np_class_pred = np.argmax(cls_confs, axis=1, keepdims=True)
        # Extract mask predictions if any
        if num_masks > 0:
            np_mask_pred = np_image_pred[:, 5 + num_classes :]
            # Construct final detections array directly
            np_detections = np.concatenate(
                [
                    np_image_pred[:, :5],
                    np_class_conf,
                    np_class_pred.astype(np.float32),
                    np_mask_pred,
                ],
                axis=1,
            )
        else:
            # Optimization: Avoid concatenation when no masks are present
            np_detections = np.concatenate(
                [np_image_pred[:, :5], np_class_conf, np_class_pred.astype(np.float32)],
                axis=1,
            )
        filtered_predictions = []
        if class_agnostic:
            # Sort by confidence directly
            sorted_indices = np.argsort(-np_detections[:, 4])
            np_detections_sorted = np_detections[sorted_indices]
            # Directly pass to optimized NMS
            filtered_predictions.extend(
                non_max_suppression_fast(np_detections_sorted, iou_thresh)
            )
        else:
            np_unique_labels = np.unique(np_class_pred)

            # Process each class
            for c in np_unique_labels:
                class_mask = np.atleast_1d(np_class_pred.squeeze() == c)
                np_detections_class = np_detections[class_mask]

                # Skip empty arrays
                if np_detections_class.shape[0] == 0:
                    continue

                # Sort by confidence (highest first)
                sorted_indices = np.argsort(-np_detections_class[:, 4])
                np_detections_sorted = np_detections_class[sorted_indices]

                # Apply optimized NMS and extend filtered predictions
                filtered_predictions.extend(
                    non_max_suppression_fast(np_detections_sorted, iou_thresh)
                )

        # Sort final predictions by confidence and limit to max_detections
        if filtered_predictions:
            # Use numpy sort for better performance
            filtered_np = np.array(filtered_predictions)
            idx = np.argsort(-filtered_np[:, 4])
            filtered_np = filtered_np[idx]

            # Limit to max_detections
            if len(filtered_np) > max_detections:
                filtered_np = filtered_np[:max_detections]

            batch_predictions.append(list(filtered_np))
        else:
            batch_predictions.append([])

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


def nms_rle(
    rles: List[Dict], confidences: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """
    NMS (Non-Maximum Suppression) for RLE-encoded masks.
    Removes duplicate detections by keeping only the highest confidence detection
    when multiple detections overlap the same object.

    Args:
        rles: List of RLE-encoded masks from pycocotools
        confidences: Array of confidence scores for each mask
        iou_threshold: IoU threshold above which detections are considered duplicates

    Returns:
        Boolean array indicating which detections to keep (True) or suppress (False)
    """
    num_detections = len(rles)
    if num_detections == 0:
        return np.array([], dtype=bool)

    # Sort by confidence descending
    sort_index = np.argsort(confidences)[::-1]
    sorted_rles = [rles[i] for i in sort_index]

    # Calculate IoU matrix: ious[i,j] = overlap between detection i and j
    ious = mask_utils.iou(sorted_rles, sorted_rles, [0] * num_detections)

    # Greedy NMS: keep highest confidence, suppress overlaps
    keep = np.ones(num_detections, dtype=bool)
    for i in range(num_detections):
        if keep[i]:
            suppress = (ious[i, i+1:] > iou_threshold)
            keep[i+1:][suppress] = False

    # Return indices in original order
    return keep[np.argsort(sort_index)]
