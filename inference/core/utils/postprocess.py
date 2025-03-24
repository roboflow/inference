from copy import deepcopy
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from inference.core.exceptions import PostProcessingError
from inference.core.utils.preprocess import (
    STATIC_CROP_KEY,
    static_crop_should_be_applied,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[np.number, np.ndarray]:
    """
    Compute the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector A.
        b (np.ndarray): Vector B.

    Returns:
        float: Cosine similarity between vectors A and B.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def masks2poly(masks: np.ndarray) -> List[np.ndarray]:
    """Converts binary masks to polygonal segments.

    Args:
        masks (numpy.ndarray): A set of binary masks, where masks are multiplied by 255 and converted to uint8 type.

    Returns:
        list: A list of segments, where each segment is obtained by converting the corresponding mask.
    """
    segments = []
    masks = (masks * 255.0).astype(np.uint8)
    for mask in masks:
        segments.append(mask2poly(mask))
    return segments


def masks2multipoly(masks: np.ndarray) -> List[np.ndarray]:
    """Converts binary masks to polygonal segments.

    Args:
        masks (numpy.ndarray): A set of binary masks, where masks are multiplied by 255 and converted to uint8 type.

    Returns:
        list: A list of segments, where each segment is obtained by converting the corresponding mask.
    """
    segments = []
    masks = (masks * 255.0).astype(np.uint8)
    for mask in masks:
        segments.append(mask2multipoly(mask))
    return segments


def mask2poly(mask: np.ndarray) -> np.ndarray:
    """
    Find contours in the mask and return them as a float32 array.

    Args:
        mask (np.ndarray): A binary mask.

    Returns:
        np.ndarray: Contours represented as a float32 array.
    """
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        contours = np.array(
            contours[np.array([len(x) for x in contours]).argmax()]
        ).reshape(-1, 2)
    else:
        contours = np.zeros((0, 2))
    return contours.astype("float32")


def mask2multipoly(mask: np.ndarray) -> np.ndarray:
    """
    Find all contours in the mask and return them as a float32 array.

    Args:
        mask (np.ndarray): A binary mask.

    Returns:
        np.ndarray: Contours represented as a float32 array.
    """
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        contours = [c.reshape(-1, 2).astype("float32") for c in contours]
    else:
        contours = [np.zeros((0, 2)).astype("float32")]
    return contours


def post_process_bboxes(
    predictions: List[List[List[float]]],
    infer_shape: Tuple[int, int],
    img_dims: List[Tuple[int, int]],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
    resize_method: str = "Stretch to",
) -> List[List[List[float]]]:
    """
    Postprocesses each patch of detections by scaling them to the original image coordinates and by shifting them based on a static crop preproc (if applied).

    Args:
        predictions (List[List[List[float]]]): The predictions output from NMS, indices are: batch x prediction x [x1, y1, x2, y2, ...].
        infer_shape (Tuple[int, int]): The shape of the inference image.
        img_dims (List[Tuple[int, int]]): The dimensions of the original image for each batch, indices are: batch x [height, width].
        preproc (dict): Preprocessing configuration dictionary.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
        resize_method (str, optional): Resize method for image. Defaults to "Stretch to".

    Returns:
        List[List[List[float]]]: The scaled and shifted predictions, indices are: batch x prediction x [x1, y1, x2, y2, ...].
    """

    # Get static crop params
    scaled_predictions = []
    # Loop through batches
    for i, batch_predictions in enumerate(predictions):
        if len(batch_predictions) == 0:
            scaled_predictions.append([])
            continue
        np_batch_predictions = np.array(batch_predictions)
        # Get bboxes from predictions (x1,y1,x2,y2)
        predicted_bboxes = np_batch_predictions[:, :4]
        (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
            img_dims[i],
            preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        if resize_method == "Stretch to":
            predicted_bboxes = stretch_bboxes(
                predicted_bboxes=predicted_bboxes,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        elif (
            resize_method == "Fit (black edges) in"
            or resize_method == "Fit (white edges) in"
            or resize_method == "Fit (grey edges) in"
        ):
            predicted_bboxes = undo_image_padding_for_predicted_boxes(
                predicted_bboxes=predicted_bboxes,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        predicted_bboxes = clip_boxes_coordinates(
            predicted_bboxes=predicted_bboxes,
            origin_shape=origin_shape,
        )
        predicted_bboxes = shift_bboxes(
            bboxes=predicted_bboxes,
            shift_x=crop_shift_x,
            shift_y=crop_shift_y,
        )
        np_batch_predictions[:, :4] = predicted_bboxes
        scaled_predictions.append(np_batch_predictions.tolist())
    return scaled_predictions


def stretch_bboxes(
    predicted_bboxes: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    scale_height = origin_shape[0] / infer_shape[0]
    scale_width = origin_shape[1] / infer_shape[1]
    return scale_bboxes(
        bboxes=predicted_bboxes,
        scale_x=scale_width,
        scale_y=scale_height,
    )


def undo_image_padding_for_predicted_boxes(
    predicted_bboxes: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    scale = min(infer_shape[0] / origin_shape[0], infer_shape[1] / origin_shape[1])
    inter_h = round(origin_shape[0] * scale)
    inter_w = round(origin_shape[1] * scale)
    pad_x = (infer_shape[1] - inter_w) / 2
    pad_y = (infer_shape[0] - inter_h) / 2
    predicted_bboxes = shift_bboxes(
        bboxes=predicted_bboxes, shift_x=-pad_x, shift_y=-pad_y
    )
    predicted_bboxes /= scale
    return predicted_bboxes


def clip_boxes_coordinates(
    predicted_bboxes: np.ndarray,
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    predicted_bboxes[:, 0] = np.round(
        np.clip(predicted_bboxes[:, 0], a_min=0, a_max=origin_shape[1])
    )
    predicted_bboxes[:, 2] = np.round(
        np.clip(predicted_bboxes[:, 2], a_min=0, a_max=origin_shape[1])
    )
    predicted_bboxes[:, 1] = np.round(
        np.clip(predicted_bboxes[:, 1], a_min=0, a_max=origin_shape[0])
    )
    predicted_bboxes[:, 3] = np.round(
        np.clip(predicted_bboxes[:, 3], a_min=0, a_max=origin_shape[0])
    )
    return predicted_bboxes


def shift_bboxes(
    bboxes: np.ndarray,
    shift_x: Union[int, float],
    shift_y: Union[int, float],
) -> np.ndarray:
    bboxes[:, 0] += shift_x
    bboxes[:, 2] += shift_x
    bboxes[:, 1] += shift_y
    bboxes[:, 3] += shift_y
    return bboxes


def process_mask_accurate(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Returns masks that are the size of the original image.

    Args:
        protos (numpy.ndarray): Prototype masks.
        masks_in (numpy.ndarray): Input masks.
        bboxes (numpy.ndarray): Bounding boxes.
        shape (tuple): Target shape.

    Returns:
        numpy.ndarray: Processed masks.
    """
    masks = preprocess_segmentation_masks(
        protos=protos,
        masks_in=masks_in,
        shape=shape,
    )

    # Order = 1 -> bilinear
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
    masks = masks.transpose((1, 2, 0))
    masks = cv2.resize(masks, (shape[1], shape[0]), cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=2)
    masks = masks.transpose((2, 0, 1))
    masks = crop_mask(masks, bboxes)
    masks[masks < 0.5] = 0
    return masks


def process_mask_tradeoff(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: Tuple[int, int],
    tradeoff_factor: float,
) -> np.ndarray:
    """Returns masks that are the size of the original image with a tradeoff factor applied.

    Args:
        protos (numpy.ndarray): Prototype masks.
        masks_in (numpy.ndarray): Input masks.
        bboxes (numpy.ndarray): Bounding boxes.
        shape (tuple): Target shape.
        tradeoff_factor (float): Tradeoff factor for resizing masks.

    Returns:
        numpy.ndarray: Processed masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = preprocess_segmentation_masks(
        protos=protos,
        masks_in=masks_in,
        shape=shape,
    )

    # Order = 1 -> bilinear
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
    masks = masks.transpose((1, 2, 0))
    ih, iw = shape
    h = int(mh * (1 - tradeoff_factor) + ih * tradeoff_factor)
    w = int(mw * (1 - tradeoff_factor) + iw * tradeoff_factor)
    size = (h, w)
    if tradeoff_factor != 0:
        masks = cv2.resize(masks, size, cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=2)
    masks = masks.transpose((2, 0, 1))
    c, mh, mw = masks.shape
    down_sampled_boxes = scale_bboxes(
        bboxes=deepcopy(bboxes),
        scale_x=mw / iw,
        scale_y=mh / ih,
    )
    masks = crop_mask(masks, down_sampled_boxes)
    masks[masks < 0.5] = 0
    return masks


def process_mask_fast(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Returns masks in their original size.

    Args:
        protos (numpy.ndarray): Prototype masks.
        masks_in (numpy.ndarray): Input masks.
        bboxes (numpy.ndarray): Bounding boxes.
        shape (tuple): Target shape.

    Returns:
        numpy.ndarray: Processed masks.
    """
    ih, iw = shape
    c, mh, mw = protos.shape  # CHW
    masks = preprocess_segmentation_masks(
        protos=protos,
        masks_in=masks_in,
        shape=shape,
    )
    down_sampled_boxes = scale_bboxes(
        bboxes=deepcopy(bboxes),
        scale_x=mw / iw,
        scale_y=mh / ih,
    )
    masks = crop_mask(masks, down_sampled_boxes)
    masks[masks < 0.5] = 0
    return masks


def preprocess_segmentation_masks(
    protos: np.ndarray,
    masks_in: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    c, mh, mw = protos.shape  # CHW
    masks = protos.astype(np.float32)
    masks = masks.reshape((c, -1))
    masks = masks_in @ masks
    masks = sigmoid(masks)
    masks = masks.reshape((-1, mh, mw))
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    return masks[:, top:bottom, left:right]


def scale_bboxes(bboxes: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    bboxes[:, 0] *= scale_x
    bboxes[:, 2] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 3] *= scale_y
    return bboxes


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    masks = masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    return masks


def post_process_polygons(
    origin_shape: Tuple[int, int],
    polys: List[List[Tuple[float, float]]],
    infer_shape: Tuple[int, int],
    preproc: dict,
    resize_method: str = "Stretch to",
) -> List[List[Tuple[float, float]]]:
    """Scales and shifts polygons based on the given image shapes and preprocessing method.

    This function performs polygon scaling and shifting based on the specified resizing method and
    pre-processing steps. The polygons are transformed according to the ratio and padding between two images.

    Args:
        origin_shape (tuple of int): Shape of the source image (height, width).
        infer_shape (tuple of int): Shape of the target image (height, width).
        polys (list of list of tuple): List of polygons, where each polygon is represented by a list of (x, y) coordinates.
        preproc (object): Preprocessing details used for generating the transformation.
        resize_method (str, optional): Resizing method, either "Stretch to", "Fit (black edges) in", "Fit (white edges) in", or "Fit (grey edges) in". Defaults to "Stretch to".

    Returns:
        list of list of tuple: A list of shifted and scaled polygons.
    """
    (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
        origin_shape, preproc
    )
    new_polys = []
    if resize_method == "Stretch to":
        width_ratio = origin_shape[1] / infer_shape[1]
        height_ratio = origin_shape[0] / infer_shape[0]
        new_polys = scale_polygons(
            polygons=polys,
            x_scale=width_ratio,
            y_scale=height_ratio,
        )
    elif resize_method in {
        "Fit (black edges) in",
        "Fit (white edges) in",
        "Fit (grey edges) in",
    }:
        new_polys = undo_image_padding_for_predicted_polygons(
            polygons=polys,
            infer_shape=infer_shape,
            origin_shape=origin_shape,
        )
    shifted_polys = []
    for poly in new_polys:
        poly = [(p[0] + crop_shift_x, p[1] + crop_shift_y) for p in poly]
        shifted_polys.append(poly)
    return shifted_polys


def scale_polygons(
    polygons: List[List[Tuple[float, float]]],
    x_scale: float,
    y_scale: float,
) -> List[List[Tuple[float, float]]]:
    result = []
    for poly in polygons:
        poly = [(p[0] * x_scale, p[1] * y_scale) for p in poly]
        result.append(poly)
    return result


def undo_image_padding_for_predicted_polygons(
    polygons: List[List[Tuple[float, float]]],
    origin_shape: Tuple[int, int],
    infer_shape: Tuple[int, int],
) -> List[List[Tuple[float, float]]]:
    scale = min(infer_shape[0] / origin_shape[0], infer_shape[1] / origin_shape[1])
    inter_w = int(origin_shape[1] * scale)
    inter_h = int(origin_shape[0] * scale)
    pad_x = (infer_shape[1] - inter_w) / 2
    pad_y = (infer_shape[0] - inter_h) / 2
    result = []
    for poly in polygons:
        poly = [((p[0] - pad_x) / scale, (p[1] - pad_y) / scale) for p in poly]
        result.append(poly)
    return result


def get_static_crop_dimensions(
    orig_shape: Tuple[int, int],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Generates a transformation based on preprocessing configuration.

    Args:
        orig_shape (tuple): The original shape of the object (e.g., image) - (height, width).
        preproc (dict): Preprocessing configuration dictionary, containing information such as static cropping.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

    Returns:
        tuple: A tuple containing the shift in the x and y directions, and the updated original shape after cropping.
    """
    try:
        if static_crop_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        ):
            x_min, y_min, x_max, y_max = standardise_static_crop(
                static_crop_config=preproc[STATIC_CROP_KEY]
            )
        else:
            x_min, y_min, x_max, y_max = 0, 0, 1, 1
        crop_shift_x, crop_shift_y = (
            round(x_min * orig_shape[1]),
            round(y_min * orig_shape[0]),
        )
        cropped_percent_x = x_max - x_min
        cropped_percent_y = y_max - y_min
        orig_shape = (
            round(orig_shape[0] * cropped_percent_y),
            round(orig_shape[1] * cropped_percent_x),
        )
        return (crop_shift_x, crop_shift_y), orig_shape
    except KeyError as error:
        raise PostProcessingError(
            f"Could not find a proper configuration key {error} in post-processing."
        )


def standardise_static_crop(
    static_crop_config: Dict[str, int],
) -> Tuple[float, float, float, float]:
    return tuple(static_crop_config[key] / 100 for key in ["x_min", "y_min", "x_max", "y_max"])  # type: ignore


def post_process_keypoints(
    predictions: List[List[List[float]]],
    keypoints_start_index: int,
    infer_shape: Tuple[int, int],
    img_dims: List[Tuple[int, int]],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
    resize_method: str = "Stretch to",
) -> List[List[List[float]]]:
    """Scales and shifts keypoints based on the given image shapes and preprocessing method.

    This function performs polygon scaling and shifting based on the specified resizing method and
    pre-processing steps. The polygons are transformed according to the ratio and padding between two images.

    Args:
        predictions: predictions from model
        keypoints_start_index: offset in the 3rd dimension pointing where in the prediction start keypoints [(x, y, cfg), ...] for each keypoint class
        img_dims list of (tuple of int): Shape of the source image (height, width).
        infer_shape (tuple of int): Shape of the target image (height, width).
        preproc (object): Preprocessing details used for generating the transformation.
        resize_method (str, optional): Resizing method, either "Stretch to", "Fit (black edges) in", "Fit (white edges) in", or "Fit (grey edges) in". Defaults to "Stretch to".
        disable_preproc_static_crop: flag to disable static crop
    Returns:
        list of list of list: predictions with post-processed keypoints
    """
    # Get static crop params
    scaled_predictions = []
    # Loop through batches
    for i, batch_predictions in enumerate(predictions):
        if len(batch_predictions) == 0:
            scaled_predictions.append([])
            continue
        np_batch_predictions = np.array(batch_predictions)
        keypoints = np_batch_predictions[:, keypoints_start_index:]
        (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
            img_dims[i],
            preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        if resize_method == "Stretch to":
            keypoints = stretch_keypoints(
                keypoints=keypoints,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        elif (
            resize_method == "Fit (black edges) in"
            or resize_method == "Fit (white edges) in"
            or resize_method == "Fit (grey edges) in"
        ):
            keypoints = undo_image_padding_for_predicted_keypoints(
                keypoints=keypoints,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        keypoints = clip_keypoints_coordinates(
            keypoints=keypoints, origin_shape=origin_shape
        )
        keypoints = shift_keypoints(
            keypoints=keypoints, shift_x=crop_shift_x, shift_y=crop_shift_y
        )
        np_batch_predictions[:, keypoints_start_index:] = keypoints
        scaled_predictions.append(np_batch_predictions.tolist())
    return scaled_predictions


def stretch_keypoints(
    keypoints: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    scale_width = origin_shape[1] / infer_shape[1]
    scale_height = origin_shape[0] / infer_shape[0]
    for keypoint_id in range(keypoints.shape[1] // 3):
        keypoints[:, keypoint_id * 3] *= scale_width
        keypoints[:, keypoint_id * 3 + 1] *= scale_height
    return keypoints


def undo_image_padding_for_predicted_keypoints(
    keypoints: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    # Undo scaling and padding from letterbox resize preproc operation
    scale = min(infer_shape[0] / origin_shape[0], infer_shape[1] / origin_shape[1])
    inter_w = int(origin_shape[1] * scale)
    inter_h = int(origin_shape[0] * scale)

    pad_x = (infer_shape[1] - inter_w) / 2
    pad_y = (infer_shape[0] - inter_h) / 2
    for coord_id in range(keypoints.shape[1] // 3):
        keypoints[:, coord_id * 3] -= pad_x
        keypoints[:, coord_id * 3] /= scale
        keypoints[:, coord_id * 3 + 1] -= pad_y
        keypoints[:, coord_id * 3 + 1] /= scale
    return keypoints


def clip_keypoints_coordinates(
    keypoints: np.ndarray,
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    for keypoint_id in range(keypoints.shape[1] // 3):
        keypoints[:, keypoint_id * 3] = np.round(
            np.clip(keypoints[:, keypoint_id * 3], a_min=0, a_max=origin_shape[1])
        )
        keypoints[:, keypoint_id * 3 + 1] = np.round(
            np.clip(keypoints[:, keypoint_id * 3 + 1], a_min=0, a_max=origin_shape[0])
        )
    return keypoints


def shift_keypoints(
    keypoints: np.ndarray,
    shift_x: Union[int, float],
    shift_y: Union[int, float],
) -> np.ndarray:
    for keypoint_id in range(keypoints.shape[1] // 3):
        keypoints[:, keypoint_id * 3] += shift_x
        keypoints[:, keypoint_id * 3 + 1] += shift_y
    return keypoints


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.number, np.ndarray]:
    """Computes the sigmoid function for the given input.

    The sigmoid function is defined as:
    f(x) = 1 / (1 + exp(-x))

    Args:
        x (float or numpy.ndarray): Input value or array for which the sigmoid function is to be computed.

    Returns:
        float or numpy.ndarray: The computed sigmoid value(s).
    """
    return 1 / (1 + np.exp(-x))
