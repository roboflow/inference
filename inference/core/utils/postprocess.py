import multiprocessing
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from time import perf_counter
from typing import List, Tuple

import cv2
import numpy as np

from inference.core.env import DISABLE_PREPROC_STATIC_CROP


def clip_boxes(boxes, shape):
    """
    Clip the bounding box coordinates to lie within the image dimensions.

    Args:
        boxes (np.ndarray): An array of bounding boxes, shape (n, 4).
        shape (Tuple[int, int]): The shape of the image (height, width).
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_point(point, shape):
    """
    Clip a point to lie within the given shape.

    Args:
        point (Tuple[int, int]): The coordinates of the point (x, y).
        shape (Tuple[int, int]): The shape of the region (width, height).

    Returns:
        Tuple[int, int]: The clipped coordinates of the point.
    """
    x = min(point[0], shape[0])
    x = max(x, 0)
    y = min(point[1], shape[1])
    y = max(y, 0)
    return (x, y)


def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector A.
        b (np.ndarray): Vector B.

    Returns:
        float: Cosine similarity between vectors A and B.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def crop_mask(masks, boxes):
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


def mask2poly_(mask):
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


def mask2poly(masks):
    """Converts binary masks to polygonal segments.

    Args:
        masks (numpy.ndarray): A set of binary masks, where masks are multiplied by 255 and converted to uint8 type.

    Returns:
        list: A list of segments, where each segment is obtained by converting the corresponding mask.
    """
    segments = []
    masks = (masks * 255.0).astype(np.uint8)
    for mask in masks:
        segments.append(mask2poly_(mask))
    return segments


def generate_transform_from_proc(
    orig_shape,
    preproc,
    disable_preproc_static_crop: bool = False,
):
    """
    Generates a transformation based on preprocessing configuration.

    Args:
        orig_shape (tuple): The original shape of the object (e.g., image).
        preproc (dict): Preprocessing configuration dictionary, containing information such as static cropping.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

    Returns:
        tuple: A tuple containing the shift in the x and y directions, and the updated original shape after cropping.
    """
    if (
        "static-crop" in preproc.keys()
        and not DISABLE_PREPROC_STATIC_CROP
        and not disable_preproc_static_crop
        and preproc["static-crop"]["enabled"] == True
    ):
        x_min = preproc["static-crop"]["x_min"] / 100
        y_min = preproc["static-crop"]["y_min"] / 100
        x_max = preproc["static-crop"]["x_max"] / 100
        y_max = preproc["static-crop"]["y_max"] / 100
    else:
        x_min = 0
        y_min = 0
        x_max = 1
        y_max = 1
    crop_shift_x, crop_shift_y = (
        int(x_min * orig_shape[0]),
        int(y_min * orig_shape[1]),
    )
    cropped_percent_x = x_max - x_min
    cropped_percent_y = y_max - y_min
    orig_shape = (
        orig_shape[0] * cropped_percent_x,
        orig_shape[1] * cropped_percent_y,
    )
    return (crop_shift_x, crop_shift_y), orig_shape


def postprocess_predictions(
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
        img_dims (List[Tuple[int, int]]): The dimensions of the original image for each batch, indices are: batch x [ width, height ].
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
        # Get coords from predictions (x1,y1,x2,y2)
        coords = np_batch_predictions[:, :4]

        # Shape before resize to infer shape
        orig_shape = img_dims[i][-1::-1]
        # Adjust shape and get shift pased on static crop preproc
        (crop_shift_x, crop_shift_y), orig_shape = generate_transform_from_proc(
            orig_shape, preproc, disable_preproc_static_crop=disable_preproc_static_crop
        )
        if resize_method == "Stretch to":
            scale_height = orig_shape[1] / infer_shape[1]
            scale_width = orig_shape[0] / infer_shape[0]
            coords[:, 0] *= scale_width
            coords[:, 2] *= scale_width
            coords[:, 1] *= scale_height
            coords[:, 3] *= scale_height

        elif (
            resize_method == "Fit (black edges) in"
            or resize_method == "Fit (white edges) in"
        ):
            # Undo scaling and padding from letterbox resize preproc operation
            scale = min(infer_shape[0] / orig_shape[0], infer_shape[1] / orig_shape[1])
            inter_w = int(orig_shape[0] * scale)
            inter_h = int(orig_shape[1] * scale)

            pad_x = (infer_shape[0] - inter_w) / 2
            pad_y = (infer_shape[1] - inter_h) / 2

            coords[:, 0] -= pad_x
            coords[:, 2] -= pad_x
            coords[:, 1] -= pad_y
            coords[:, 3] -= pad_y

            coords /= scale

        coords[:, 0] = np.round_(
            np.clip(coords[:, 0], a_min=0, a_max=orig_shape[0])
        ).astype(int)
        coords[:, 2] = np.round_(
            np.clip(coords[:, 2], a_min=0, a_max=orig_shape[0])
        ).astype(int)
        coords[:, 1] = np.round_(
            np.clip(coords[:, 1], a_min=0, a_max=orig_shape[1])
        ).astype(int)
        coords[:, 3] = np.round_(
            np.clip(coords[:, 3], a_min=0, a_max=orig_shape[1])
        ).astype(int)

        # Apply static crop prostproc shift
        coords[:, 0] += crop_shift_x
        coords[:, 2] += crop_shift_x
        coords[:, 1] += crop_shift_y
        coords[:, 3] += crop_shift_y

        np_batch_predictions[:, :4] = coords
        scaled_predictions.append(np_batch_predictions.tolist())
    return scaled_predictions


def process_mask_accurate(protos, masks_in, bboxes, shape):
    """Returns masks that are the size of the original image.

    Args:
        protos (numpy.ndarray): Prototype masks.
        masks_in (numpy.ndarray): Input masks.
        bboxes (numpy.ndarray): Bounding boxes.
        shape (tuple): Target shape.

    Returns:
        numpy.ndarray: Processed masks.
    """
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
    masks = masks[:, top:bottom, left:right]

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


def process_mask_tradeoff(protos, masks_in, bboxes, shape, tradeoff_factor):
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
    masks = protos.astype(np.float32)
    masks = masks.reshape((c, -1))
    masks = masks_in @ masks
    masks = sigmoid(masks)
    masks = masks.reshape((-1, mh, mw))
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

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
    downsampled_boxes = deepcopy(bboxes)
    downsampled_boxes[:, 0] *= mw / iw
    downsampled_boxes[:, 2] *= mw / iw
    downsampled_boxes[:, 1] *= mh / ih
    downsampled_boxes[:, 3] *= mh / ih
    masks = crop_mask(masks, downsampled_boxes)
    masks[masks < 0.5] = 0
    return masks


def process_mask_fast(protos, masks_in, bboxes, shape):
    """Returns masks in their original size.

    Args:
        protos (numpy.ndarray): Prototype masks.
        masks_in (numpy.ndarray): Input masks.
        bboxes (numpy.ndarray): Bounding boxes.
        shape (tuple): Target shape.

    Returns:
        numpy.ndarray: Processed masks.
    """
    t1 = perf_counter()
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = protos.astype(np.float32)
    masks = masks.reshape((c, -1))
    masks = masks_in @ masks
    masks = sigmoid(masks)
    masks = masks.reshape((-1, mh, mw))
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    downsampled_boxes = deepcopy(bboxes)
    downsampled_boxes[:, 0] *= mw / iw
    downsampled_boxes[:, 2] *= mw / iw
    downsampled_boxes[:, 1] *= mh / ih
    downsampled_boxes[:, 3] *= mh / ih

    masks = crop_mask(masks, downsampled_boxes)
    masks[masks < 0.5] = 0
    return masks


def scale_boxes(
    img1_shape, boxes, img0_shape, ratio_pad=None, resize_method="Stretch to"
):
    """Scales boxes according to a specified resize method.

    Args:
        img1_shape (tuple): Shape of the first image.
        boxes (numpy.ndarray): Bounding boxes to scale.
        img0_shape (tuple): Shape of the second image.
        ratio_pad (tuple, optional): Padding ratio. Defaults to None.
        resize_method (str, optional): Resize method for image. Defaults to "Stretch to".

    Returns:
        numpy.ndarray: Scaled boxes.
    """
    if resize_method == "Stretch to":
        height_ratio = img0_shape[0] / img1_shape[0]
        width_ratio = img0_shape[1] / img1_shape[1]
        boxes[:, [0, 2]] *= width_ratio
        boxes[:, [1, 3]] *= height_ratio
    elif resize_method in ["Fit (black edges) in", "Fit (white edges) in"]:
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (
                (img1_shape[1] - img0_shape[1] * gain) / 2,
                (img1_shape[0] - img0_shape[0] * gain) / 2,
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_polys(
    img1_shape, polys, img0_shape, preproc, ratio_pad=None, resize_method="Stretch to"
):
    """Scales and shifts polygons based on the given image shapes and preprocessing method.

    This function performs polygon scaling and shifting based on the specified resizing method and
    pre-processing steps. The polygons are transformed according to the ratio and padding between two images.

    Args:
        img1_shape (tuple of int): Shape of the target image (height, width).
        polys (list of list of tuple): List of polygons, where each polygon is represented by a list of (x, y) coordinates.
        img0_shape (tuple of int): Shape of the source image (height, width).
        preproc (object): Preprocessing details used for generating the transformation.
        ratio_pad (tuple, optional): Ratio and padding information for resizing. Defaults to None.
        resize_method (str, optional): Resizing method, either "Stretch to", "Fit (black edges) in", or "Fit (white edges) in". Defaults to "Stretch to".

    Returns:
        list of list of tuple: A list of shifted and scaled polygons.
    """
    (crop_shift_x, crop_shift_y), img0_shape = generate_transform_from_proc(
        img0_shape, preproc
    )
    new_polys = []
    if resize_method == "Stretch to":
        height_ratio = img0_shape[0] / img1_shape[0]
        width_ratio = img0_shape[1] / img1_shape[1]
        for poly in polys:
            poly = [(p[0] * width_ratio, p[1] * height_ratio) for p in poly]
            new_polys.append(poly)
    elif resize_method in ["Fit (black edges) in", "Fit (white edges) in"]:
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (
                (img1_shape[1] - img0_shape[1] * gain) / 2,
                (img1_shape[0] - img0_shape[0] * gain) / 2,
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        for poly in polys:
            poly = [((p[0] - pad[0]) / gain, (p[1] - pad[1]) / gain) for p in poly]
            new_polys.append(poly)
    shifted_polys = []
    for poly in new_polys:
        poly = [(p[0] + crop_shift_x, p[1] + crop_shift_y) for p in poly]
        shifted_polys.append(poly)
    return shifted_polys


def sigmoid(x):
    """Computes the sigmoid function for the given input.

    The sigmoid function is defined as:
    f(x) = 1 / (1 + exp(-x))

    Args:
        x (float or numpy.ndarray): Input value or array for which the sigmoid function is to be computed.

    Returns:
        float or numpy.ndarray: The computed sigmoid value(s).
    """
    return 1 / (1 + np.exp(-x))
