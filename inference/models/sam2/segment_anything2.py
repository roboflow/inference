import base64
import hashlib
import logging
import os
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio.features
import torch

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except:
    logging.error(
        "Could not import sam2. See the instructions at "
        "https://github.com/facebookresearch/segment-anything-2/?tab=readme-ov-file#installation"
    )
    raise
from shapely.geometry import Polygon as ShapelyPolygon

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2InferenceRequest,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationResponse,
)
from inference.core.env import SAM2_VERSION_ID, SAM_MAX_EMBEDDING_CACHE_SIZE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2poly


class SegmentAnything2(RoboflowCoreModel):
    """SegmentAnything class for handling segmentation tasks.

    Attributes:
        sam: The segmentation model.
        predictor: The predictor for the segmentation model.
        ort_session: ONNX runtime inference session.
        embedding_cache: Cache for embeddings.
        image_size_cache: Cache for image sizes.
        embedding_cache_keys: Keys for the embedding cache.
        low_res_logits_cache: Cache for low resolution logits.
        segmentation_cache_keys: Keys for the segmentation cache.
    """

    def __init__(self, *args, model_id: str = f"sam2/{SAM2_VERSION_ID}", **kwargs):
        """Initializes the SegmentAnything.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id=model_id, **kwargs)
        checkpoint = self.cache_file("weights.pt")
        model_cfg = {
            "hiera_large": "sam2_hiera_l.yaml",
            "hiera_small": "sam2_hiera_s.yaml",
            "hiera_tiny": "sam2_hiera_t.yaml",
            "hiera_b_plus": "sam2_hiera_b+.yaml",
        }[self.version_id]

        self.sam = build_sam2(model_cfg, checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam.to(self.device)

        self.predictor = SAM2ImagePredictor(self.sam)

        self.embedding_cache = {}
        self.image_size_cache = {}
        self.embedding_cache_keys = []

        self.low_res_logits_cache = {}
        self.segmentation_cache_keys = []
        self.task_type = "unsupervised-segmentation"

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: List of file names.
        """
        return ["weights.pt"]

    def embed_image(self, image: Any, image_id: Optional[str] = None, **kwargs):
        """
        Embeds an image and caches the result if an image_id is provided. If the image has been embedded before and cached,
        the cached result will be returned.

        Args:
            image (Any): The image to be embedded. The format should be compatible with the preproc_image method.
            image_id (Optional[str]): An identifier for the image. If provided, the embedding result will be cached
                                      with this ID. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple where the first element is the embedding of the image
                                               and the second element is the shape (height, width) of the processed image.

        Notes:
            - Embeddings and image sizes are cached to improve performance on repeated requests for the same image.
            - The cache has a maximum size defined by SAM_MAX_EMBEDDING_CACHE_SIZE. When the cache exceeds this size,
              the oldest entries are removed.

        Example:
            >>> img_array = ... # some image array
            >>> embed_image(img_array, image_id="sample123")
            (array([...]), (224, 224))
        """
        if image_id and image_id in self.embedding_cache:
            return (
                self.embedding_cache[image_id],
                self.image_size_cache[image_id],
                image_id,
            )

        img_in = self.preproc_image(image)
        if image_id is None:
            image_id = hashlib.md5(img_in.tobytes()).hexdigest()[:12]

        if image_id and image_id in self.embedding_cache:
            return (
                self.embedding_cache[image_id],
                self.image_size_cache[image_id],
                image_id,
            )

        with torch.inference_mode():
            self.predictor.set_image(img_in)
            embedding_dict = self.predictor._features

        self.embedding_cache[image_id] = embedding_dict
        self.image_size_cache[image_id] = img_in.shape[:2]
        self.embedding_cache_keys.append(image_id)
        if len(self.embedding_cache_keys) > SAM_MAX_EMBEDDING_CACHE_SIZE:
            cache_key = self.embedding_cache_keys.pop(0)
            del self.embedding_cache[cache_key]
            del self.image_size_cache[cache_key]
        return (embedding_dict, img_in.shape[:2], image_id)

    def infer_from_request(self, request: Sam2InferenceRequest):
        """Performs inference based on the request type.

        Args:
            request (SamInferenceRequest): The inference request.

        Returns:
            Union[SamEmbeddingResponse, SamSegmentationResponse]: The inference response.
        """
        t1 = perf_counter()
        if isinstance(request, Sam2EmbeddingRequest):
            _, _, image_id = self.embed_image(**request.dict())
            inference_time = perf_counter() - t1
            return Sam2EmbeddingResponse(time=inference_time, image_id=image_id)
        elif isinstance(request, Sam2SegmentationRequest):
            masks, low_res_masks = self.segment_image(**request.dict())
            if request.format == "json":
                masks = masks > self.predictor.mask_threshold
                masks = masks2poly(masks)
                low_res_masks = low_res_masks > self.predictor.mask_threshold
                low_res_masks = masks2poly(low_res_masks)
            elif request.format == "binary":
                binary_vector = BytesIO()
                np.savez_compressed(
                    binary_vector, masks=masks, low_res_masks=low_res_masks
                )
                binary_vector.seek(0)
                binary_data = binary_vector.getvalue()
                return binary_data
            else:
                raise ValueError(f"Invalid format {request.format}")

            response = Sam2SegmentationResponse(
                masks=[m.tolist() for m in masks],
                low_res_masks=[m.tolist() for m in low_res_masks],
                time=perf_counter() - t1,
            )
            return response

    def preproc_image(self, image: InferenceRequestImage):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_rgb(image)
        return np_image

    def segment_image(
        self,
        image: Any,
        image_id: Optional[str] = None,
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Segments an image based on provided embeddings, points, masks, or cached results.
        If embeddings are not directly provided, the function can derive them from the input image or cache.

        Args:
            image (Any): The image to be segmented.
            embeddings (Optional[Union[np.ndarray, List[List[float]]]]): The embeddings of the image.
                Defaults to None, in which case the image is used to compute embeddings.
            embeddings_format (Optional[str]): Format of the provided embeddings; either 'json' or 'binary'. Defaults to 'json'.
            has_mask_input (Optional[bool]): Specifies whether mask input is provided. Defaults to False.
            image_id (Optional[str]): A cached identifier for the image. Useful for accessing cached embeddings or masks.
            mask_input (Optional[Union[np.ndarray, List[List[List[float]]]]]): Input mask for the image.
            mask_input_format (Optional[str]): Format of the provided mask input; either 'json' or 'binary'. Defaults to 'json'.
            orig_im_size (Optional[List[int]]): Original size of the image when providing embeddings directly.
            point_coords (Optional[List[List[float]]]): Coordinates of points in the image. Defaults to an empty list.
            point_labels (Optional[List[int]]): Labels associated with the provided points. Defaults to an empty list.
            use_mask_input_cache (Optional[bool]): Flag to determine if cached mask input should be used. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple where the first element is the segmentation masks of the image
                                          and the second element is the low resolution segmentation masks.

        Raises:
            ValueError: If necessary inputs are missing or inconsistent.

        Notes:
            - Embeddings, segmentations, and low-resolution logits can be cached to improve performance
              on repeated requests for the same image.
            - The cache has a maximum size defined by SAM_MAX_EMBEDDING_CACHE_SIZE. When the cache exceeds this size,
              the oldest entries are removed.
        """
        with torch.inference_mode():
            if not image and not image_id:
                raise ValueError("Must provide either image or  cached image_id")
            elif image_id and not image and image_id not in self.embedding_cache:
                raise ValueError(
                    f"Image ID {image_id} not in embedding cache, must provide the image or embeddings"
                )
            embedding, original_image_size, image_id = self.embed_image(
                image=image, image_id=image_id
            )

            if point_coords is not None:
                point_coords = np.array(point_coords, dtype=np.float32)
                point_coords = np.expand_dims(point_coords, axis=0)

            if point_labels is not None:
                point_labels = np.array(point_labels, dtype=np.float32)
                point_labels = np.expand_dims(point_labels, axis=0)

            mask_input = self.low_res_logits_cache.get(image_id, None)
            self.predictor._is_image_set = True
            self.predictor._features = embedding
            self.predictor._orig_hw = [original_image_size]
            self.predictor._is_batch = False

            masks, scores, low_res_logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=(
                    np.expand_dims(mask_input, axis=0).astype(np.float32)
                    if mask_input is not None
                    else None
                ),
                multimask_output=True,
                return_logits=True,
                normalize_coords=True,
            )

            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            low_res_logits = low_res_logits[sorted_ind]

            self.low_res_logits_cache[image_id] = low_res_logits[0]
            if image_id not in self.segmentation_cache_keys:
                self.segmentation_cache_keys.append(image_id)
            if len(self.segmentation_cache_keys) > SAM_MAX_EMBEDDING_CACHE_SIZE:
                cache_key = self.segmentation_cache_keys.pop(0)
                del self.low_res_logits_cache[cache_key]
            masks = masks[0]
            low_res_masks = low_res_logits[0]

            return masks, low_res_masks
