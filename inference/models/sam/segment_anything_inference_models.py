import base64
from io import BytesIO
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from shapely.geometry import Polygon as ShapelyPolygon

from inference.core.entities.requests import (
    SamEmbeddingRequest,
    SamInferenceRequest,
    SamSegmentationRequest,
)
from inference.core.entities.responses import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    SAM_MAX_EMBEDDING_CACHE_SIZE,
    SAM_VERSION_ID,
)
from inference.core.models.base import Model
from inference.core.models.inference_models_adapters import (
    get_extra_weights_provider_headers,
)
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.postprocess import masks2poly
from inference_models import AutoModel
from inference_models.models.sam.cache import (
    SamImageEmbeddingsInMemoryCache,
    SamLowResolutionMasksInMemoryCache,
)
from inference_models.models.sam.entities import SAMImageEmbeddings
from inference_models.models.sam.sam_torch import SAMTorch, compute_image_hash

MASK_THRESHOLD = 0.0


class InferenceModelsSAMAdapter(Model):
    def __init__(
        self,
        model_id: str = f"sam/{SAM_VERSION_ID}",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "unsupervised-segmentation"

        sam_image_embeddings_cache = SamImageEmbeddingsInMemoryCache.init(
            size_limit=SAM_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=True,
        )
        sam_low_resolution_masks_cache = SamLowResolutionMasksInMemoryCache.init(
            size_limit=SAM_MAX_EMBEDDING_CACHE_SIZE,
            send_to_cpu=True,
        )
        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: SAMTorch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            sam_image_embeddings_cache=sam_image_embeddings_cache,
            sam_low_resolution_masks_cache=sam_low_resolution_masks_cache,
            sam_allow_client_generated_hash_ids=True,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def infer_from_request(self, request: SamInferenceRequest):
        t1 = perf_counter()
        if isinstance(request, SamEmbeddingRequest):
            embedding, _ = self.embed_image(**request.dict())
            inference_time = perf_counter() - t1
            if request.format == "json":
                return SamEmbeddingResponse(
                    embeddings=embedding.tolist(), time=inference_time
                )
            else:
                binary_vector = BytesIO()
                np.save(binary_vector, embedding)
                binary_vector.seek(0)
                return SamEmbeddingResponse(
                    embeddings=binary_vector.getvalue(), time=inference_time
                )
        elif isinstance(request, SamSegmentationRequest):
            masks, low_res_masks = self.segment_image(**request.dict())
            if request.format == "json":
                masks = masks > MASK_THRESHOLD
                masks = masks2poly(masks)
                low_res_masks = low_res_masks > MASK_THRESHOLD
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

            response = SamSegmentationResponse(
                masks=[m.tolist() for m in masks],
                low_res_masks=[m.tolist() for m in low_res_masks],
                time=perf_counter() - t1,
            )
            return response

    def embed_image(self, image: Any, image_id: Optional[str] = None, **kwargs):
        # yeah, this is reverse of control around hash_id generation
        # we let clients running what the flow of the hash_id generation is,
        # which should not be the case - as single client may effectively
        # override state of the cache for other clients - letting it be only for
        # the sake of interface compatibility for inference 1.0 - moving forward
        # my recommendation is to remove that.
        loaded_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        enbeddings = self._model.embed_images(
            images=loaded_image, image_hashes=image_id, **kwargs
        )[0]
        return enbeddings.embeddings.cpu().numpy(), enbeddings.image_size_hw

    def segment_image(
        self,
        image: Any,
        embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None,
        embeddings_format: Optional[str] = "json",
        has_mask_input: Optional[bool] = False,
        image_id: Optional[str] = None,
        mask_input: Optional[Union[np.ndarray, List[List[List[float]]]]] = None,
        mask_input_format: Optional[str] = "json",
        orig_im_size: Optional[List[int]] = None,
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        use_mask_input_cache: Optional[bool] = True,
        use_embeddings_cache: Optional[bool] = True,
        **kwargs,
    ):
        # yeah, this is reverse of control around hash_id generation
        # we let clients running what the flow of the hash_id generation is,
        # which should not be the case - as single client may effectively
        # override state of the cache for other clients - letting it be only for
        # the sake of interface compatibility for inference 1.0 - moving forward
        # my recommendation is to remove that.
        loaded_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        if embeddings is not None:
            if embeddings_format == "json":
                embeddings = np.array(embeddings)
            elif embeddings_format == "binary":
                embeddings = np.load(BytesIO(embeddings))
            elif isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            if image_id is not None:
                image_hash = image_id
            else:
                image_hash = compute_image_hash(image=loaded_image)
            image_size_hw: Tuple[int, int] = (
                loaded_image.shape[0],
                loaded_image.shape[1],
            )
            embeddings = SAMImageEmbeddings(
                image_hash=image_hash,
                image_size_hw=image_size_hw,
                embeddings=torch.from_numpy(embeddings),
            )
        if point_coords is not None:
            point_coords = np.array(point_coords)
        if point_labels is not None:
            point_labels = np.array(point_labels)
        if has_mask_input:
            if mask_input is not None:
                if mask_input_format == "json":
                    polys = mask_input
                    mask_input = np.zeros((1, len(polys), 256, 256), dtype=np.uint8)
                    for i, poly in enumerate(polys):
                        poly = ShapelyPolygon(poly)
                        raster = rasterio.features.rasterize(
                            [poly], out_shape=(256, 256)
                        )
                        mask_input[0, i, :, :] = raster
                elif mask_input_format == "binary":
                    binary_data = base64.b64decode(mask_input)
                    mask_input = np.load(BytesIO(binary_data))
                elif isinstance(mask_input, list):
                    mask_input = np.array(mask_input)

        predictions = self._model.segment_images(
            images=loaded_image,
            embeddings=embeddings,
            image_hashes=image_id,
            point_coordinates=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multi_mask_output=False,
            enforce_mask_input=has_mask_input,
            use_mask_input_cache=use_mask_input_cache,
            use_embeddings_cache=use_embeddings_cache,
        )
        return predictions[0].masks.cpu().numpy(), predictions[0].scores.cpu().numpy()

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass
