import base64
from io import BytesIO
from time import perf_counter
from typing import List

import numpy as np
import onnxruntime
import rasterio.features
import torch
from segment_anything import SamPredictor, sam_model_registry
from shapely.geometry import Polygon as ShapelyPolygon

from inference.core.data_models import (
    InferenceRequestImage,
    SamEmbeddingRequest,
    SamEmbeddingResponse,
    SamInferenceRequest,
    SamSegmentationRequest,
    SamSegmentationResponse,
)
from inference.core.env import SAM_MAX_EMBEDDING_CACHE_SIZE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.postprocess import mask2poly


class SegmentAnythingRoboflowCoreModel(RoboflowCoreModel):
    """SegmentAnythingRoboflowCoreModel class for handling segmentation tasks.

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

    def __init__(self, *args, **kwargs):
        """Initializes the SegmentAnythingRoboflowCoreModel.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.sam = sam_model_registry[self.version_id](
            checkpoint=self.cache_file("encoder.pth")
        )
        self.sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.sam)
        self.ort_session = onnxruntime.InferenceSession(
            self.cache_file("decoder.onnx"),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.embedding_cache = {}
        self.image_size_cache = {}
        self.embedding_cache_keys = []

        self.low_res_logits_cache = {}
        self.segmentation_cache_keys = []

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: List of file names.
        """
        return ["encoder.pth", "decoder.onnx"]

    def embed_image_(self, request: SamEmbeddingRequest):
        """Embeds an image.

        Args:
            request (SamEmbeddingRequest): The embedding request.

        Returns:
            Tuple: The embedding and the shape of the image.
        """
        if request.image_id and request.image_id in self.embedding_cache:
            return (
                self.embedding_cache[request.image_id],
                self.image_size_cache[request.image_id],
            )
        img_in = self.preproc_image(request.image)
        self.predictor.set_image(img_in)
        embedding = self.predictor.get_image_embedding().cpu().numpy()
        if request.image_id:
            self.embedding_cache[request.image_id] = embedding
            self.image_size_cache[request.image_id] = img_in.shape[:2]
            self.embedding_cache_keys.append(request.image_id)
            if len(self.embedding_cache_keys) > SAM_MAX_EMBEDDING_CACHE_SIZE:
                cache_key = self.embedding_cache_keys.pop(0)
                del self.embedding_cache[cache_key]
                del self.image_size_cache[cache_key]
        return (embedding, img_in.shape[:2])

    def embed_image(self, request: SamEmbeddingRequest):
        """Embeds an image and returns the response.

        Args:
            request (SamEmbeddingRequest): The embedding request.

        Returns:
            SamEmbeddingResponse: The embedding response.
        """
        t1 = perf_counter()
        embedding, _ = self.embed_image_(request)
        inference_time = perf_counter() - t1
        if request.format == "json":
            return SamEmbeddingResponse(
                embeddings=embedding.tolist(), time=inference_time
            )
        elif request.format == "binary":
            binary_vector = BytesIO()
            np.save(binary_vector, embedding)
            binary_vector.seek(0)
            return SamEmbeddingResponse(
                embeddings=binary_vector.getvalue(), time=inference_time
            )

    def infer(self, request: SamInferenceRequest):
        """Performs inference based on the request type.

        Args:
            request (SamInferenceRequest): The inference request.

        Returns:
            Union[SamEmbeddingResponse, SamSegmentationResponse]: The inference response.
        """
        if isinstance(request, SamEmbeddingRequest):
            return self.embed_image(request)
        elif isinstance(request, SamSegmentationRequest):
            return self.segment_image(request)

    def preproc_image(self, image: InferenceRequestImage):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        pil_image = self.load_image(image.type, image.value).convert("RGB")
        return np.array(pil_image)

    def segment_image(self, request: SamSegmentationRequest):
        """Segments an image.

        Args:
            request (SamSegmentationRequest): The segmentation request.

        Returns:
            SamSegmentationResponse: The segmentation response.
        """
        t1 = perf_counter()
        if not request.embeddings:
            if not request.image and not request.image_id:
                raise ValueError(
                    "Must provide either image, cached image_id, or embeddings"
                )
            elif (
                request.image_id
                and not request.image
                and request.image_id not in self.embedding_cache
            ):
                raise ValueError(
                    f"Image ID {request.image_id} not in embedding cache, must provide the image or embeddings"
                )
            embedding_request = SamEmbeddingRequest(
                image=request.image, image_id=request.image_id
            )
            embedding, original_image_size = self.embed_image_(embedding_request)
        else:
            if not request.orig_im_size:
                raise ValueError(
                    "Must provide original image size if providing embeddings"
                )
            original_image_size = request.orig_im_size
            if request.embeddings_format == "json":
                embedding = np.array(request.embeddings)
            elif request.embeddings_format == "binary":
                embedding = np.load(BytesIO(request.embeddings))

        point_coords = request.point_coords
        point_coords.append([0, 0])
        point_coords = np.array(point_coords, dtype=np.float32)
        point_coords = np.expand_dims(point_coords, axis=0)
        point_coords = self.predictor.transform.apply_coords(
            point_coords,
            original_image_size,
        )

        point_labels = request.point_labels
        point_labels.append(-1)
        point_labels = np.array(point_labels, dtype=np.float32)
        point_labels = np.expand_dims(point_labels, axis=0)

        if request.has_mask_input:
            if (
                request.image_id
                and request.image_id in self.low_res_logits_cache
                and request.use_mask_input_cache
            ):
                mask_input = self.low_res_logits_cache[request.image_id]
            elif not request.mask_input and (
                not request.image_id
                or request.image_id not in self.low_res_logits_cache
            ):
                raise ValueError("Must provide either mask_input or cached image_id")
            else:
                if request.mask_input_format == "json":
                    polys = request.mask_input
                    mask_input = np.zeros((1, len(polys), 256, 256), dtype=np.uint8)
                    for i, poly in enumerate(polys):
                        poly = ShapelyPolygon(poly)
                        raster = rasterio.features.rasterize(
                            [poly], out_shape=(256, 256)
                        )
                        mask_input[0, i, :, :] = raster
                elif request.mask_input_format == "binary":
                    binary_data = base64.b64decode(request.mask_input)
                    mask_input = np.load(BytesIO(binary_data))
        else:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

        ort_inputs = {
            "image_embeddings": embedding.astype(np.float32),
            "point_coords": point_coords.astype(np.float32),
            "point_labels": point_labels,
            "mask_input": mask_input.astype(np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32)
            if not request.has_mask_input
            else np.ones(1, dtype=np.float32),
            "orig_im_size": np.array(original_image_size, dtype=np.float32),
        }
        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        if request.image_id:
            self.low_res_logits_cache[request.image_id] = low_res_logits
            if request.image_id not in self.segmentation_cache_keys:
                self.segmentation_cache_keys.append(request.image_id)
            if len(self.segmentation_cache_keys) > SAM_MAX_EMBEDDING_CACHE_SIZE:
                cache_key = self.segmentation_cache_keys.pop(0)
                del self.low_res_logits_cache[cache_key]
        masks = masks[0]
        low_res_masks = low_res_logits[0]

        if request.format == "json":
            masks = masks > self.predictor.model.mask_threshold
            masks = mask2poly(masks, False)
            low_res_masks = low_res_masks > self.predictor.model.mask_threshold
            low_res_masks = mask2poly(low_res_masks, False)
        elif request.format == "binary":
            binary_vector = BytesIO()
            np.savez_compressed(binary_vector, masks=masks, low_res_masks=low_res_masks)
            binary_vector.seek(0)
            binary_data = binary_vector.getvalue()
            return binary_data
        else:
            raise ValueError(f"Invalid format {request.format}")

        response = SamSegmentationResponse(
            masks=masks,
            low_res_masks=low_res_masks,
            time=perf_counter() - t1,
        )
        return response
