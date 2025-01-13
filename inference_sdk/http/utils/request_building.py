from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from inference_sdk.http.utils.iterables import make_batches
from inference_sdk.http.utils.requests import inject_images_into_payload


class ImagePlacement(Enum):
    DATA = "data"
    JSON = "json"


@dataclass(frozen=True)
class RequestData:
    """Data class for request data.

    Attributes:
        url: The URL of the request.
        request_elements: The number of request elements.
        headers: The headers of the request.
        parameters: The parameters of the request.
        data: The data of the request.
        payload: The payload of the request.
        image_scaling_factors: The scaling factors of the images.
    """

    url: str
    request_elements: int
    headers: Optional[Dict[str, str]]
    parameters: Optional[Dict[str, Union[str, List[str]]]]
    data: Optional[Union[str, bytes]]
    payload: Optional[Dict[str, Any]]
    image_scaling_factors: List[Optional[float]]


def prepare_requests_data(
    url: str,
    encoded_inference_inputs: List[Tuple[str, Optional[float]]],
    headers: Optional[Dict[str, str]],
    parameters: Optional[Dict[str, Union[str, List[str]]]],
    payload: Optional[Dict[str, Any]],
    max_batch_size: int,
    image_placement: ImagePlacement,
) -> List[RequestData]:
    """Prepare requests data.

    Args:
        url: The URL of the request.
        encoded_inference_inputs: The encoded inference inputs.
        headers: The headers of the request.
        parameters: The parameters of the request.
        payload: The payload of the request.
        max_batch_size: The maximum batch size.
        image_placement: The image placement.

    Returns:
        The list of request data.
    """
    batches = list(
        make_batches(
            iterable=encoded_inference_inputs,
            batch_size=max_batch_size,
        )
    )
    requests_data = []
    for batch_inference_inputs in batches:
        request_data = assembly_request_data(
            url=url,
            batch_inference_inputs=batch_inference_inputs,
            headers=headers,
            parameters=parameters,
            payload=payload,
            image_placement=image_placement,
        )
        requests_data.append(request_data)
    return requests_data


def assembly_request_data(
    url: str,
    batch_inference_inputs: List[Tuple[str, Optional[float]]],
    headers: Optional[Dict[str, str]],
    parameters: Optional[Dict[str, Union[str, List[str]]]],
    payload: Optional[Dict[str, Any]],
    image_placement: ImagePlacement,
) -> RequestData:
    """Assemble request data.

    Args:
        url: The URL of the request.
        batch_inference_inputs: The batch inference inputs.
        headers: The headers of the request.
        parameters: The parameters of the request.
        payload: The payload of the request.
        image_placement: The image placement.

    Returns:
        The request data.
    """
    data = None
    if image_placement is ImagePlacement.DATA and len(batch_inference_inputs) != 1:
        raise ValueError("Only single image can be placed in request `data`")
    if image_placement is ImagePlacement.JSON and payload is None:
        payload = {}
    if image_placement is ImagePlacement.JSON:
        payload = deepcopy(payload)
        payload = inject_images_into_payload(
            payload=payload,
            encoded_images=batch_inference_inputs,
        )
    elif image_placement is ImagePlacement.DATA:
        data = batch_inference_inputs[0][0]
    else:
        raise NotImplemented(
            f"Not implemented request building method for {image_placement}"
        )
    scaling_factors = [e[1] for e in batch_inference_inputs]
    return RequestData(
        url=url,
        request_elements=len(batch_inference_inputs),
        headers=headers,
        parameters=parameters,
        data=data,
        payload=payload,
        image_scaling_factors=scaling_factors,
    )
