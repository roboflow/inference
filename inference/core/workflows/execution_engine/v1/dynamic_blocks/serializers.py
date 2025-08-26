"""
Serialization utilities for Modal Custom Python Blocks execution.

This module handles converting between workflow objects and JSON-safe
representations for transport to/from Modal sandboxes.
"""

import base64
import json
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import supervision as sv

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult


def serialize_workflow_image(image: WorkflowImageData) -> Dict[str, Any]:
    """Serialize a WorkflowImageData object for transport.
    
    Args:
        image: WorkflowImageData to serialize
        
    Returns:
        JSON-safe dictionary representation
    """
    result = {
        "type": "WorkflowImageData",
        "parent_metadata": image.parent_metadata,
        "workflow_root_ancestor_metadata": image.workflow_root_ancestor_metadata,
    }
    
    # Serialize the numpy image data as base64
    if image.numpy_image is not None:
        _, buffer = cv2.imencode('.png', image.numpy_image)
        result["numpy_image"] = base64.b64encode(buffer).decode('utf-8')
        result["image_shape"] = image.numpy_image.shape
    
    return result


def deserialize_workflow_image(data: Dict[str, Any]) -> WorkflowImageData:
    """Deserialize a WorkflowImageData object from transport format.
    
    Args:
        data: Serialized representation
        
    Returns:
        WorkflowImageData object
    """
    numpy_image = None
    if "numpy_image" in data:
        image_bytes = base64.b64decode(data["numpy_image"])
        nparr = np.frombuffer(image_bytes, np.uint8)
        numpy_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return WorkflowImageData(
        numpy_image=numpy_image,
        parent_metadata=data.get("parent_metadata"),
        workflow_root_ancestor_metadata=data.get("workflow_root_ancestor_metadata"),
    )


def serialize_batch(batch: Batch) -> List[Any]:
    """Serialize a Batch object for transport.
    
    Args:
        batch: Batch to serialize
        
    Returns:
        JSON-safe list representation
    """
    result = []
    for item in batch:
        if isinstance(item, WorkflowImageData):
            result.append(serialize_workflow_image(item))
        elif isinstance(item, np.ndarray):
            result.append(serialize_numpy_array(item))
        elif isinstance(item, sv.Detections):
            result.append(serialize_sv_detections(item))
        else:
            result.append(serialize_value(item))
    return result


def deserialize_batch(data: List[Any]) -> Batch:
    """Deserialize a Batch object from transport format.
    
    Args:
        data: Serialized representation
        
    Returns:
        Batch object
    """
    result = []
    for item in data:
        if isinstance(item, dict):
            if item.get("type") == "WorkflowImageData":
                result.append(deserialize_workflow_image(item))
            elif item.get("type") == "numpy_array":
                result.append(deserialize_numpy_array(item))
            elif item.get("type") == "sv_detections":
                result.append(deserialize_sv_detections(item))
            else:
                result.append(item)
        else:
            result.append(item)
    return Batch(content=result)


def serialize_numpy_array(arr: np.ndarray) -> Dict[str, Any]:
    """Serialize a numpy array for transport.
    
    Args:
        arr: Numpy array to serialize
        
    Returns:
        JSON-safe dictionary representation
    """
    return {
        "type": "numpy_array",
        "data": base64.b64encode(arr.tobytes()).decode('utf-8'),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def deserialize_numpy_array(data: Dict[str, Any]) -> np.ndarray:
    """Deserialize a numpy array from transport format.
    
    Args:
        data: Serialized representation
        
    Returns:
        Numpy array
    """
    bytes_data = base64.b64decode(data["data"])
    arr = np.frombuffer(bytes_data, dtype=np.dtype(data["dtype"]))
    return arr.reshape(data["shape"])


def serialize_sv_detections(detections: sv.Detections) -> Dict[str, Any]:
    """Serialize supervision Detections for transport.
    
    Args:
        detections: Detections object to serialize
        
    Returns:
        JSON-safe dictionary representation
    """
    result = {
        "type": "sv_detections",
        "xyxy": serialize_numpy_array(detections.xyxy) if detections.xyxy is not None else None,
        "mask": serialize_numpy_array(detections.mask) if detections.mask is not None else None,
        "confidence": detections.confidence.tolist() if detections.confidence is not None else None,
        "class_id": detections.class_id.tolist() if detections.class_id is not None else None,
        "tracker_id": detections.tracker_id.tolist() if detections.tracker_id is not None else None,
        "data": {},
    }
    
    # Serialize additional data fields
    for key, value in detections.data.items():
        if isinstance(value, np.ndarray):
            result["data"][key] = serialize_numpy_array(value)
        else:
            result["data"][key] = value
    
    return result


def deserialize_sv_detections(data: Dict[str, Any]) -> sv.Detections:
    """Deserialize supervision Detections from transport format.
    
    Args:
        data: Serialized representation
        
    Returns:
        Detections object
    """
    xyxy = deserialize_numpy_array(data["xyxy"]) if data["xyxy"] else np.empty((0, 4))
    
    detections = sv.Detections(
        xyxy=xyxy,
        mask=deserialize_numpy_array(data["mask"]) if data["mask"] else None,
        confidence=np.array(data["confidence"]) if data["confidence"] else None,
        class_id=np.array(data["class_id"]) if data["class_id"] else None,
        tracker_id=np.array(data["tracker_id"]) if data["tracker_id"] else None,
    )
    
    # Deserialize additional data fields
    for key, value in data.get("data", {}).items():
        if isinstance(value, dict) and value.get("type") == "numpy_array":
            detections.data[key] = deserialize_numpy_array(value)
        else:
            detections.data[key] = value
    
    return detections


def serialize_value(value: Any) -> Any:
    """Serialize a generic value for transport.
    
    Args:
        value: Value to serialize
        
    Returns:
        JSON-safe representation
    """
    if isinstance(value, WorkflowImageData):
        return serialize_workflow_image(value)
    elif isinstance(value, Batch):
        return {"type": "Batch", "content": serialize_batch(value)}
    elif isinstance(value, np.ndarray):
        return serialize_numpy_array(value)
    elif isinstance(value, sv.Detections):
        return serialize_sv_detections(value)
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        # For primitive types and unknown objects, return as-is
        return value


def deserialize_value(value: Any) -> Any:
    """Deserialize a generic value from transport format.
    
    Args:
        value: Serialized value
        
    Returns:
        Deserialized value
    """
    if isinstance(value, dict):
        if value.get("type") == "WorkflowImageData":
            return deserialize_workflow_image(value)
        elif value.get("type") == "Batch":
            return deserialize_batch(value["content"])
        elif value.get("type") == "numpy_array":
            return deserialize_numpy_array(value)
        elif value.get("type") == "sv_detections":
            return deserialize_sv_detections(value)
        else:
            return {k: deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [deserialize_value(v) for v in value]
    else:
        return value


def serialize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize function inputs for Modal transport.
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        JSON-safe dictionary
    """
    return {key: serialize_value(value) for key, value in inputs.items()}


def deserialize_inputs(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize function inputs from Modal transport format.
    
    Args:
        serialized: Serialized inputs
        
    Returns:
        Deserialized inputs dictionary
    """
    return {key: deserialize_value(value) for key, value in serialized.items()}


def serialize_outputs(outputs: Any) -> Any:
    """Serialize function outputs for Modal transport.
    
    Args:
        outputs: Function outputs (typically BlockResult)
        
    Returns:
        JSON-safe representation
    """
    if isinstance(outputs, dict):
        # BlockResult is a TypedDict, so it's already a dict
        return serialize_value(outputs)
    else:
        return serialize_value(outputs)


def deserialize_outputs(serialized: Any) -> BlockResult:
    """Deserialize function outputs from Modal transport format.
    
    Args:
        serialized: Serialized outputs
        
    Returns:
        BlockResult or deserialized value
    """
    result = deserialize_value(serialized)
    
    # Ensure it's a valid BlockResult format
    if isinstance(result, dict):
        return result  # BlockResult is a TypedDict
    else:
        # Wrap non-dict results in a BlockResult-compatible format
        return {"result": result}
