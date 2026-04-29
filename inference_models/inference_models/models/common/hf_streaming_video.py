"""Shared base class for HuggingFace-transformers streaming video trackers.

SAM2 and SAM3 both expose a streaming inference interface in
``transformers`` (``Sam2VideoModel`` / ``Sam3VideoModel``).  The
per-frame shape is identical:

- A ``Sam{N}VideoProcessor.init_video_session`` creates an opaque
  ``inference_session`` object that carries the model's temporal
  memory.
- Per frame, ``processor(images=frame, device=..., return_tensors="pt")``
  prepares inputs, then ``model(inference_session=..., frame=pixel_values)``
  produces outputs that the processor post-processes into masks +
  object ids.
- Prompts (text / boxes / points) are added to the session via the
  processor's ``add_text_prompt`` / ``add_inputs_to_inference_session``
  methods.

``HFStreamingVideoBase`` encapsulates all of this.  Concrete subclasses
(``SAM2Video`` today, plus any future HF video tracker) just declare
which transformers classes to load and which prompt types they accept;
they inherit the streaming ``prompt`` / ``track`` methods unchanged.
"""

from threading import RLock
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelRuntimeError

#: Key under which the HF ``inference_session`` lives inside the
#: opaque ``state_dict`` callers pass back on subsequent frames.
SESSION_KEY = "_video_inference_session"


class HFStreamingVideoBase:
    """Frame-by-frame SAM-family video tracker built on transformers.

    Concrete subclasses set class-level ``_transformers_model_cls`` and
    ``_transformers_processor_cls`` (lazy-initialised, typically in
    ``from_pretrained``) so this base can call them generically.

    State management
    ----------------
    Callers keep the opaque ``state_dict`` returned from ``prompt`` /
    ``track`` and pass it back on the next call.  Unlike SAM2's
    PyTorch state dict (a serialisable ``{str: Tensor}`` map), the HF
    inference session is a live Python object with GPU tensor
    references — it is **not serialisable across processes**.
    """

    _transformers_model_cls: Any = None
    _transformers_processor_cls: Any = None
    #: Whether ``prompt`` should accept ``text=...`` prompts.  SAM3
    #: supports them; SAM2 does not.
    _supports_text_prompts: bool = False

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """Load model + processor from a HF-layout package directory.

        ``model_name_or_path`` must point at a local directory
        containing the standard transformers files; the processor's
        own ``from_pretrained`` discovers everything it needs by
        convention.
        """
        model_cls, processor_cls = cls._resolve_transformers_classes()
        model = (
            model_cls.from_pretrained(model_name_or_path, local_files_only=True)
            .eval()
            .to(device, dtype=dtype)
        )
        processor = processor_cls.from_pretrained(
            model_name_or_path, local_files_only=True
        )
        return cls(model=model, processor=processor, device=device, dtype=dtype)

    @classmethod
    def _resolve_transformers_classes(cls) -> Tuple[Any, Any]:
        """Return ``(model_cls, processor_cls)`` for the concrete subclass.

        Subclasses may override this to lazy-import the transformers
        symbols (keeping the import cost off the module import path).
        """
        if (
            cls._transformers_model_cls is None
            or cls._transformers_processor_cls is None
        ):
            raise NotImplementedError(
                f"{cls.__name__} must set _transformers_model_cls and "
                f"_transformers_processor_cls (or override "
                f"_resolve_transformers_classes)."
            )
        return cls._transformers_model_cls, cls._transformers_processor_cls

    # ------------------------------------------------------------------
    # Streaming API — matches SAM2ForStream's shape so the two can be
    # swapped at call-sites.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prompt(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bboxes: Optional[
            Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]
        ] = None,
        text: Optional[str] = None,
        state_dict: Optional[dict] = None,
        clear_old_prompts: bool = True,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Seed a session and run one streaming step.

        ``clear_old_prompts=True`` starts a fresh session (discarding
        any prior state).  Pass ``False`` along with ``state_dict`` to
        append prompts to an ongoing session.
        """
        if text is not None and not self._supports_text_prompts:
            raise ModelRuntimeError(
                message=(
                    f"{type(self).__name__} does not support text prompts; "
                    "use `bboxes=` instead."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "models-runtime/#modelruntimeerror"
                ),
            )
        with self._lock:
            image_np = _ensure_numpy_image(image)
            session = self._resolve_session(
                state_dict=state_dict, reset=clear_old_prompts
            )

            inputs = self._processor(
                images=image_np, device=self._device, return_tensors="pt"
            )
            original_sizes = inputs.original_sizes

            if text is not None:
                session = self._processor.add_text_prompt(
                    inference_session=session,
                    text=text,
                )

            box_list = _normalise_bboxes(bboxes)
            if box_list:
                self._processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_idx,
                    obj_ids=list(range(len(box_list))),
                    input_boxes=[[[float(v) for v in xyxy] for xyxy in box_list]],
                    original_size=original_sizes[0],
                )

            model_outputs = self._model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            masks, object_ids = self._extract_masks_and_ids(
                session=session,
                model_outputs=model_outputs,
                original_sizes=original_sizes,
            )
            return masks, object_ids, {SESSION_KEY: session}

    @torch.inference_mode()
    def track(
        self,
        image: Union[np.ndarray, torch.Tensor],
        state_dict: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Propagate existing tracks onto ``image``.

        Requires a prior call to ``prompt`` — the state_dict returned
        there must be threaded back in here.
        """
        with self._lock:
            image_np = _ensure_numpy_image(image)
            session = state_dict.get(SESSION_KEY) if state_dict is not None else None
            if session is None:
                raise ModelRuntimeError(
                    message=(
                        "Attempt to track with no prior call to prompt; "
                        "prompt must be called first (pass the state_dict "
                        "returned from prompt back into track)"
                    ),
                    help_url=(
                        "https://inference-models.roboflow.com/errors/"
                        "models-runtime/#modelruntimeerror"
                    ),
                )

            inputs = self._processor(
                images=image_np, device=self._device, return_tensors="pt"
            )
            model_outputs = self._model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            masks, object_ids = self._extract_masks_and_ids(
                session=session,
                model_outputs=model_outputs,
                original_sizes=inputs.original_sizes,
            )
            return masks, object_ids, {SESSION_KEY: session}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_session(self, state_dict: Optional[dict], reset: bool) -> Any:
        if reset:
            return self._new_session()
        if state_dict is not None:
            session = state_dict.get(SESSION_KEY)
            if session is not None:
                return session
        return self._new_session()

    def _new_session(self) -> Any:
        return self._processor.init_video_session(
            inference_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype,
        )

    def _extract_masks_and_ids(
        self,
        session: Any,
        model_outputs: Any,
        original_sizes: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        postprocess = getattr(self._processor, "postprocess_outputs", None)
        if callable(postprocess):
            processed = postprocess(
                session,
                model_outputs,
                original_sizes=original_sizes,
            )
            masks, object_ids = _unpack_processed_outputs(processed)
            if masks is not None:
                return masks, object_ids

        pred_masks = getattr(model_outputs, "pred_masks", None)
        if pred_masks is None:
            return np.zeros((0, 0, 0), dtype=bool), np.zeros((0,), dtype=np.int64)
        masks = self._processor.post_process_masks(
            [pred_masks],
            original_sizes=original_sizes,
            binarize=True,
        )[0]
        masks_np = _to_numpy_binary_masks(masks)
        object_ids = _extract_object_ids(model_outputs, n=masks_np.shape[0])
        return masks_np, object_ids


# ---------------------------------------------------------------------------
# Module-level helpers (pure, easy to unit-test)
# ---------------------------------------------------------------------------


def _ensure_numpy_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    return image


def _normalise_bboxes(
    bboxes: Optional[Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]],
) -> List[Tuple[float, float, float, float]]:
    if bboxes is None:
        return []
    if not isinstance(bboxes, list):
        bboxes = [bboxes]
    out: List[Tuple[float, float, float, float]] = []
    for box in bboxes:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        x_lt = float(min(x1, x2))
        y_lt = float(min(y1, y2))
        x_rb = float(max(x1, x2))
        y_rb = float(max(y1, y2))
        out.append((x_lt, y_lt, x_rb, y_rb))
    return out


def _to_numpy_binary_masks(masks: Any) -> np.ndarray:
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if hasattr(masks, "detach"):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr.astype(bool)


def _extract_object_ids(model_outputs: Any, n: int) -> np.ndarray:
    for attr in ("obj_ids", "object_ids"):
        ids = getattr(model_outputs, attr, None)
        if ids is not None:
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            return np.asarray(list(ids), dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _first_present(container: Any, keys: Tuple[str, ...]) -> Any:
    """Return the first value in ``container`` whose key is in ``keys``.

    Needed because ``dict.get(a) or dict.get(b)`` raises on numpy /
    tensor values (ambiguous truthiness).
    """
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def _unpack_processed_outputs(
    processed: Any,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Best-effort extraction of ``(masks, obj_ids)`` from the
    processor's ``postprocess_outputs`` return — shape varies across
    transformers versions.
    """
    if processed is None:
        return None, np.zeros((0,), dtype=np.int64)

    if isinstance(processed, dict):
        masks = _first_present(processed, ("masks", "pred_masks"))
        ids = _first_present(processed, ("obj_ids", "object_ids", "track_ids"))
        if masks is None:
            return None, np.zeros((0,), dtype=np.int64)
        masks_np = _to_numpy_binary_masks(masks)
        if ids is None:
            ids = np.arange(masks_np.shape[0], dtype=np.int64)
        else:
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            ids = np.asarray(list(ids), dtype=np.int64)
        return masks_np, ids

    if isinstance(processed, (list, tuple)) and processed:
        mask_list: List[np.ndarray] = []
        id_list: List[int] = []
        for idx, item in enumerate(processed):
            if not isinstance(item, dict):
                continue
            m = _first_present(item, ("mask", "masks", "segmentation"))
            if m is None:
                continue
            oid = _first_present(item, ("obj_id", "object_id", "track_id"))
            if oid is None:
                oid = idx
            m_np = _to_numpy_binary_masks(m)
            if m_np.ndim == 3 and m_np.shape[0] == 1:
                mask_list.append(m_np[0])
            elif m_np.ndim == 2:
                mask_list.append(m_np)
            else:
                for row in m_np:
                    mask_list.append(row)
            id_list.append(int(oid))
        if mask_list:
            return (
                np.stack(mask_list, axis=0).astype(bool),
                np.asarray(id_list, dtype=np.int64),
            )

    return None, np.zeros((0,), dtype=np.int64)
