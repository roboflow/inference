"""Unit tests for registry_defaults — class-vs-MRO skip logic on _register_from_config."""

from inference_model_manager import registry_defaults
from inference_model_manager.registry import ModelRegistry


def test_subclass_override_registers_after_base(monkeypatch):
    """Subclass config must register even when a base class with same task is already registered.

    Regression: previously _register_from_config used get_entry_for_class (MRO-walking)
    to decide whether to skip. After Base.infer was registered, Sub.infer would be
    silently skipped, and Sub instances would inherit Base's validator instead of
    their own (e.g. open-vocabulary detection missing the `classes` requirement).
    """

    def base_validator(kwargs):
        return {"validator": "base", **kwargs}

    def sub_validator(kwargs):
        return {"validator": "sub", **kwargs}

    def fake_serializer(out, model):
        return {}

    fake_configs = {
        "FakeBase": [
            ("infer", "infer", True, {}, "base_v", "fake_ser", "fake-base-v1"),
        ],
        "FakeSub": [
            ("infer", "infer", True, {}, "sub_v", "fake_ser", "fake-sub-v1"),
        ],
    }
    fake_validators = {"base_v": base_validator, "sub_v": sub_validator}
    fake_serializers = {"fake_ser": fake_serializer}

    test_registry = ModelRegistry()
    monkeypatch.setattr(registry_defaults, "registry", test_registry)
    monkeypatch.setattr(registry_defaults, "_TASK_CONFIGS", fake_configs)
    monkeypatch.setattr(
        registry_defaults,
        "_resolve_validator",
        lambda name: fake_validators[name],
    )
    monkeypatch.setattr(
        registry_defaults,
        "_resolve_serializer",
        lambda name: fake_serializers[name],
    )

    class FakeBase:
        pass

    class FakeSub(FakeBase):
        pass

    registry_defaults._register_from_config(FakeBase)
    registry_defaults._register_from_config(FakeSub)

    base_entry = test_registry.get_entry(FakeBase(), "infer")
    sub_entry = test_registry.get_entry(FakeSub(), "infer")

    assert base_entry is not None
    assert sub_entry is not None
    assert base_entry.response_type == "fake-base-v1"
    assert sub_entry.response_type == "fake-sub-v1"
    assert sub_entry.validator({}) == {"validator": "sub"}


def test_register_from_config_idempotent_for_same_class(monkeypatch):
    """Calling _register_from_config twice on same class must not duplicate or overwrite."""

    def validator(kwargs):
        return kwargs

    def serializer(out, model):
        return {}

    fake_configs = {
        "FakeBase": [
            ("infer", "infer", True, {}, "v", "s", "fake-v1"),
        ],
    }

    test_registry = ModelRegistry()
    monkeypatch.setattr(registry_defaults, "registry", test_registry)
    monkeypatch.setattr(registry_defaults, "_TASK_CONFIGS", fake_configs)
    monkeypatch.setattr(registry_defaults, "_resolve_validator", lambda _: validator)
    monkeypatch.setattr(registry_defaults, "_resolve_serializer", lambda _: serializer)

    class FakeBase:
        pass

    registry_defaults._register_from_config(FakeBase)
    first_entry = test_registry.get_entry(FakeBase(), "infer")

    registry_defaults._register_from_config(FakeBase)
    second_entry = test_registry.get_entry(FakeBase(), "infer")

    assert first_entry is second_entry
    assert test_registry.registered_tasks(FakeBase) == ["infer"]


class TestRegistryThreadSafety:
    def test_concurrent_register_and_mro_lookup(self):
        import threading

        reg = ModelRegistry()

        def _register(name):
            reg.register(
                type(name, (), {}),
                "infer",
                default=True,
                validator=lambda kw: kw,
                serializer=lambda out, model: {},
                response_type="x",
            )

        for i in range(300):
            _register(f"Seed{i}")

        errors: list = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    reg.get_entry_by_mro_names(["Seed0"], "infer")
                    reg.get_default_task_by_mro_names(["Seed299"])
                except Exception as exc:
                    errors.append(exc)
                    return

        def writer():
            for i in range(3000):
                _register(f"W{threading.get_ident()}_{i}")

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writers = [threading.Thread(target=writer) for _ in range(2)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join(timeout=30)
        stop.set()
        for t in readers:
            t.join(timeout=5)

        assert errors == []


def test_moondream2_task_configs_match_model_signatures():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS["MoonDream2HF"]}

    assert cfgs["detect"][3]["classes"] == {"type": "list[str]", "required": True}
    assert cfgs["detect"][4] == "validate_images_and_classes"

    assert cfgs["query"][7] == {"prompt": "question"}

    assert cfgs["point"][3]["classes"] == {"type": "list[str]", "required": True}
    assert cfgs["point"][5] == "serialize_keypoints_compact"
    assert cfgs["point"][6] == "roboflow-keypoints-compact-v1"

    assert cfgs["caption"][3]["length"] == {"type": "str", "required": False}
    for task in ("caption", "detect", "query", "point"):
        assert "max_new_tokens" in cfgs[task][3]


def test_florence2_task_configs():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS["Florence2HF"]}

    assert cfgs["segment_phrase"][7] == {"prompt": "phrase"}
    assert cfgs["ground_phrase"][7] == {"prompt": "phrase"}
    assert cfgs["caption"][3]["granularity"] == {"type": "str", "required": False}
    assert cfgs["detect"][3]["labels_mode"] == {"type": "str", "required": False}
    assert cfgs["detect"][3]["classes"] == {"type": "list[str]", "required": False}
    for task in ("caption", "detect", "ocr", "parse_document"):
        assert "max_new_tokens" in cfgs[task][3]


def test_sam_and_sam2_segment_configs():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    for key, cache_flags in (
        ("SAMTorch", {"enforce_mask_input", "use_mask_input_cache"}),
        ("SAM2Torch", {"load_from_mask_input_cache", "save_to_mask_input_cache"}),
    ):
        cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS[key]}
        seg = cfgs["segment"]
        assert "points" not in seg[3]
        assert seg[3]["embeddings"]["required"] is False
        for p in (
            "images",
            "image_hashes",
            "point_coordinates",
            "point_labels",
            "boxes",
            "mask_input",
            "multi_mask_output",
            "return_logits",
            "mask_threshold",
            "use_embeddings_cache",
            *cache_flags,
        ):
            assert p in seg[3], p
        assert seg[4] == "validate_sam_segment"
        assert seg[5] == "serialize_sam_segmentation_compact"
        emb = cfgs["embed"]
        assert "image_hashes" in emb[3] and "use_embeddings_cache" in emb[3]


def test_sam2_stream_configs_match_signatures():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS["SAM2ForStream"]}
    assert cfgs["prompt"][3]["image"]["required"] is True
    assert cfgs["prompt"][3]["bboxes"]["required"] is True
    assert "prompt" not in cfgs["prompt"][3]
    assert cfgs["track"][3]["image"]["required"] is True
    assert cfgs["prompt"][5] == "serialize_passthrough"


def test_owlv2_few_shot_task_config():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS["OWLv2HF"]}
    few_shot = cfgs["infer_with_reference_examples"]

    assert few_shot[1] == "infer_with_reference_examples"
    assert few_shot[2] is False
    assert few_shot[3]["images"] == {"type": "image", "required": True}
    assert few_shot[3]["reference_examples"] == {"type": "list", "required": True}
    for param in ("confidence", "iou_threshold", "max_detections"):
        assert few_shot[3][param]["required"] is False
        assert "default" not in few_shot[3][param]
    assert "classes" not in few_shot[3]
    assert "class_agnostic_nms" not in few_shot[3]
    assert few_shot[4] == "validate_images_required"
    assert few_shot[5] == "serialize_detections_compact"
    assert few_shot[6] == "roboflow-object-detection-compact-v1"


def test_cosmos3_edge_reasoner_task_config():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    cfgs = {c[0]: _unpack_config(c) for c in _TASK_CONFIGS["Cosmos3EdgeReasoner"]}
    prompt_task = cfgs["prompt"]

    assert prompt_task[1] == "prompt"
    assert prompt_task[2] is True
    assert prompt_task[3]["images"] == {"type": "image", "required": True}
    assert prompt_task[3]["prompt"] == {"type": "str", "required": True}
    for param in ("max_new_tokens", "do_sample", "skip_special_tokens", "return_thinking"):
        assert param in prompt_task[3], param
        assert prompt_task[3][param]["required"] is False
    assert prompt_task[3]["max_new_tokens"]["type"] == "int"
    assert prompt_task[3]["do_sample"]["type"] == "bool"
    assert prompt_task[3]["skip_special_tokens"]["type"] == "bool"
    assert prompt_task[3]["return_thinking"]["type"] == "bool"
    assert "default" not in prompt_task[3]["max_new_tokens"]
    assert "default" not in prompt_task[3]["do_sample"]
    assert prompt_task[4] == "validate_images_and_prompt"
    assert prompt_task[5] == "serialize_text"
    assert prompt_task[6] == "roboflow-text-v1"
    assert prompt_task[7] == {}


def test_model_owned_defaults_not_injected():
    from inference_model_manager.registry_defaults import _K_ISEG, _K_KP, _K_OD

    for frag in (_K_OD, _K_ISEG, _K_KP):
        for name, spec in frag.items():
            assert "default" not in spec, name


def test_per_family_detector_contracts():
    from inference_model_manager.registry_defaults import _K_ISEG, _TASK_CONFIGS, _unpack_config

    def params_of(key):
        return _unpack_config(_TASK_CONFIGS[key][0])[3]

    for key in (
        "RFDetrForObjectDetectionTorch",
        "RFDetrForObjectDetectionONNX",
        "RFDetrForObjectDetectionTRT",
        "YOLO26ForObjectDetectionOnnx",
        "YOLO26ForObjectDetectionTorchScript",
        "YOLO26ForObjectDetectionTRT",
    ):
        p = params_of(key)
        assert set(p) == {"images", "confidence"}, key

    for key in (
        "YOLOv10ForObjectDetectionOnnx",
        "YOLOv10ForObjectDetectionTRT",
    ):
        p = params_of(key)
        assert set(p) == {"images", "confidence", "max_detections"}, key

    p = params_of("RoboflowInstantHF")
    assert set(p) == {"images", "confidence", "iou_threshold", "max_detections"}

    p = params_of("PPOCRv6DetectionOnnx")
    assert set(p) == {"images"}

    p = params_of("GroundingDinoForObjectDetectionTorch")
    assert "confidence" not in p
    assert set(p) == {
        "images",
        "classes",
        "box_confidence",
        "text_confidence",
        "iou_threshold",
        "max_detections",
        "class_agnostic_nms",
    }

    assert "mask_format" in _K_ISEG
    assert "default" not in _K_ISEG["mask_format"]

    for key in (
        "YOLOv5ForInstanceSegmentationOnnx",
        "YOLOv5ForInstanceSegmentationTRT",
        "YOLOv7ForInstanceSegmentationOnnx",
        "YOLOv7ForInstanceSegmentationTRT",
        "YOLOACTForInstanceSegmentationOnnx",
        "YOLOACTForInstanceSegmentationTRT",
    ):
        p = params_of(key)
        assert set(p) == {
            "images",
            "confidence",
            "iou_threshold",
            "max_detections",
            "class_agnostic_nms",
            "mask_format",
        }, key

    for key in (
        "RFDetrForInstanceSegmentationTorch",
        "RFDetrForInstanceSegmentationOnnx",
        "RFDetrForInstanceSegmentationTRT",
    ):
        p = params_of(key)
        assert set(p) == {"images", "confidence", "mask_format", "max_detections"}, key

    for key in (
        "YOLO26ForInstanceSegmentationOnnx",
        "YOLO26ForInstanceSegmentationTorchScript",
        "YOLO26ForInstanceSegmentationTRT",
    ):
        p = params_of(key)
        assert set(p) == {"images", "confidence", "mask_format"}, key

    for key in (
        "RFDetrForKeyPointsONNX",
        "YOLO26ForKeyPointsDetectionOnnx",
        "YOLO26ForKeyPointsDetectionTorchScript",
        "YOLO26ForKeyPointsDetectionTRT",
    ):
        p = params_of(key)
        assert set(p) == {"images", "confidence", "key_points_threshold"}, key

    for key in ("SemanticSegmentationModel", "MultiLabelClassificationModel"):
        p = params_of(key)
        assert set(p) == {"images", "confidence"}, key
        assert p["confidence"] == {"type": "float", "required": False}, key


def test_per_family_contracts_have_no_default_keys():
    from inference_model_manager.registry_defaults import _TASK_CONFIGS, _unpack_config

    keys = (
        "RFDetrForObjectDetectionTorch",
        "RFDetrForObjectDetectionONNX",
        "RFDetrForObjectDetectionTRT",
        "YOLO26ForObjectDetectionOnnx",
        "YOLO26ForObjectDetectionTorchScript",
        "YOLO26ForObjectDetectionTRT",
        "YOLOv10ForObjectDetectionOnnx",
        "YOLOv10ForObjectDetectionTRT",
        "PPOCRv6DetectionOnnx",
        "RoboflowInstantHF",
        "GroundingDinoForObjectDetectionTorch",
        "YOLOv5ForInstanceSegmentationOnnx",
        "YOLOv5ForInstanceSegmentationTRT",
        "YOLOv7ForInstanceSegmentationOnnx",
        "YOLOv7ForInstanceSegmentationTRT",
        "YOLOACTForInstanceSegmentationOnnx",
        "YOLOACTForInstanceSegmentationTRT",
        "RFDetrForInstanceSegmentationTorch",
        "RFDetrForInstanceSegmentationOnnx",
        "RFDetrForInstanceSegmentationTRT",
        "YOLO26ForInstanceSegmentationOnnx",
        "YOLO26ForInstanceSegmentationTorchScript",
        "YOLO26ForInstanceSegmentationTRT",
        "RFDetrForKeyPointsONNX",
        "YOLO26ForKeyPointsDetectionOnnx",
        "YOLO26ForKeyPointsDetectionTorchScript",
        "YOLO26ForKeyPointsDetectionTRT",
        "SemanticSegmentationModel",
        "MultiLabelClassificationModel",
    )
    for key in keys:
        for cfg in _TASK_CONFIGS[key]:
            for name, spec in _unpack_config(cfg)[3].items():
                assert "default" not in spec, (key, name)


def test_owlv2_few_shot_registers_alongside_zero_shot_default(monkeypatch):
    test_registry = ModelRegistry()
    monkeypatch.setattr(registry_defaults, "registry", test_registry)

    class OpenVocabularyObjectDetectionModel:
        pass

    class OWLv2HF(OpenVocabularyObjectDetectionModel):
        pass

    for cls in OWLv2HF.__mro__:
        registry_defaults._register_from_config(cls)

    model = OWLv2HF()
    zero_shot = test_registry.get_entry(model, "infer")
    few_shot = test_registry.get_entry(model, "infer_with_reference_examples")

    assert zero_shot is not None
    assert zero_shot.default is True
    assert few_shot is not None
    assert few_shot.default is False
    assert few_shot.method == "infer_with_reference_examples"
