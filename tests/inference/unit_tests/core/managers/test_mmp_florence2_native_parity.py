"""Parity guard: the pure-Python Florence2 post-processing port in
inference.core.managers.mmp_florence2 must match the installed transformers
Florence2 processor exactly, parameterized with the post-processor config
shipped in the served Florence2 model packages."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("transformers.models.florence2.processing_florence2")

from inference.core.managers import mmp_florence2

_PARSE_CONFIG = {
    "pure_text": {},
    "ocr": {
        "area_threshold": 0.0,
        "pattern": (
            r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
            r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
        ),
    },
    "phrase_grounding": {
        "banned_grounding_tokens": sorted(mmp_florence2._BANNED_GROUNDING_TOKENS)
    },
    "description_with_bboxes": {},
    "description_with_polygons": {},
    "polygons": {},
    "bboxes": {},
    "description_with_bboxes_or_polygons": {},
}

TASKS = list(mmp_florence2._TASK_POST_PROCESSING_TYPE) + ["<CUSTOM>", "hello>", ">"]

TEXTS = [
    "</s><s>A green car parked in front of a yellow building.</s>",
    "</s><s>A very long caption.  </s>",
    "</s><s>car<loc_54><loc_375><loc_906><loc_707>door<loc_710><loc_276>"
    "<loc_908><loc_537>wheel<loc_708><loc_531><loc_906><loc_704></s>",
    "</s><s><loc_54><loc_375><loc_906><loc_707><loc_710><loc_276><loc_908>"
    "<loc_537></s>",
    "</s><s>A green car<loc_54><loc_375><loc_906><loc_707> parked next to trees"
    "<loc_10><loc_20><loc_200><loc_300><loc_400><loc_100><loc_600><loc_500></s>",
    "</s><s>image<loc_0><loc_0><loc_999><loc_999>a dog<loc_100><loc_200>"
    "<loc_300><loc_400></s>",
    "</s><s>the<loc_1><loc_2><loc_3><loc_4>a set<loc_5><loc_6><loc_7><loc_8></s>",
    "</s><s>STOP<loc_100><loc_100><loc_300><loc_100><loc_300><loc_200><loc_100>"
    "<loc_200>AHEAD<loc_100><loc_300><loc_300><loc_300><loc_300><loc_400>"
    "<loc_100><loc_400></s>",
    "</s><s><loc_100><loc_100><loc_200><loc_100><loc_200><loc_200><loc_100>"
    "<loc_200></s>",
    "</s><s><loc_50><loc_60><loc_150><loc_60><loc_150><loc_160><loc_50>"
    "<loc_160><loc_50><loc_61></s>",
    "</s><s>a green car<loc_54><loc_375><loc_906><loc_707></s>",
    "</s><s>a green car<poly><loc_100><loc_100><loc_200><loc_100><loc_200>"
    "<loc_200></poly></s>",
    "</s><s>a car<poly><loc_10><loc_20><loc_30><loc_40><loc_50><loc_60></poly>"
    "a dog<poly><loc_11><loc_21><loc_31><loc_41><loc_51><loc_61></poly></s>",
    "</s><s>car<loc_154><loc_258><loc_903><loc_621></s>",
    "</s><s>whatever the model says</s>",
    "</s><s></s>",
    "</s><s>  <pad></s>",
    "</s><s>no locations here at all</s>",
    "</s><s>broken<loc_1><loc_2><loc_3></s>",
    "</s><s>a<loc_1><loc_2><loc_3><loc_4><loc_5></s>",
    "</s><s>phrase<loc_999><loc_999><loc_999><loc_999></s>",
    "</s><s>x<loc_0><loc_0><loc_0><loc_0></s>",
    "</s><s>big<loc_12345><loc_2><loc_3><loc_4></s>",
    "</s><s>x<loc_0><loc_0><loc_1000><loc_1000></s>",
    "</s><s><loc_999><loc_999><loc_12345><loc_67890><loc_1><loc_2><loc_3>"
    "<loc_4></s>",
    "</s><s>café au lait<loc_10><loc_10><loc_20><loc_20></s>",
    "</s><s><ground>dog<loc_1><loc_2><loc_3><loc_4></s>",
    "</s><s><obj>dog<loc_1><loc_2><loc_3><loc_4></s>",
    "</s><s><loc_5><loc_6><loc_7><loc_8><sep><loc_9><loc_10><loc_11><loc_12></s>",
    "</s><s>region<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8><sep>"
    "<loc_9><loc_10><loc_11><loc_12></s>",
    "</s><s>mask<poly><loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><sep><loc_7>"
    "<loc_8><loc_9><loc_10><loc_11><loc_12></poly></s>",
    "</s><s>od<od>text</od><loc_1><loc_2><loc_3><loc_4></s>",
    "</s><s>multi words label<loc_100><loc_150><loc_800><loc_900>second thing"
    "<loc_5><loc_5><loc_10><loc_10></s>",
    "car<loc_154><loc_258><loc_903><loc_621>",
    "</s><s>STOP sign<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8>"
    "tail</s>",
    "A green car parked in front of a yellow building.",
    "car door wheel",
    "",
]

SIZES = [(640, 480), (1000, 1000), (13, 7), (1920, 1080), (3, 1000)]


@pytest.fixture(scope="module")
def native_post_process_generation():
    fake_tokenizer = SimpleNamespace(
        all_special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )
    post_processor = native.Florence2PostProcessor(
        config=_PARSE_CONFIG, tokenizer=fake_tokenizer
    )
    processor_like = SimpleNamespace(
        tasks_answer_post_processing_type=mmp_florence2._TASK_POST_PROCESSING_TYPE,
        post_processor=post_processor,
    )

    def run(text, task, image_size):
        return native.Florence2Processor.post_process_generation(
            processor_like, text=text, task=task, image_size=image_size
        )

    return run


@pytest.mark.parametrize("task", TASKS)
def test_port_matches_native_processor(task, native_post_process_generation):
    for text in TEXTS:
        for size in SIZES:
            expected = native_post_process_generation(text, task, size)
            actual = mmp_florence2.post_process_generation(
                text=text, task=task, image_size=size
            )
            assert actual == expected, f"task={task!r} text={text!r} size={size!r}"
