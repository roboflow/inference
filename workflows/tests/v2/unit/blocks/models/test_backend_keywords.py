"""The supported-keyword rule checked against the ``inference_models`` source.

The blocks pass an option to a model only through a ``post_process`` keyword
it names (``common.post_process_options``). That is truthful only while no
task backend reads a block option from ``**kwargs`` and every named-but-unused
keyword is in ``common.IGNORED_KEYWORDS``. The survey reads source; it imports
no backend.
"""

from roboflow_workflows.v2.blocks.models.common import IGNORED_KEYWORDS

from tests.v2.support.models.backend_survey import UNFOLLOWED, survey

# post_process keywords the six blocks bind.
BLOCK_KEYWORDS = frozenset(
    {
        "confidence",
        "iou_threshold",
        "max_detections",
        "class_agnostic_nms",
        "key_points_threshold",
        "mask_format",
    }
)


def test_no_task_backend_reads_a_block_option_through_kwargs() -> None:
    backends = survey()

    hidden = {
        backend.qualified_name: sorted(
            set(backend.kwargs_reads) & (BLOCK_KEYWORDS | {UNFOLLOWED})
        )
        for backend in backends
    }

    assert len(backends) > 50  # the survey found the backends at all
    assert {name: reads for name, reads in hidden.items() if reads} == {}


def test_ignored_keywords_are_exactly_the_named_but_unused_block_options() -> None:
    unused = {
        backend.qualified_name: frozenset(backend.named_unused) & BLOCK_KEYWORDS
        for backend in survey()
    }

    assert {name: keys for name, keys in unused.items() if keys} == dict(
        IGNORED_KEYWORDS
    )
