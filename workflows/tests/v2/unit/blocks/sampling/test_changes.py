"""``roboflow_core/identify_changes@v1``: V1 statistics per source, bounded and checked.

Calls run inside a hand-built execution context whose scope answers the
call's source, as the engine's does. ``V1_CAMERA_A`` was recorded by running
V1 ``identify_changes/v1_tensor.py`` on ``camera(0.0, 9, 14)`` in float32
(threshold 0.2, warmup 3, smoothing 0.1, window 4); the recorder lives in the
harness task ``ecosystem-e4-embedding-sampling/v1_trace``.
"""

import math
from typing import Any, Dict, List, Optional

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.sampling import (
    IdentifyChangesBlock,
    SamplingCapacityError,
)
from roboflow_workflows.v2.blocks.sampling._state import unit_vector
from roboflow_workflows.v2.blocks.sampling.identify_changes.v1 import EMA, SLIDING_WINDOW, SMA, _cosine
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)

STRATEGIES = [EMA, SMA, SLIDING_WINDOW]
PARAMS = {
    "threshold_percentile": 0.2,
    "warmup": 3,
    "smoothing_factor": 0.1,
    "window_size": 4,
}


def camera(phase: float, shift_at: int, steps: int) -> List[List[float]]:
    """4-dim embeddings wobbling around one direction, then another from shift_at."""
    rows = []
    for i in range(steps):
        wobble = (
            round(0.08 * math.sin(1.7 * i + phase), 3),
            round(0.06 * math.cos(1.1 * i + phase), 3),
        )
        if i < shift_at:
            row = [1.0, 0.2 + wobble[0], 0.1 + wobble[1], 0.05]
        else:
            row = [0.1 + wobble[0], 1.0, 0.3 + wobble[1], -0.2]
        rows.append(row)

    return rows


A = camera(0.0, 9, 14)
B = camera(2.0, 6, 10)

# (is_outlier, warming_up, percentile, z_score, average, std) per call.
# fmt: off
V1_CAMERA_A = {
    EMA: [
        (False, True, 0.5, 0.0,
         [0.967595935, 0.19351919, 0.154815346, 0.0483797975],
         [0.0, 0.0, 0.0, 0.0]),
        (False, True, 0.5, 0.0,
         [0.966335893, 0.200811654, 0.151462257, 0.0483167954],
         [0.00358597329, 0.0207547415, 0.00954304449, 0.000179298906]),
        (False, True, 0.999217298871, -3.16227766017,
         [0.96780163, 0.198388353, 0.142692491, 0.048390083],
         [0.00538289314, 0.0208626203, 0.0265504122, 0.000269144977]),
        (True, False, 0.999057853856, -3.10788836663,
         [0.970033407, 0.191025019, 0.132482737, 0.0485016704],
         [0.00815011468, 0.0288252365, 0.0384547673, 0.000407505722]),
        (False, False, 0.149371460106, 1.03913292486,
         [0.9698475, 0.19515869, 0.127173483, 0.0484923758],
         [0.00774995703, 0.0297693182, 0.0394869, 0.000387497828]),
        (False, False, 0.11207926459, 1.21554438987,
         [0.968529403, 0.200898811, 0.128136456, 0.0484264679],
         [0.00825403724, 0.0326263346, 0.0375606865, 0.000412701454]),
        (False, False, 0.194180085309, 0.862595017775,
         [0.969364882, 0.19487606, 0.130659893, 0.0484682433],
         [0.00818355475, 0.0353814214, 0.0363497362, 0.000409177504]),
        (False, False, 0.113777804982, 1.20667946606,
         [0.970619559, 0.190215319, 0.128296733, 0.0485309772],
         [0.00854546297, 0.0360917225, 0.035134133, 0.000427273131]),
        (True, False, 0.917148729834, -1.38614529983,
         [0.969895899, 0.197108775, 0.120380312, 0.0484947935],
         [0.00836445577, 0.0394621715, 0.0402317196, 0.000418222626]),
        (True, False, 0.999217053267, -3.16218630463,
         [0.885387838, 0.271955192, 0.131697923, 0.0247338545],
         [0.240644917, 0.216281325, 0.0499424562, 0.0676259771]),
        (True, False, 0.985127958211, -2.1734816747,
         [0.799012184, 0.338809729, 0.146743149, 0.00345045328],
         [0.335486263, 0.27982682, 0.0638618171, 0.0882332623]),
        (True, False, 0.909153523913, -1.33556054885,
         [0.727235973, 0.397258133, 0.164753437, -0.015360469],
         [0.378187448, 0.313279599, 0.0793593004, 0.099361904]),
        (False, False, 0.753986812983, -0.687089430944,
         [0.670988679, 0.449067503, 0.180132329, -0.0321314558],
         [0.392873466, 0.331770688, 0.0870852694, 0.105658717]),
        (False, False, 0.729011583959, -0.609826368769,
         [0.612441361, 0.498133868, 0.189371303, -0.047712937],
         [0.40826422, 0.344333351, 0.0866998509, 0.109608062]),
    ],
    SMA: [
        (False, True, 0.5, 0.0,
         [0.967595935, 0.19351919, 0.154815346, 0.0483797975],
         [0.0, 0.0, 0.0, 0.0]),
        (False, True, 0.5, 0.0,
         [0.961295962, 0.229981542, 0.13804993, 0.0480647981],
         [0.00890944339, 0.0515655689, 0.0237098932, 0.000445472426]),
        (False, True, 0.875893460505, -1.15470053838,
         [0.967861772, 0.212180629, 0.113288142, 0.0483930893],
         [0.0130007165, 0.0477506109, 0.0460490771, 0.00065003708]),
        (True, False, 0.922630475523, -1.42299001797,
         [0.973426282, 0.190324247, 0.0951148346, 0.0486713126],
         [0.0153796682, 0.0585737675, 0.0522948839, 0.000768984028]),
        (True, False, 0.0936722675075, 1.31847545902,
         [0.972375929, 0.198731765, 0.0919699296, 0.0486187935],
         [0.0135246813, 0.0540980287, 0.0458314121, 0.000676234253]),
        (False, False, 0.305554110404, 0.508492181502,
         [0.969757676, 0.207703114, 0.0994421542, 0.0484878793],
         [0.0136918267, 0.0531430915, 0.0448934212, 0.000684590545]),
        (False, False, 0.535400672178, -0.0888531015017,
         [0.970775783, 0.198127151, 0.107146256, 0.0485387854],
         [0.0127858343, 0.0547301024, 0.04577104, 0.000639291422]),
        (False, False, 0.135145202506, 1.1023940319,
         [0.97216779, 0.191894844, 0.107131518, 0.048608385],
         [0.0124749718, 0.0536489114, 0.0423757695, 0.000623748871]),
        (False, False, 0.641746156212, -0.363130117354,
         [0.971191704, 0.199367642, 0.100687191, 0.04855958],
         [0.0120310588, 0.0549637564, 0.0441022366, 0.000601552718]),
        (True, False, 0.997786172504, -2.84596867703,
         [0.886554122, 0.273988158, 0.113974124, 0.0247921646],
         [0.26788789, 0.241593808, 0.0591127798, 0.0751613081]),
        (True, False, 0.978381322757, -2.02140986365,
         [0.807924807, 0.334580213, 0.129262865, 0.00543831848],
         [0.364137083, 0.304821551, 0.0756048188, 0.0959405228]),
        (True, False, 0.903918455877, -1.30420678169,
         [0.747368574, 0.383639693, 0.145728126, -0.0104031041],
         [0.405643135, 0.336676747, 0.0919222608, 0.10667339]),
        (False, False, 0.774437874605, -0.753542053004,
         [0.702552795, 0.424540639, 0.159021527, -0.023685202],
         [0.420647204, 0.354475409, 0.100214027, 0.112802163]),
        (False, False, 0.76897541268, -0.735476782819,
         [0.658478737, 0.46133998, 0.167128712, -0.0354181379],
         [0.436495513, 0.367349833, 0.100947984, 0.116930738]),
    ],
    SLIDING_WINDOW: [
        (False, True, 0.5, 0.0,
         [0.967595935, 0.19351919, 0.154815346, 0.0483797975],
         [0.0, 0.0, 0.0, 0.0]),
        (False, True, 0.5, 0.0,
         [0.954996049, 0.266443908, 0.1212845, 0.0477498025],
         [0.0, 0.0, 0.0, 0.0]),
        (False, True, 0.5, 0.0,
         [0.96799469, 0.221511364, 0.092524536, 0.048399739],
         [0.0129986703, 0.0449325442, 0.0287599638, 0.000649934635]),
        (False, False, 0.841344746069, -1.0,
         [0.975369751, 0.189259276, 0.0752146617, 0.0487684906],
         [0.0148803834, 0.058535032, 0.0339217819, 0.000744019169]),
        (True, False, 0.0788328413812, 1.41296613119,
         [0.973570943, 0.200034916, 0.0762585774, 0.0486785471],
         [0.0132580791, 0.0540194958, 0.029432714, 0.000662903883]),
        (False, False, 0.356479518, 0.367884932111,
         [0.973988473, 0.1965639, 0.0801382661, 0.0486994237],
         [0.0126802195, 0.0499323271, 0.0355079435, 0.000634010648]),
        (False, False, 0.611132544106, -0.282272054697,
         [0.972961187, 0.187587053, 0.102539837, 0.0486480594],
         [0.0122295609, 0.0556219108, 0.0450856127, 0.000611477473]),
        (False, False, 0.152907686605, 1.02404213615,
         [0.970909238, 0.193465441, 0.11914821, 0.0485454611],
         [0.0095799407, 0.0495859049, 0.0283305123, 0.000478996953]),
        (False, False, 0.821981031593, -0.922941030547,
         [0.969711423, 0.20016247, 0.111583754, 0.0484855697],
         [0.0101307156, 0.0558058694, 0.039696008, 0.000506535755]),
        (True, False, 0.958365709137, -1.73202797471,
         [0.761748731, 0.373415709, 0.135772064, -0.0107514001],
         [0.367795914, 0.333647102, 0.0674613193, 0.10297858]),
        (False, False, 0.771047042662, -0.742299467022,
         [0.52293551, 0.573373079, 0.167966902, -0.0699874908],
         [0.451236755, 0.371740937, 0.0937749147, 0.118620858]),
        (False, False, 0.281256818661, 0.579111949924,
         [0.297769994, 0.767129362, 0.222921312, -0.128426075],
         [0.386033505, 0.293398231, 0.105622031, 0.10197068]),
        (True, False, 0.0978496445633, 1.29390195506,
         [0.0981150791, 0.931179821, 0.290273786, -0.186235964],
         [0.0531250387, 0.012316185, 0.0368085802, 0.00246323645]),
        (False, False, 0.189678549049, 0.879081480903,
         [0.0882900655, 0.929719448, 0.300015211, -0.185943887],
         [0.0508644879, 0.0107725598, 0.023120217, 0.00215451187]),
    ],
}
# fmt: on


class _Scope:
    """Engine side of a call whose ``embedding`` binding has context ``sample``."""

    def __init__(self, sample: Optional[SampleContext]) -> None:
        self._sample = sample

    def sample_at(self, index, *, field=None):
        # The state key must come from the embedding, not the whole call.
        assert field == "embedding", field
        return self._sample

    def temporal_at(self, index, *, field=None):
        return None


def _call(
    block: IdentifyChangesBlock,
    embedding: Any,
    *,
    source: Optional[str] = "cam-a",
    index=(0,),
    **overrides: Any,
) -> Dict[str, Any]:
    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("changes",),
        block_type=IdentifyChangesBlock.type,
        session_id="session",
        run_id="run",
        indices=(index,),
        call_scope=_Scope(sample),
    )
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)
    params = {
        "strategy": EMA,
        **PARAMS,
        "scope": "source",
        "max_sources": 256,
        **overrides,
    }
    with use_execution_context(context):
        result = IdentifyChangesBlock.run(block, embedding=embedding, **params)

    return result


def _scalars(result: Dict[str, Any]) -> tuple:
    scalars = (
        result["is_outlier"],
        result["warming_up"],
        result["percentile"],
        result["z_score"],
    )

    return scalars


# ------------------------------------------------------------------ V1 trace
def test_the_trace_flags_the_scene_shift():
    flags = [row[0] for row in V1_CAMERA_A[EMA]]

    assert flags[9:11] == [True, True]
    assert not any(row[0] for row in V1_CAMERA_A[EMA][:3])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_matches_the_recorded_v1_trace(strategy):
    block = IdentifyChangesBlock()

    for embedding, expected in zip(A, V1_CAMERA_A[strategy]):
        result = _call(block, embedding, strategy=strategy)
        is_outlier, warming_up, percentile, z_score, average, std = expected

        assert (result["is_outlier"], result["warming_up"]) == (
            is_outlier,
            warming_up,
        )
        assert result["percentile"] == pytest.approx(percentile, abs=1e-9)
        assert result["z_score"] == pytest.approx(z_score, abs=1e-9)
        assert type(result["z_score"]) is float
        assert result["average"].dtype == torch.float32
        torch.testing.assert_close(
            result["average"], torch.tensor(average), atol=1e-7, rtol=0
        )
        torch.testing.assert_close(result["std"], torch.tensor(std), atol=1e-7, rtol=0)


def test_average_and_std_are_per_dimension_tensors_on_the_input_device():
    block = IdentifyChangesBlock()
    embedding = torch.tensor(A[0], dtype=torch.float64)

    result = _call(block, embedding)

    assert result["average"].shape == (4,)
    assert result["average"].dtype == torch.float64
    assert result["average"].device == embedding.device
    assert result["std"].shape == (4,)


# --------------------------------------------------------------- per source
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_two_cameras_keep_separate_statistics(strategy):
    alone = IdentifyChangesBlock()
    expected_a = [_scalars(_call(alone, row, strategy=strategy)) for row in A]
    alone = IdentifyChangesBlock()
    expected_b = [
        _scalars(_call(alone, row, source="cam-b", strategy=strategy)) for row in B
    ]
    block = IdentifyChangesBlock()
    got_a, got_b = [], []

    for n, row in enumerate(A):
        got_a.append(_scalars(_call(block, row, strategy=strategy)))
        if n < len(B):
            got_b.append(
                _scalars(_call(block, B[n], source="cam-b", strategy=strategy))
            )

    assert got_a == expected_a
    assert got_b == expected_b


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_global_scope_shares_one_state_like_v1(strategy):
    interleaved = [row for pair in zip(A, B) for row in pair] + A[len(B) :]
    single = IdentifyChangesBlock()
    expected = [_scalars(_call(single, row, strategy=strategy)) for row in interleaved]
    block = IdentifyChangesBlock()
    sources = ["cam-a", "cam-b"] * len(B) + ["cam-a"] * (len(A) - len(B))

    got = [
        _scalars(_call(block, row, source=source, strategy=strategy, scope="global"))
        for row, source in zip(interleaved, sources)
    ]

    assert got == expected
    per_source = IdentifyChangesBlock()
    separate = [
        _scalars(_call(per_source, row, source=source, strategy=strategy))
        for row, source in zip(interleaved, sources)
    ]
    assert separate != expected


def test_without_a_source_each_top_level_position_is_a_source():
    block = IdentifyChangesBlock()

    first = [_call(block, A[n], source=None, index=(0, 5)) for n in range(4)]
    other = _call(block, B[0], source=None, index=(1,))

    assert [r["warming_up"] for r in first] == [True, True, True, False]
    assert other["warming_up"] is True


def test_a_source_too_many_fails_and_keeps_the_others():
    block = IdentifyChangesBlock()
    _call(block, A[0], source="cam-a", max_sources=2)
    _call(block, A[0], source="cam-b", max_sources=2)

    with pytest.raises(SamplingCapacityError, match="max_sources=2"):
        _call(block, A[0], source="cam-c", max_sources=2)

    assert _call(block, A[1], source="cam-a", max_sources=2)["warming_up"] is True
    assert _call(block, A[0], source="cam-c", max_sources=3)["warming_up"] is True


def test_global_scope_needs_one_state_whatever_max_sources():
    block = IdentifyChangesBlock()

    for source in ["cam-a", "cam-b", "cam-c"]:
        _call(block, A[0], source=source, scope="global", max_sources=1)

    result = _call(block, A[1], source="cam-d", scope="global", max_sources=1)
    assert result["warming_up"] is False


def test_reset_state_forgets_every_source():
    block = IdentifyChangesBlock()
    for row in A[:4]:
        _call(block, row)
        _call(block, row, source="cam-b")

    block.reset_state()

    assert _call(block, A[0])["warming_up"] is True
    assert _call(block, A[0], source="cam-b")["warming_up"] is True


# ------------------------------------------------------- warmup and constants
def test_warmup_still_reports_the_percentile_as_v1_does():
    block = IdentifyChangesBlock()

    results = [_call(block, row, warmup=100) for row in A]

    assert all(r["warming_up"] and not r["is_outlier"] for r in results)
    expected = [row[2] for row in V1_CAMERA_A[EMA]]
    assert [r["percentile"] for r in results] == pytest.approx(expected, abs=1e-9)


def test_selected_warmup_may_change_between_calls():
    block = IdentifyChangesBlock()

    flags = [_call(block, row, warmup=w)["warming_up"] for row, w in zip(A, [5, 5, 1])]

    assert flags == [True, True, False]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_constant_embedding_never_changes(strategy):
    block = IdentifyChangesBlock()

    # Normalizes exactly in float32, so no rounding residue enters the stats.
    results = [_call(block, [0.0, 0.0, 2.0], strategy=strategy) for _ in range(8)]

    assert all(r["percentile"] == 0.5 and r["z_score"] == 0.0 for r in results)
    assert not any(r["is_outlier"] for r in results)
    torch.testing.assert_close(results[-1]["average"], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(results[-1]["std"], torch.zeros(3))


# V1 on [0, 3, 4] (EMA, float32): rounding gave a cosine std of ~3e-8 and
# these flags and this third percentile. V2 treats that std as noise.
V1_CONSTANT_FLAGS = [False] * 3 + [True] * 3
V1_CONSTANT_PERCENTILE_2 = 0.000782701129001


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "embedding",
    [[0.0, 3.0, 4.0], torch.randn(512, generator=torch.Generator().manual_seed(5))],
    ids=["v1-034", "random-512"],
)
def test_rounding_of_a_constant_embedding_is_not_a_change(embedding, strategy, dtype):
    block = IdentifyChangesBlock()
    embedding = torch.as_tensor(embedding, dtype=dtype)

    results = [_call(block, embedding, strategy=strategy) for _ in range(12)]

    assert [r["is_outlier"] for r in results] == [False] * 12
    assert all(r["percentile"] == 0.5 and r["z_score"] == 0.0 for r in results)


def test_v2_does_not_flag_the_constant_embedding_that_v1_flagged():
    block = IdentifyChangesBlock()

    results = [_call(block, [0.0, 3.0, 4.0]) for _ in range(6)]

    assert [r["is_outlier"] for r in results] == [False] * 6 != V1_CONSTANT_FLAGS
    assert results[2]["percentile"] == 0.5 != V1_CONSTANT_PERCENTILE_2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_near_constant_embedding_keeps_real_statistics(strategy):
    block = IdentifyChangesBlock()
    generator = torch.Generator().manual_seed(3)
    scene = torch.randn(64, generator=generator)
    rows = [scene + 1e-2 * torch.randn(64, generator=generator) for _ in range(12)]

    results = [_call(block, row, strategy=strategy) for row in rows]

    # 1% noise moves the cosine by ~1e-5, far above float32 rounding (~6e-8),
    # so every call with a cosine history is scored. Noise of ~0.1% moves it
    # by only a few rounding steps and sits at the floor.
    assert all(r["z_score"] != 0.0 for r in results[3:])
    state = block._states.get(("source", "cam-a"))
    assert state.cs_std > 10 * 2 * torch.finfo(torch.float32).eps


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_real_change_after_a_constant_scene_is_flagged(strategy):
    block = IdentifyChangesBlock()
    constant = [_call(block, [0.0, 3.0, 4.0], strategy=strategy) for _ in range(20)]

    changed = _call(block, [3.0, 0.0, 4.0], strategy=strategy)

    assert not any(r["is_outlier"] for r in constant)
    assert changed["is_outlier"] is True
    assert changed["percentile"] >= 1 - PARAMS["threshold_percentile"] / 2


def test_the_cosine_is_clamped_to_one():
    e = unit_vector(torch.randn(2, generator=torch.Generator().manual_seed(0)))
    average = e.clone()
    for _ in range(5):
        average = 0.9 * average + 0.1 * e
        raw = torch.dot(e, average) / torch.sqrt(
            torch.dot(e, e) * torch.dot(average, average)
        )
        if raw.item() > 1:
            break

    # float32 rounding: this unclamped cosine is 1 + eps.
    assert raw.item() == 1 + torch.finfo(torch.float32).eps
    assert _cosine(e, average) == 1.0


def test_a_zero_vector_is_kept_as_a_new_zero_tensor():
    block = IdentifyChangesBlock()
    zero = torch.zeros(3)

    result = _call(block, zero)

    assert result["average"] is not zero
    assert result["average"].untyped_storage().data_ptr() != zero.data_ptr()
    torch.testing.assert_close(result["average"], zero)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "rows",
    [
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        [[0.0, 0.0]] * 5,
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
    ],
    ids=["zero-then-finite", "all-zero", "opposite"],
)
def test_zero_vectors_have_cosine_0_and_never_poison_the_state(rows, strategy):
    # V1 computed 0/0 = NaN for a zero vector and kept NaN statistics forever.
    block = IdentifyChangesBlock()

    results = [
        _call(block, row, strategy=strategy, window_size=2, warmup=1) for row in rows
    ]

    for r in results:
        assert math.isfinite(r["percentile"]) and math.isfinite(r["z_score"])
        assert torch.isfinite(r["average"]).all() and torch.isfinite(r["std"]).all()
    later = _call(block, [0.0, 1.0], strategy=strategy, window_size=2)
    assert math.isfinite(later["z_score"])


def test_sliding_windows_trim_to_a_smaller_selected_size_at_once():
    block = IdentifyChangesBlock()
    rows = camera(0.0, 9, 14)
    for row in rows[:8]:
        _call(block, row, strategy=SLIDING_WINDOW, window_size=5)
    state = block._states.get(("source", "cam-a"))

    sizes = []
    for row, size in zip(rows[8:12], [2, 2, 1, 3]):
        result = _call(block, row, strategy=SLIDING_WINDOW, window_size=size)
        sizes.append((len(state.window), len(state.cs_window)))

    assert sizes == [(2, 2), (2, 2), (1, 1), (2, 2)]
    newest = torch.stack([unit_vector(torch.tensor(row)) for row in rows[10:12]])
    torch.testing.assert_close(result["average"], newest.mean(dim=0))
    assert state.cs_avg == pytest.approx(sum(state.cs_window) / 2)
    assert state.samples == 12


def test_outputs_do_not_change_when_later_calls_update_the_state():
    block = IdentifyChangesBlock()
    first = _call(block, A[0])
    kept = first["average"].clone()

    _call(block, A[9])

    torch.testing.assert_close(first["average"], kept)


# ----------------------------------------------------------- invalid inputs
@pytest.mark.parametrize(
    "embedding",
    [
        torch.zeros(0),
        torch.zeros(2, 3),
        torch.tensor([1, 2, 3]),
        [1.0, 2.0],
    ],
    ids=["empty", "2-D", "integer", "list"],
)
def test_an_embedding_must_be_a_non_empty_1d_float_tensor(embedding):
    block = IdentifyChangesBlock()

    with pytest.raises(ValueError, match="non-empty 1-D floating tensor"):
        IdentifyChangesBlock.run(
            block,
            embedding=embedding,
            strategy=EMA,
            **PARAMS,
            scope="global",
            max_sources=1,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("scope", ["source", "global"])
def test_a_nonfinite_embedding_fails_and_changes_nothing(bad, scope):
    fresh, block = IdentifyChangesBlock(), IdentifyChangesBlock()
    expected = [_scalars(_call(fresh, row)) for row in A[:6]]

    # A rejected first call of cam-b takes no max_sources slot.
    with pytest.raises(ValueError, match="must be finite"):
        _call(block, [bad, 1.0, 0.0, 0.0], source="cam-b", scope=scope, max_sources=1)
    got = [_scalars(_call(block, row, scope=scope, max_sources=1)) for row in A[:3]]
    with pytest.raises(ValueError, match="must be finite"):
        _call(block, [1.0, bad, 0.0, 0.0], scope=scope, max_sources=1)
    got += [_scalars(_call(block, row, scope=scope, max_sources=1)) for row in A[3:6]]

    assert got == expected
    assert block._states.get(("source", "cam-b")) is None


@pytest.mark.parametrize(
    "changed",
    [torch.tensor(A[1][:3]), torch.tensor(A[1], dtype=torch.float64)],
    ids=["length", "dtype"],
)
def test_an_embedding_unlike_the_source_fails_and_changes_nothing(changed):
    fresh, block = IdentifyChangesBlock(), IdentifyChangesBlock()
    expected = [_scalars(_call(fresh, row)) for row in A[:5]]
    got = [_scalars(_call(block, row)) for row in A[:2]]

    with pytest.raises(ValueError, match="does not match this source"):
        _call(block, changed)

    got += [_scalars(_call(block, row)) for row in A[2:5]]
    assert got == expected
    assert _call(block, changed, source="cam-b")["warming_up"] is True


def test_the_strategy_parameters_it_needs_must_be_present():
    block = IdentifyChangesBlock()

    with pytest.raises(ValueError, match="needs smoothing_factor"):
        _call(block, A[0], smoothing_factor=None)
    with pytest.raises(ValueError, match="needs window_size"):
        _call(block, A[0], strategy=SLIDING_WINDOW, window_size=None)

    _call(block, A[0], strategy=SMA, smoothing_factor=None, window_size=None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold_percentile": 1.5},
        {"threshold_percentile": True},
        {"smoothing_factor": -0.1},
        {"strategy": SLIDING_WINDOW, "window_size": 0},
    ],
)
def test_selected_parameters_are_checked_before_any_update(overrides):
    block = IdentifyChangesBlock()

    with pytest.raises(ValueError):
        _call(block, A[0], **overrides)

    assert _call(block, A[0])["warming_up"] is True
    assert _call(block, A[1])["average"].shape == (4,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold_percentile": 1.5},
        {"smoothing_factor": 2.0},
        {"window_size": 0},
        {"strategy": "EMA"},
        {"scope": "camera"},
        {"max_sources": 0},
    ],
)
def test_literal_parameters_are_checked_when_the_workflow_compiles(overrides):
    with pytest.raises(ValidationError):
        IdentifyChangesBlock.Params(embedding="$inputs.embedding", **overrides)


def test_defaults_match_v1():
    params = IdentifyChangesBlock.Params(embedding="$inputs.embedding")

    assert (
        params.strategy,
        params.threshold_percentile,
        params.warmup,
        params.smoothing_factor,
        params.window_size,
    ) == (EMA, 0.2, 3, 0.1, 10)
    assert (params.scope, params.max_sources) == ("source", 256)
