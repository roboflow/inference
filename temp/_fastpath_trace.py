"""Rigorous kill-switch trace for the widen-scope fast path.

Patches three surfaces in inference_models.models.rfdetr.pre_processing:
  - triton_path_eligible   (did the gate say yes or no?)
  - triton_preprocess_rfdetr_stretch (did the kernel fire?)
  - triton_path_preprocess (did the fast-path code path actually run end-to-end?)

Also patches the name `pre_process_numpy_image` used by the PIL fallback so
we can count PIL-fallback invocations per-image (the PIL path iterates one
image at a time inside pre_process_network_input's loop).

Import + call install() before model construction; counts print at exit.
"""
import atexit

COUNTERS = {
    "eligible_true":  0,
    "eligible_false": 0,
    "fastpath_runs":  0,
    "kernel_calls":   0,
    "pil_numpy_calls": 0,
    "pil_tensor_calls": 0,
}

_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    import inference_models.models.rfdetr.pre_processing as pp

    orig_eligible = pp.triton_path_eligible
    def traced_eligible(*a, **kw):
        r = orig_eligible(*a, **kw)
        COUNTERS["eligible_true" if r else "eligible_false"] += 1
        return r
    pp.triton_path_eligible = traced_eligible

    orig_fast = pp.triton_path_preprocess
    def traced_fast(*a, **kw):
        COUNTERS["fastpath_runs"] += 1
        return orig_fast(*a, **kw)
    pp.triton_path_preprocess = traced_fast

    orig_kernel = pp.triton_preprocess_rfdetr_stretch
    if orig_kernel is not None:
        def traced_kernel(*a, **kw):
            COUNTERS["kernel_calls"] += 1
            return orig_kernel(*a, **kw)
        pp.triton_preprocess_rfdetr_stretch = traced_kernel

    orig_pil_np = pp._pre_process_numpy
    def traced_pil_np(*a, **kw):
        COUNTERS["pil_numpy_calls"] += 1
        return orig_pil_np(*a, **kw)
    pp._pre_process_numpy = traced_pil_np

    orig_pil_t = pp._pre_process_tensor
    def traced_pil_t(*a, **kw):
        COUNTERS["pil_tensor_calls"] += 1
        return orig_pil_t(*a, **kw)
    pp._pre_process_tensor = traced_pil_t

    atexit.register(_print_counters)


def _print_counters():
    import os
    env = os.environ.get("USE_TRITON_FOR_PREPROCESSING", "<unset>")
    print(f"\n[fastpath-trace env={env}]")
    for k, v in COUNTERS.items():
        print(f"  {k:<22s}: {v}")
