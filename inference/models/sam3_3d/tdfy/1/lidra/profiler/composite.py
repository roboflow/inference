from typing import List, Union
from lightning.pytorch.profilers import (
    Profiler,
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    XLAProfiler,
)


class Composite(Profiler):
    PROFILERS = {
        "simple": SimpleProfiler,
        "advanced": AdvancedProfiler,
        "pytorch": PyTorchProfiler,
        "xla": XLAProfiler,
    }

    def __init__(
        self,
        profilers: Union[str, List[Profiler]],
        dirpath: str = "profs",
    ):
        super().__init__(dirpath=dirpath, filename=None)

        def init_profiler(profiler):
            if isinstance(profiler, str):
                klass = Composite.PROFILERS[profiler]
                profiler = klass(dirpath=dirpath, filename=f"{profiler}.prof")
            return profiler

        profilers = [init_profiler(p) for p in profilers]
        self._profilers = {type(p).__name__: p for p in profilers}

    def describe(self):
        for profiler in self._profilers.values():
            profiler.describe()

    def start(self, action_name):
        for profiler in self._profilers.values():
            profiler.start(action_name)

    def stop(self, action_name):
        for profiler in self._profilers.values():
            profiler.stop(action_name)

    def summary(self):
        result = [f"\nComposite Profiler Summary: \n"]

        for name, profiler in self._profilers.items():
            result.append(f"\n ===== [{name}] ===== \n\n")
            result.append(profiler.summary())

        result = "".join(result)
        return result

    def teardown(self, stage):
        for profiler in self._profilers.values():
            profiler.teardown(stage)
        super().teardown(stage=stage)
