"""Historical name of `inference.core.interfaces.legacy_stream.stream`.

The implementation needs the `inference` server (models, model managers or the
Roboflow platform), so it lives outside the host-neutral stream runtime. This
name is an exact alias rather than a re-export: importing it yields the
implementation module object itself, so an attribute read, set or patched
through either name is the same attribute. The `TYPE_CHECKING`-guarded
wildcard import below exists only so static analyzers and IDEs see the
historical names; it never executes at runtime.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # static analyzers only; at runtime this module object IS the target
    from inference.core.interfaces.legacy_stream.stream import *  # noqa: F401,F403

import sys

from inference.core.interfaces.legacy_stream import stream as _implementation

sys.modules[__name__] = _implementation
