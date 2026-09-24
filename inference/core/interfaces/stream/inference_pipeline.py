"""Historical name of `inference.core.interfaces.legacy_stream.inference_pipeline`.

The implementation needs the `inference` server (models, model managers or the
Roboflow platform), so it lives outside the host-neutral stream runtime. This
name is an exact alias rather than a re-export: importing it yields the
implementation module object itself, so an attribute read, set or patched
through either name is the same attribute.
"""

import sys

from inference.core.interfaces.legacy_stream import (
    inference_pipeline as _implementation,
)

sys.modules[__name__] = _implementation
