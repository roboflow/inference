"""Legacy import entrypoint for ``inference.core.interfaces.stream``.

The source moved to the ``streamvision`` distribution. This shim installs the
compat finder and rebinds this module in ``sys.modules`` to the canonical
``streamvision.stream`` package so the two names share one object.
"""

from inference._workflows_compat import alias_legacy_root as _alias_legacy_root

_alias_legacy_root(__name__)
