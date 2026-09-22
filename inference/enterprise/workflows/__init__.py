"""Legacy import entrypoint for ``inference.enterprise.workflows``.

The old enterprise root was empty; only its ``enterprise_blocks`` subtree
carried code (moved into ``roboflow_workflows.enterprise_blocks``). This
package deliberately does not alias to the canonical root — it stays empty —
but it installs the compat finder so child imports of ``enterprise_blocks``
resolve to canonical modules.
"""

from inference._workflows_compat import install as _install

_install()
