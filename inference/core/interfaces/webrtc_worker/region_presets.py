"""Region presets that map a frontend-facing region key to a deterministic
``(cloud, region)`` pair on Modal.

The frontend exposes a small set of region keys (e.g. ``us``, ``eu``, ``ap``)
and each maps to one specific cloud + region pair so that:

* The WebRTC GPU container always lands on a known cloud/region (no more
  "I asked for ``eu`` and Modal placed me on OCI Frankfurt").
* The matching ``webexec-<cloud>-<region>`` app can be pre-deployed for
  same-region Custom Python Block execution.

Usage
-----

Frontend / SDK callers should send ``requested_region`` set to one of the
keys in :data:`WEBRTC_REGION_PRESETS`. The dispatcher resolves it via
:func:`resolve_region_preset` and pins ``cloud=`` + ``region=`` on the
Modal class.

If you add a new preset here, also deploy a webexec for it (see
``inference/modal/deploy_modal_app.py`` and
``inference/modal/deploy_all_webexecs.py``).
"""

from typing import Dict, NamedTuple, Optional


class RegionPreset(NamedTuple):
    cloud: str
    region: str


# Frontend-facing region keys -> deterministic (cloud, region) pair.
#
# Short keys (``us``, ``eu``, ``ap``) match the values currently emitted by
# the workflow code generators in the roboflow app and the inference SDK,
# so existing callers keep working but now get deterministic placement.
#
# Cloud choice matters: pin a region to a cloud where the GPUs you actually
# spawn are available on Modal. L40S in particular is broadly available on
# OCI (eu-frankfurt-1, ap-tokyo-1) but only in a couple of AWS regions
# (us-east-1 / us-east-2). If you map ``eu`` -> AWS Ireland the spawn will
# silently sit in the queue forever because Modal can't find an L40S worker
# there, and the client will time out without ever showing a function call
# in the dashboard. So ``eu`` -> OCI Frankfurt where capacity actually
# exists. See https://modal.com/docs/guide/gpu#gpu-availability.
#
# Long keys (``oci-eu-frankfurt-1`` etc.) let advanced callers pick a
# specific cloud+region directly. They're also what
# :func:`_resolve_webexec_app_name` reconstructs from
# ``MODAL_CLOUD_PROVIDER`` + ``MODAL_REGION`` to find the matching app.
WEBRTC_REGION_PRESETS: Dict[str, RegionPreset] = {
    # Short, user-friendly keys.
    "us": RegionPreset(cloud="aws", region="us-east-1"),
    "eu": RegionPreset(cloud="oci", region="eu-frankfurt-1"),
    "ap": RegionPreset(cloud="oci", region="ap-tokyo-1"),
    # Specific cloud+region keys (callers who care). Each entry MUST be
    # a (cloud, region) Modal supports for the GPUs you spawn.
    "aws-us-east-1": RegionPreset(cloud="aws", region="us-east-1"),
    "oci-eu-frankfurt-1": RegionPreset(cloud="oci", region="eu-frankfurt-1"),
    "oci-ap-tokyo-1": RegionPreset(cloud="oci", region="ap-tokyo-1"),
}


def resolve_region_preset(key: Optional[str]) -> Optional[RegionPreset]:
    """Return the ``(cloud, region)`` pair for a preset key, or ``None``
    if the key is empty or unknown.
    """
    if not key:
        return None
    return WEBRTC_REGION_PRESETS.get(key)


def webexec_app_name_for(preset: RegionPreset) -> str:
    """Return the webexec app name that backs a given preset."""
    return f"webexec-{preset.cloud}-{preset.region}"
