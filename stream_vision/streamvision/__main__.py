"""Run the stream manager standalone: `python -m streamvision --host-factory ...`."""

import argparse
import importlib.util
from typing import List, Optional, Tuple

from streamvision.stream.configuration import get_configuration
from streamvision.stream_manager.manager_app.bootstrap import install_process_settings
from streamvision.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    import_attribute,
)


def _parse_setting(value: str) -> Tuple[str, str]:
    key, separator, setting = value.partition("=")
    if not (key and separator):
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")

    return key, setting


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser of the standalone stream manager.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="python -m streamvision")
    parser.add_argument(
        "--host-factory",
        required=True,
        help="Pipeline host factory as 'package.module:attribute'.",
    )
    parser.add_argument(
        "--host-setting",
        type=_parse_setting,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="String keyword argument for the host factory (repeatable).",
    )
    parser.add_argument(
        "--warm-pipelines",
        type=_non_negative_int,
        default=0,
        help="Number of idle pipeline processes kept ready.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Launch the stream manager; its address comes from `STREAM_MANAGER_*` env.

    Args:
        argv: Command-line arguments; `sys.argv[1:]` when `None`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if importlib.util.find_spec("aiortc") is None:
        parser.error(
            "the stream manager needs the webrtc extra: pip install 'streamvision[webrtc]'"
        )
    if importlib.util.find_spec("roboflow_workflows") is None:
        parser.error(
            "the stream manager needs the workflows extra: "
            "pip install 'streamvision[webrtc,workflows]'"
        )
    host_descriptor = PipelineHostDescriptor(
        factory=args.host_factory,
        settings=dict(args.host_setting),
    )

    # The host module may install its own configuration on import, so import it first.
    import_attribute(host_descriptor.factory)
    # Returns what the host installed, else installs and returns the defaults.
    configuration = get_configuration()
    install_process_settings(
        configuration=configuration,
        host_descriptor=host_descriptor,
    )

    from streamvision.stream_manager.manager_app.app import start

    start(
        expected_warmed_up_pipelines=args.warm_pipelines,
        host_descriptor=host_descriptor,
    )


if __name__ == "__main__":
    main()
