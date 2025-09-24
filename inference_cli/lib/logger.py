import logging

from rich.logging import RichHandler

from inference_cli.lib.env import CLI_LOG_LEVEL

CLI_LOGGER = logging.getLogger("inference-cli")
CLI_LOGGER.setLevel(CLI_LOG_LEVEL)
CLI_LOGGER.addHandler(RichHandler())
