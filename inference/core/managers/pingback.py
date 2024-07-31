import time
import traceback

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from inference.core.devices.utils import GLOBAL_DEVICE_ID, GLOBAL_INFERENCE_SERVER_ID
from inference.core.env import (
    API_KEY,
    METRICS_ENABLED,
    METRICS_INTERVAL,
    METRICS_URL,
    TAGS,
)
from inference.core.logger import logger
from inference.core.managers.metrics import (
    get_inference_results_for_model,
    get_system_info,
)
from inference.core.utils.requests import api_key_safe_raise_for_status
from inference.core.utils.url_utils import wrap_url
from inference.core.version import __version__


class PingbackInfo:
    """Class responsible for managing pingback information for Roboflow.

    This class initializes a scheduler to periodically post data to Roboflow, containing information about the models,
    container, and device.

    Attributes:
        scheduler (BackgroundScheduler): A scheduler for running jobs in the background.
        model_manager (ModelManager): Reference to the model manager object.
        process_startup_time (str): Unix timestamp indicating when the process started.
        METRICS_URL (str): URL to send the pingback data to.
        system_info (dict): Information about the system.
        window_start_timestamp (str): Unix timestamp indicating the start of the current window.
    """

    def __init__(self, manager):
        """Initializes PingbackInfo with the given manager.

        Args:
            manager (ModelManager): Reference to the model manager object.
        """
        try:
            self.scheduler = BackgroundScheduler(
                job_defaults={"coalesce": True, "max_instances": 1}
            )
            self.model_manager = manager
            self.process_startup_time = str(int(time.time()))
            logger.debug(
                "UUID: " + self.model_manager.uuid
            )  # To correlate with UI container view
            self.window_start_timestamp = str(int(time.time()))
            context = {
                "api_key": API_KEY,
                "timestamp": str(int(time.time())),
                "device_id": GLOBAL_DEVICE_ID,
                "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
                "inference_server_version": __version__,
                "tags": TAGS,
            }
            self.environment_info = context | get_system_info()

            # we will set this from model manager when a new api key is used
            # to use in case there is no global ENV api key configured
            self.fallback_api_key = None

        except Exception as e:
            logger.debug(
                "Error sending pingback to Roboflow, if you want to disable this feature unset the ROBOFLOW_ENABLED environment variable. "
                + str(e)
            )

    def start(self):
        """Starts the scheduler to periodically post data to Roboflow.

        If METRICS_ENABLED is False, a warning is logged, and the method returns without starting the scheduler.
        """
        if METRICS_ENABLED == False:
            logger.warning(
                "Metrics reporting to Roboflow is disabled; not sending back stats to Roboflow."
            )
            return
        try:
            self.scheduler.add_job(
                self.post_data,
                "interval",
                seconds=METRICS_INTERVAL,
                args=[self.model_manager],
                replace_existing=True,
            )
            self.scheduler.start()
        except Exception as e:
            logger.debug(e)

    def stop(self):
        """Stops the scheduler."""
        self.scheduler.shutdown()

    def post_data(self, model_manager):
        """Posts data to Roboflow about the models, container, device, and other relevant metrics.

        Args:
            model_manager (ModelManager): Reference to the model manager object.

        The data is collected and reset for the next window, and a POST request is made to the pingback URL.
        """
        all_data = self.environment_info.copy()
        all_data["inference_results"] = []

        # use fallback api key if env didn't have one
        if self.fallback_api_key and not all_data.get("api_key"):
            all_data["api_key"] = self.fallback_api_key

        try:
            now = time.time()
            start = now - METRICS_INTERVAL
            for model_id in model_manager.models():
                results = get_inference_results_for_model(
                    GLOBAL_INFERENCE_SERVER_ID, model_id, min=start, max=now
                )
                all_data["inference_results"] = all_data["inference_results"] + results
            res = requests.post(wrap_url(METRICS_URL), json=all_data, timeout=10)
            try:
                api_key_safe_raise_for_status(response=res)
                logger.debug(
                    "Sent metrics to Roboflow {} at {}.".format(
                        METRICS_URL, str(all_data)
                    )
                )
            except Exception as e:
                logger.debug(
                    f"Error sending metrics to Roboflow, if you want to disable this feature unset the METRICS_ENABLED environment variable."
                )

        except Exception as e:
            try:
                logger.debug(
                    f"Error sending metrics to Roboflow, if you want to disable this feature unset the METRICS_ENABLED environment variable. Error was: {e}. Data was: {all_data}"
                )
                traceback.print_exc()

            except Exception as e2:
                logger.debug(
                    f"Error sending metrics to Roboflow, if you want to disable this feature unset the METRICS_ENABLED environment variable. Error was: {e}."
                )
