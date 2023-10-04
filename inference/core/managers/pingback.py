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
from inference.core.managers.metrics import get_model_metrics, get_system_info
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
            self.scheduler = BackgroundScheduler()
            self.model_manager = manager
            self.process_startup_time = str(int(time.time()))
            logger.debug(
                "UUID: " + self.model_manager.uuid
            )  # To correlate with UI container view
            self.METRICS_URL = METRICS_URL  # Test URL

            self.system_info = get_system_info()
            self.window_start_timestamp = str(int(time.time()))
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
            logger.debug(
                "Metrics reporting to Roboflow is disabled; not sending back stats to Roboflow."
            )
            return
        try:
            self.scheduler.add_job(
                self.post_data,
                "interval",
                seconds=METRICS_INTERVAL,
                args=[self.model_manager],
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
        all_data = {}
        try:
            all_data = {
                "api_key": API_KEY,
                "timestamp": self.window_start_timestamp,
                "device": {
                    "id": GLOBAL_DEVICE_ID,
                    "name": GLOBAL_DEVICE_ID,
                    "type": f"roboflow-inference-server=={__version__}",
                    "tags": TAGS,
                    "system_info": self.system_info,
                    "containers": [
                        {
                            "startup_time": self.process_startup_time,
                            "uuid": GLOBAL_INFERENCE_SERVER_ID,
                            "models": [],
                        }
                    ],
                },
            }
            now = time.time()
            start = now - METRICS_INTERVAL
            for key in model_manager._models:
                model = model_manager._models[key]
                if all_data["api_key"] is None and model.api_key is not None:
                    all_data["api_key"] = model.api_key
                model_data = {
                    "api_key": model.api_key,
                    "dataset_id": model.dataset_id,
                    "version": model.version_id,
                    "metrics": get_model_metrics(
                        GLOBAL_INFERENCE_SERVER_ID, key, min=start
                    ),
                }
                all_data["device"]["containers"][0]["models"].append(model_data)

            timestamp = str(int(time.time()))
            all_data["timestamp"] = timestamp
            self.window_start_timestamp = timestamp
            res = requests.post(METRICS_URL, json=all_data)
            try:
                res.raise_for_status()
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
