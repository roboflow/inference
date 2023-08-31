import json
import logging
import platform
import re
import socket
import time
import uuid

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from inference.core.devices.utils import get_device_id
from inference.core.env import (
    API_KEY,
    PINGBACK_ENABLED,
    PINGBACK_INTERVAL_SECONDS,
    PINGBACK_URL,
)
from inference.core.logger import logger


def getSystemInfo():
    """Collects system information such as platform, architecture, hostname, IP address, MAC address, and processor details.

    Returns:
        dict: A dictionary containing detailed system information.
    """
    info = {}
    try:
        info["platform"] = platform.system()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip_address"] = socket.gethostbyname(socket.gethostname())
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)
    finally:
        return info


class PingbackInfo:
    """Class responsible for managing pingback information for Roboflow.

    This class initializes a scheduler to periodically post data to Roboflow, containing information about the models,
    container, and device.

    Attributes:
        scheduler (BackgroundScheduler): A scheduler for running jobs in the background.
        model_manager (ModelManager): Reference to the model manager object.
        process_startup_time (str): Unix timestamp indicating when the process started.
        pingback_url (str): URL to send the pingback data to.
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
            logger.info(
                "UUID: " + self.model_manager.uuid
            )  # To correlate with UI container view
            self.pingback_url = PINGBACK_URL  # Test URL

            self.system_info = getSystemInfo()
            self.window_start_timestamp = str(int(time.time()))
        except Exception as e:
            logger.error(
                "Error sending pingback to Roboflow, if you want to disable this feature unset the ROBOFLOW_ENABLED environment variable. "
                + str(e)
            )

    def start(self):
        """Starts the scheduler to periodically post data to Roboflow.

        If PINGBACK_ENABLED is False, a warning is logged, and the method returns without starting the scheduler.
        """
        if PINGBACK_ENABLED == False:
            logger.warn(
                "Pingback to Roboflow is disabled; not sending back stats to Roboflow."
            )
            return
        try:
            self.scheduler.add_job(
                self.post_data,
                "interval",
                seconds=PINGBACK_INTERVAL_SECONDS,
                args=[self.model_manager],
            )
            self.scheduler.start()
        except Exception as e:
            print(e)

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
                "api_key": API_KEY or "no_model_used",
                "container": {
                    "startup_time": self.process_startup_time,
                    "uuid": self.model_manager.uuid,
                },
                "models": [],
                "window_start_timestamp": self.window_start_timestamp,
                "device": {
                    "id": get_device_id(),
                    "name": get_device_id(),
                    "type": "inference_server",
                    "tags": [],
                    "system_info": self.system_info,
                },
                "num_errors": self.model_manager.num_errors,
            }
            for key in model_manager._models:
                post_data = {}
                model = model_manager._models[key]
                all_data["api_key"] = model.api_key
                post_data["model"] = {
                    "api_key": model.api_key,
                    "dataset_id": model.dataset_id,
                    "version": model.version_id,
                }
                post_data["data"] = {}
                post_data["data"]["metrics"] = {
                    "num_inferences": model.metrics["num_inferences"],
                    "avg_inference_time": model.metrics["avg_inference_time"]
                    / model.metrics["num_inferences"]
                    if model.metrics["num_inferences"] > 0
                    else 0,
                    "num_errors": self.model_manager.num_errors,  # This is not really per model, its per container; kept this for v1
                }
                all_data["models"].append(post_data)
                # Reset metrics
                model.metrics["num_inferences"] = 0
                model.metrics["avg_inference_time"] = 0
                self.model_manager.num_errors = 0

            timestamp = str(int(time.time()))
            all_data["timestamp"] = timestamp
            self.window_start_timestamp = timestamp
            requests.post(PINGBACK_URL, json=all_data)
            logger.info(
                "Sent pingback to Roboflow {} at {}.".format(
                    PINGBACK_URL, str(all_data)
                )
            )

        except Exception as e:
            logger.error(
                "Error sending pingback to Roboflow, if you want to disable this feature unset the ROBOFLOW_ENABLED environment variable. "
                + str(e)
            )
