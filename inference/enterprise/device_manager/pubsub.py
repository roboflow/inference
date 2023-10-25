import time
import uuid
import json

from paho.mqtt import client as mqtt_client
from inference.enterprise.device_manager.command_handler import handle_command
from inference.core.env import (
    API_KEY,
    MESSAGE_BROKER_HOST,
)
from inference.core.logger import logger
from inference.enterprise.device_manager.helpers import (
    get_device_id,
)

METRICS_TOPIC = f"roboflow/device-management/v1/{API_KEY}/metrics"
COMMANDS_TOPIC = f"roboflow/device-management/v1/{API_KEY}/commands"
STREAM_TOPIC = f"roboflow/device-management/v1/{API_KEY}/stream"
SYS_TOPIC = f"roboflow/device-management/v1/{API_KEY}/system"

MAX_RECONNECT_COUNT = 12
RECONNECT_RATE = 2
MAX_RECONNECT_DELAY = 60

CLIENT_ID = f"device-managed-{hex(uuid.getnode())}"


def on_connect(client, userdata, flags, rc):
    if rc == 0 and client.is_connected():
        logger.info("Connection established")
        client.subscribe(COMMANDS_TOPIC)
    else:
        logger.error("Failed to connect to messaging broker, return code %d\n" % rc)


def on_disconnect(client, userdata, rc):
    reconnect_count, reconnect_delay = 0, 1
    while reconnect_count < MAX_RECONNECT_COUNT:
        time.sleep(reconnect_delay)
        try:
            client.reconnect()
            return
        except Exception as err:
            logger.warn("%s. Reconnect failed. Retrying...", err)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1

    logger.error("Reconnect failed after %s attempts. Exiting...", reconnect_count)


def on_message(client, userdata, msg):
    if msg.topic == COMMANDS_TOPIC:
        logger.info(f"Received `{msg.payload.decode()}` from commands topic")
        payload = json.loads(msg.payload.decode())
        handle_command(payload)


def connect_mqtt():
    client = mqtt_client.Client(CLIENT_ID, clean_session=False)
    client.tls_set(ca_certs="./inference/enterprise/device_manager/server.crt")
    client.username_pw_set("testuser1", "foobarqux")
    client.enable_logger(logger=logger)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.will_set(
        SYS_TOPIC,
        payload=json.dumps({"device_id": get_device_id(), "type": "offline"}),
        qos=1,
        retain=False,
    )
    client.connect(MESSAGE_BROKER_HOST, 8883, keepalive=60)
    return client


client = connect_mqtt()


def dispatch(topic, message, qos=0):
    client.loop_start()
    time.sleep(1)
    if client.is_connected():
        result = client.publish(topic, json.dumps(message), qos=qos)
        logger.info(f"Message published result: {topic} {result}")
    else:
        logger.error("Failed to connect to message broker")
        client.loop_stop()
