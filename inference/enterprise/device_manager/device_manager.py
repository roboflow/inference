from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel

from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.enterprise.device_manager.container_service import ContainerService
from inference.enterprise.device_manager.metrics_service import MetricsService

class Command(BaseModel):
    container_id: str
    command: str

app = FastAPI(
    title="Roboflow Device Manager",
    description="The device manager enables remote control and monitoring of Roboflow inference server containers",
    version="0.1.0",
    terms_of_service="https://roboflow.com/terms",
    contact={
        "name": "Roboflow Inc.",
        "url": "https://roboflow.com/contact",
        "email": "help@roboflow.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    root_path="/",
)

container_service = ContainerService()
metrics_service = MetricsService()

@app.get("/")
def root_route():
    return {
        "name": "Roboflow Device Manager",
        "version": "0.1.0",
        "terms_of_service": "https://roboflow.com/terms",
        "contact": {
            "name": "Roboflow Inc.",
            "url": "https://roboflow.com/contact",
            "email": "help@roboflow.com",
        },
        "license_info": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
    }

@app.post("/exec_command")
def exec_command(command: Command):
    handle_remote_command(command.dict())
    return {"status": "ok"}

def check_for_commands():
    pass

def handle_remote_command(cmd_payload):
    container_id = cmd_payload.get("container_id")
    container = container_service.get_container_by_id(container_id)
    if not container:
        print(f"Container with id {container_id} not found")
        return
    cmd = cmd_payload.get("command")
    if cmd == "restart":
        container.restart()
    elif cmd == "stop":
        container.stop()
    elif cmd == "ping":
        container.ping()
    else:
        print("Unknown command: {}".format(cmd))


scheduler = BackgroundScheduler()
# scheduler.add_job(
#     check_for_commands,
#     "interval",
#     seconds=5
# )
scheduler.add_job(
    metrics_service.report_metrics,
    "interval",
    seconds=5
)
scheduler.start()