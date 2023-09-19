from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

from inference.enterprise.device_manager.command_handler import (
    RemoteCommandHandler,
    Command,
)
from inference.enterprise.device_manager.metrics_service import MetricsService


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

metrics_service = MetricsService()
remote_commands_handler = RemoteCommandHandler()


@app.get("/")
def root():
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
    remote_commands_handler.handle_command(command)
    return {"status": "ok"}


scheduler = BackgroundScheduler()
scheduler.add_job(remote_commands_handler.fetch_commands, "interval", seconds=5)
scheduler.add_job(metrics_service.report_metrics, "interval", seconds=5)
scheduler.start()
