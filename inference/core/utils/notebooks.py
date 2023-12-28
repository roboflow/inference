import os
import subprocess

import requests

from inference.core.env import NOTEBOOK_PASSWORD, NOTEBOOK_PORT


def check_notebook_is_running():
    try:
        response = requests.get(f"http://localhost:{NOTEBOOK_PORT}/")
        return response.status_code == 200
    except:
        return False


def start_notebook():
    if not check_notebook_is_running():
        os.makedirs("/notebooks", exist_ok=True)
        subprocess.Popen(
            f"jupyter-lab --allow-root --port={NOTEBOOK_PORT} --ip=0.0.0.0 --notebook-dir=/notebooks --NotebookApp.token='{NOTEBOOK_PASSWORD}' --NotebookApp.password='{NOTEBOOK_PASSWORD}'".split(
                " "
            )
        )
