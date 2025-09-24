import os
import subprocess
import tempfile

import yaml

CONFIG = {"protocols": ["tcp"], "paths": {"all": {"source": "publisher"}}}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "rtsp-simple-server.yml")
        with open(config_path, "w") as f:
            yaml.dump(CONFIG, f)
        run_server(config_path=config_path)


def run_server(config_path: str) -> None:
    command = (
        f"docker run --rm --name rtsp_server -v {config_path}:/rtsp-simple-server.yml "
        f"-p 8554:8554 aler9/rtsp-simple-server:v1.3.0"
    )
    return_code = subprocess.run(command.split()).returncode
    if return_code != 0:
        raise RuntimeError("Could not run RTSP server!")


if __name__ == '__main__':
    main()
