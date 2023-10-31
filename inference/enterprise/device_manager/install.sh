#!/bin/bash
set -e

#TODO:use correct inference server for device (detect jetpack version)

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Exiting."
   exit 1
fi

ENV_FILE="/root/.roboflow"


uninstall() {
  local name=$1
  
  echo "Stopping and disabling $name."
  systemctl stop $name || echo 'already stopped'
  systemctl disable $name || echo 'already disabled'
  echo "Removing systemd service file for $name."
  rm -f /etc/systemd/system/$name.service
  systemctl daemon-reload
}

create_service_file() {
  local name=$1
  local docker_args=$2

  cat > /etc/systemd/system/$name.service << EOL
[Unit]
Description=Run $name
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/bin/docker run -t --rm --name $name --env-file $ENV_FILE -e METRICS_ENABLED=true -e METRICS_INTERVAL=60 $docker_args
ExecStop=/usr/bin/docker stop $name
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOL
}

start_or_restart_service() {
  local name=$1

  if systemctl --quiet is-active "$name"; then
    echo "$name already running. Restarting."
    systemctl restart "$name"
  else
    echo "Installing and starting $name."
    systemctl enable "$name"
    systemctl start "$name"
  fi
}

if [[ "$1" == "uninstall" ]]; then
  uninstall "roboflow-inference"
  uninstall "roboflow-device-manager"
  uninstall "roboflow-redis"
  rm -f $ENV_FILE
  echo "Services and containers removed."
  exit 0
fi


if [[ ! -f $ENV_FILE ]]; then
  read -p "Please enter your Roboflow API key: " API_KEY
  read -p "Please enter your Device Manager username: " DEVICEMGR_USER
  read -p -s "Please enter your Device Manager password: " DEVICEMGR_PASS
  cat >> $ENV_FILE << EOL
ROBOFLOW_API_KEY=$API_KEY
API_KEY=$API_KEY
INFERENCE_SERVER_ID=roboflow-inference
NUM_WORKERS=1
REDIS_HOST=0.0.0.0
REDIS_PORT=6379
API_BASE_URL=https://api.roboflow.one
DEVICE_MANAGER_USERNAME=$DEVICEMGR_USER
DEVICE_MANAGER_PASSWORD=$DEVICEMGR_PASS
DEVICE_MANAGER_PUBSUB_HOST=z0907231.ala.us-east-1.emqxsl.com
DEVICE_ALIAS=$(hostname)
DEVICE_ID=$(openssl rand -hex 16)
EOL
  echo "Wrote environment variables to $ENV_FILE"
fi

create_service_file "roboflow-inference" "--privileged --net=host --runtime=nvidia --gpus=all --mount source=roboflow,target=/tmp/cache roboflow/roboflow-inference-server-jetson-4.5.0:latest"
create_service_file "roboflow-device-manager" "--net=host -v /var/run/docker.sock:/var/run/docker.sock roboflow/roboflow-device-manager:latest"
create_service_file "roboflow-redis" "--net=host redis:alpine3.18"

systemctl daemon-reload

start_or_restart_service "roboflow-inference.service"
start_or_restart_service "roboflow-device-manager.service"
start_or_restart_service "roboflow-redis.service"

echo "Done."

