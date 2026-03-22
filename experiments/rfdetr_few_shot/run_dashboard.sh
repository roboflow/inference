#!/bin/bash
# Auto-restarting dashboard wrapper
# Restarts on crash, logs to /tmp/dashboard_8766.log
cd /home/roboflow/inference/experiments/rfdetr_few_shot
while true; do
    echo "[$(date)] Starting dashboard on :8766..." >> /tmp/dashboard_8766.log
    ~/.pyenv/versions/inference-exp/bin/python \
        /home/roboflow/inference/experiments/rfdetr_few_shot/server.py \
        --db results_finetune_baseline.db \
        --compare-db results_phase3b.db \
        --host 0.0.0.0 --port 8766 \
        >> /tmp/dashboard_8766.log 2>&1
    echo "[$(date)] Dashboard exited (code=$?), restarting in 3s..." >> /tmp/dashboard_8766.log
    sleep 3
done
