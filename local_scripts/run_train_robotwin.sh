#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

CONFIG_NAME="${1:?Usage: $0 <config_name>}"

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}"
