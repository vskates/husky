#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="${HOME}/HuskyRLGrasp"
BAG_PATH="${1:-${REPO_DIR}/data}"
SEG_MODEL_PATH="${2:-${REPO_DIR}/models/yolo.pt}"
FRAME_INDEX="${3:-0}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repository not found: ${REPO_DIR}" >&2
  exit 1
fi

if [[ ! -d "${BAG_PATH}" && ! -f "${BAG_PATH}" ]]; then
  echo "Bag path not found: ${BAG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SEG_MODEL_PATH}" ]]; then
  echo "YOLO weights not found: ${SEG_MODEL_PATH}" >&2
  echo "Place the weights at ~/HuskyRLGrasp/models/yolo.pt or pass a custom path as the second argument." >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash

cd "${REPO_DIR}"
if [[ ! -f install/setup.bash ]]; then
  echo "ROS workspace is not built yet. Run:" >&2
  echo "  source /opt/ros/jazzy/setup.bash && colcon build --base-paths src/ws_grasp --packages-select grasp_inference_pkg --symlink-install" >&2
  exit 1
fi
source install/setup.bash

ros2 launch grasp_inference_pkg diffusion_grasp_bag_test.launch.py \
  bag_path:="${BAG_PATH}" \
  seg_model_path:="${SEG_MODEL_PATH}" \
  frame_index:="${FRAME_INDEX}"
