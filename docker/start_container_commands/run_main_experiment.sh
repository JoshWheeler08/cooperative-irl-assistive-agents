#!/bin/bash
# Run main experiments in Docker container
# Usage: Update WORKSPACE_PATH to match your local repository path

WORKSPACE_PATH="$(pwd)/../../code/src"

docker run \
  -v "${WORKSPACE_PATH}:${WORKSPACE_PATH}" \
  -w "${WORKSPACE_PATH}" \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --user $(id -u):$(id -g) \
  --rm \
  irl-intention-recognition \
  python experiments/run_main.py configuration/main_experiment_config.yaml
