#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: ./docker-startup.sh <build|deploy|deploy-gpu>"
  exit 1
fi

IMAGE_NAME="run-deepseek"

if [ "$1" = "build" ]; then
  docker build -t $IMAGE_NAME .
elif [ "$1" = "deploy" ]; then
  docker run --rm --name $IMAGE_NAME \
    -v "$PWD":/root \
    -p 11434:11434 -p 8501:8501 \
    -it $IMAGE_NAME
elif [ "$1" = "deploy-gpu" ]; then
  docker run --rm --name $IMAGE_NAME \
    -d --gpus=all \
    -v "$PWD":/root \
    -p 11434:11434 -p 8501:8501 \
    -it $IMAGE_NAME
fi