#!/bin/bash

docker_image_name="mnist-cnn"
docker_image_version="latest"
docker_image_id="${docker_image_name}:${docker_image_version}"

echo "Building Docker image: ${docker_image_id}"
docker build -t ${docker_image_id} .
docker image prune -f

