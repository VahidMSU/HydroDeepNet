#!/bin/bash
# Build the Docker image
cd /home/ubuntu/repos/HydroDeepNet
docker stop $(docker ps -q) || true
docker system prune -a -f
docker build -t hydrodeepnet_image .

mkdir -p /home/ubuntu/data/SWATGenXApp/GenXAppData
mkdir -p /home/ubuntu/data/SWATGenXApp/Users
mkdir -p /home/ubuntu/data/SWATGenXApp/codes

docker run --privileged -p 5000:5000 \
  -v /home/ubuntu/data/SWATGenXApp/GenXAppData:/data/SWATGenXApp/GenXAppData \
  -v /home/ubuntu/data/SWATGenXApp/Users:/data/SWATGenXApp/Users \
  -v /home/ubuntu/data/SWATGenXApp/codes:/data/SWATGenXApp/codes \
  hydrodeepnet_image
