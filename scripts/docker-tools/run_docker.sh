#!/bin/bash
# Build the Docker image
cd /data/SWATGenXApp/codes
docker stop $(docker ps -q)
docker system prune -a -f
docker build -t hydrodeepnet_image .

docker run --privileged -p 5000:5000 \
  -v /data/SWATGenXApp/GenXAppData:/data/SWATGenXApp/GenXAppData \
  -v /data/SWATGenXApp/Users:/data/SWATGenXApp/Users \
  -v /data/SWATGenXApp/codes:/data/SWATGenXApp/codes \
  hydrodeepnet_image
