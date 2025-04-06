
container_id=$(docker ps -q -f ancestor=hydrodeepnet_image)
docker cp /data/SWATGenXApp/codes/SWATGenX $container_id:/data/SWATGenXApp/codes/
docker cp /data/SWATGenXApp/codes/scripts $container_id:/data/SWATGenXApp/codes/

docker exec -it $container_id /bin/bash -c "python /data/SWATGenXApp/codes/SWATGenX/runSWATGenX.py"
