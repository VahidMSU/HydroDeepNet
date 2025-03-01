### kill all processes on ports: 3000, 5050, 5000, 80
echo "Killing all processes on ports: 3000, 5050, 5000, 80"
### first kill all docker containers
CONTAINERS=$(docker ps -q)
if [ -n "$CONTAINERS" ]; then
  echo "Stopping docker containers..."
  docker kill $CONTAINERS
else
  echo "No running docker containers found."
fi

## Check if ports are listening and kill them gracefully
for PORT in 3000 5050 5000 80; do
  echo "Checking port $PORT..."
  PIDS=$(lsof -ti:$PORT)
  if [ -n "$PIDS" ]; then
    echo "Killing processes on port $PORT: $PIDS"
    kill $PIDS 2>/dev/null || echo "Could not kill processes gracefully, using force..."
    sleep 1
    kill -9 $PIDS 2>/dev/null || echo "No processes left to force kill on port $PORT"
  else
    echo "No processes found on port $PORT"
  fi
done

echo "Port reset completed."
