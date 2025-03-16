#!/bin/bash

set -e  # Exit on error

### kill all processes on ports: 3000, 5050, 5000, 80
echo "Killing all processes on ports: 3000, 5050, 5000, 80"
### first kill all docker containers
if command -v docker &>/dev/null; then
  CONTAINERS=$(docker ps -q)
  if [ -n "$CONTAINERS" ]; then
    echo "Stopping docker containers..."
    docker kill $CONTAINERS || {
      echo "Warning: Failed to kill some docker containers"
      echo "Attempting to stop them gracefully..."
      docker stop $CONTAINERS || echo "Warning: Some containers could not be stopped"
    }
  else
    echo "No running docker containers found."
  fi
else
  echo "Docker command not found. Skipping container cleanup."
fi

# Make sure required tools are available
for cmd in lsof kill; do
  if ! command -v $cmd &>/dev/null; then
    echo "Warning: Required command not found: $cmd"
    echo "Port killing may not work correctly"
  fi
done

## Check if ports are listening and kill them gracefully
for PORT in 3000 5050 5000 80; do
  echo "Checking port $PORT..."
  PIDS=$(lsof -ti:$PORT 2>/dev/null || echo "")
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
