#!/bin/bash

# This script finds and kills processes using specific ports
# Usage: ./kill_port_process.sh [port_numbers]

PORTS=("${@:-5050 3000}")  # Default to ports 5050 and 3000 if not specified
LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/port_killer.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

for PORT in "${PORTS[@]}"; do
  log "Looking for processes using port $PORT..."

  # Check if required commands exist
  for cmd in lsof netstat fuser; do
    if ! command -v $cmd &> /dev/null; then
      log "Warning: Command not found: $cmd. Some process detection methods may not work."
    fi
  done

  # Try multiple ways to find the processes
  LSOF_PROCESSES=$(lsof -i :$PORT 2>/dev/null | grep -v PID | awk '{print $2}' || echo "")
  NETSTAT_PROCESSES=$(netstat -tunlp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 || echo "")
  FUSER_PROCESSES=$(fuser $PORT/tcp 2>/dev/null || echo "")

  # Combine the process lists
  ALL_PROCESSES="$LSOF_PROCESSES $NETSTAT_PROCESSES $FUSER_PROCESSES"

  # Remove duplicates
  UNIQUE_PROCESSES=$(echo "$ALL_PROCESSES" | tr ' ' '\n' | sort -u | grep -v "^$")

  if [ -z "$UNIQUE_PROCESSES" ]; then
    log "No processes found using port $PORT."
    continue
  fi

  log "Found the following processes using port $PORT: $UNIQUE_PROCESSES"

  # First, try to terminate gracefully
  for PID in $UNIQUE_PROCESSES; do
    log "Attempting to terminate process $PID gracefully..."

    # Get process name for better logging
    PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown process")
    USER_NAME=$(ps -p $PID -o user= 2>/dev/null || echo "unknown user")

    log "Process details: PID=$PID, Name=$PROCESS_NAME, User=$USER_NAME"

    if kill -15 $PID 2>/dev/null; then
      log "Sent SIGTERM to process $PID ($PROCESS_NAME)"
      sleep 2
    else
      log "Failed to send SIGTERM to process $PID. Attempting with sudo."
      sudo kill -15 $PID 2>/dev/null
      sleep 2
    fi
  done

  # Check if any processes are still using the port
  if lsof -i :$PORT > /dev/null 2>&1 || fuser $PORT/tcp > /dev/null 2>&1; then
    log "Some processes are still using port $PORT. Using force kill."

    for PID in $UNIQUE_PROCESSES; do
      log "Force killing process $PID..."
      if kill -9 $PID 2>/dev/null; then
        log "Sent SIGKILL to process $PID"
      else
        log "Attempting SIGKILL with sudo on process $PID"
        sudo kill -9 $PID 2>/dev/null
      fi
    done

    # Additional measures - using fuser directly with kill
    log "Using fuser to kill processes on port $PORT..."
    sudo fuser -k $PORT/tcp 2>/dev/null
    sleep 3
  else
    log "All processes on port $PORT have been terminated gracefully"
  fi

  # Final check
  if lsof -i :$PORT > /dev/null 2>&1 || fuser $PORT/tcp > /dev/null 2>&1; then
    log "WARNING: Port $PORT is still in use after kill attempts"

    # Last resort - check for zombie processes and restart networking
    log "Checking for zombie processes..."
    ZOMBIES=$(ps aux | grep defunct | grep -v grep || echo "")
    if [ -n "$ZOMBIES" ]; then
      log "Found zombie processes. Attempting to clean up parent processes."
      echo "$ZOMBIES" >> "$LOG_FILE"
    fi

    log "Restarting networking as a last resort..."
    sudo systemctl restart networking 2>/dev/null || sudo service networking restart 2>/dev/null

    exit 1
  else
    log "Port $PORT is now free"
  fi
done

exit 0
