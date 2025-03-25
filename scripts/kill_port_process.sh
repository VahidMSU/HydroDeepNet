#!/bin/bash

# This script finds and kills processes using specific ports
# Usage: ./kill_port_process.sh [port_numbers]

# For port 3000, exclude VS Code processes
sudo lsof -i:3000 -sTCP:LISTEN | grep -v "code" | awk '{print $2}' | xargs -r sudo kill -9

# Kill all processes on port 5050
sudo kill -9 $(sudo lsof -t -i:5050)