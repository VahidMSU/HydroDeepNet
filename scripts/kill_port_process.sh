#!/bin/bash

# This script finds and kills processes using specific ports
# Usage: ./kill_port_process.sh [port_numbers]


# Kill all processes on port 5050
#kill -9 $(sudo lsof -t -i:3000)
# Kill all processes on port 5050
sudo kill -9 $(sudo lsof -t -i:5050)