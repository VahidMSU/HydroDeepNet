#!/bin/bash

# This script finds and kills processes using specific ports
# Usage: ./kill_port_process.sh [port_numbers]

#sudo kill -9 $(sudo lsof -t -i:3000)
sudo kill -9 $(sudo lsof -t -i:5050)
