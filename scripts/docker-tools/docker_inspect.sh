#!/bin/bash
# Script to inspect Docker images and containers for SWATGenXApp

# Define colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Header
echo -e "${BLUE}=== SWATGenXApp Docker Inspection Tool ===${NC}"
echo

# Function to display section headers
section() {
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

# Check Docker images
section "Docker Images"
echo -e "${CYAN}The following Docker images are available:${NC}"
docker images | grep -E 'swatgenx|REPOSITORY' --color=never

# Check Docker containers
section "Docker Containers"
echo -e "${CYAN}The following Docker containers exist:${NC}"
docker ps -a | grep -E 'swatgenx_app|CONTAINER ID' --color=never

# Check running containers
section "Running Containers"
running_containers=$(docker ps -q --filter "name=swatgenx")
if [ -z "$running_containers" ]; then
    echo -e "${CYAN}No running SWATGenXApp containers.${NC}"
else
    echo -e "${CYAN}Running SWATGenXApp containers with port mappings:${NC}"
    docker ps --filter "name=swatgenx" --format "ID: {{.ID}} | Status: {{.Status}} | Ports: {{.Ports}}"
fi

# Volume information
section "Docker Volumes"
echo -e "${CYAN}Volumes used by SWATGenXApp:${NC}"
docker volume ls | grep -E 'swatgenx|DRIVER' --color=never

# Container stats
if [ ! -z "$running_containers" ]; then
    section "Container Statistics"
    echo -e "${CYAN}Current resource usage:${NC}"
    docker stats --no-stream $(docker ps -q --filter "name=swatgenx")
fi

# Help information
section "Management Commands"
echo -e "To stop the container:    ${GREEN}docker stop swatgenx_app${NC}"
echo -e "To start the container:   ${GREEN}docker start swatgenx_app${NC}"
echo -e "To restart the container: ${GREEN}docker restart swatgenx_app${NC}"
echo -e "To view logs:             ${GREEN}docker logs swatgenx_app${NC}"
echo -e "To access shell:          ${GREEN}docker exec -it swatgenx_app bash${NC}"
echo -e "To remove the container:  ${GREEN}docker rm -f swatgenx_app${NC}"
echo -e "To remove the image:      ${GREEN}docker rmi swatgenx${NC}"
echo -e "To rebuild everything:    ${GREEN}bash run_docker.sh --rebuild --force-remove${NC}"
echo

# Explain what's happening
section "Explanation"
echo -e "Your SWATGenXApp setup consists of:"
echo -e "1. A ${GREEN}Docker image${NC} named '${YELLOW}swatgenx${NC}' that contains the application code and dependencies"
echo -e "2. A ${GREEN}Docker container${NC} named '${YELLOW}swatgenx_app${NC}' created from that image"
echo -e "3. ${GREEN}Persistent volumes${NC} for user data and Redis database"
echo
echo -e "When you run ${YELLOW}run_docker.sh${NC} without --force-remove, it detects"
echo -e "the existing container and simply shows its status instead of creating a new one."
echo
echo -e "To create a fresh container, run: ${GREEN}bash run_docker.sh --force-remove${NC}"
