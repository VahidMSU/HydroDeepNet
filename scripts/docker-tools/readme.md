# SWATGenXApp Docker Tools

This directory contains tools for running SWATGenXApp in Docker containers.

## Requirements

- Docker Engine (version 20.10+)
- At least 24GB RAM recommended
- At least 100GB free disk space

## Quick Start

```bash
# Build and run the container with alternative ports (8080:80, 5051:5050, 5001:5000)
bash run_docker.sh

# Force rebuild of the image
bash run_docker.sh --rebuild

# Remove existing container and run a new one
bash run_docker.sh --force-remove

# Run with specific port mappings
bash run_docker.sh --http-port=8888 --api-port=5555
```

## Port Configuration

By default, the container uses the following ports internally:
- 80: Web interface (Nginx/Apache)
- 5050: Flask-SocketIO API server
- 5000: Flask API (secondary)

To avoid conflicts with the host, we map these to alternative ports by default:
- 8080 → 80
- 5051 → 5050
- 5001 → 5000

## Available Commands

### Building the Docker Image

```bash
# Build the Docker image
docker build -t swatgenx -f /data/SWATGenXApp/codes/Dockerfile /data/SWATGenXApp/codes

# Force a clean build
docker build --no-cache -t swatgenx -f /data/SWATGenXApp/codes/Dockerfile /data/SWATGenXApp/codes
```

### Running the Container

```bash
# Run container with alternative ports (safer default to avoid conflicts)
docker run --name swatgenx_app -p 8080:80 -p 5051:5050 -p 5001:5000 \
  -v swatgenx_users:/data/SWATGenXApp/Users \
  -v swatgenx_redis:/var/lib/redis \
  -v /data/SWATGenXApp/codes/web_application/logs:/data/SWATGenXApp/codes/web_application/logs \
  -d swatgenx

# Only use standard ports if you're certain they are available
docker run --name swatgenx_app -p 80:80 -p 5050:5050 -p 5000:5000 \
  -v swatgenx_users:/data/SWATGenXApp/Users \
  -v swatgenx_redis:/var/lib/redis \
  -v /data/SWATGenXApp/codes/web_application/logs:/data/SWATGenXApp/codes/web_application/logs \
  -d swatgenx

# Run in interactive mode for debugging
docker run --name swatgenx_app -p 8080:80 -p 5051:5050 -p 5001:5000 \
  -v swatgenx_users:/data/SWATGenXApp/Users \
  -v swatgenx_redis:/var/lib/redis \
  -v /data/SWATGenXApp/codes/web_application/logs:/data/SWATGenXApp/codes/web_application/logs \
  -it swatgenx bash
```

### Managing the Container

```bash
# Stop the container
docker stop swatgenx_app

# Start an existing container
docker start swatgenx_app

# Remove the container (must be stopped first)
docker rm swatgenx_app

# Remove the container forcefully (even if running)
docker rm -f swatgenx_app

# View container logs
docker logs swatgenx_app

# Follow container logs in real-time
docker logs -f swatgenx_app

# Access the container shell
docker exec -it swatgenx_app bash
```

### Using Docker Compose

```bash
# Start using Docker Compose (uses ports defined in docker-compose.yml)
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Override port mappings
PORT_HTTP=8888 PORT_FLASK=5555 PORT_SOCKETIO=5252 docker-compose up -d
```

## Troubleshooting

### Port Conflicts

If you see "port is already in use" errors:

```bash
# Check what's using port 80
sudo lsof -i :80

# Check what's using port 5050
sudo lsof -i :5050

# Run with different ports
docker run --name swatgenx_app -p 8888:80 -p 5252:5050 -p 5555:5000 \
  -v swatgenx_users:/data/SWATGenXApp/Users \
  -v swatgenx_redis:/var/lib/redis \
  -v /data/SWATGenXApp/codes/web_application/logs:/data/SWATGenXApp/codes/web_application/logs \
  -d swatgenx
```

### Container Won't Start

```bash
# Check for errors in the logs
docker logs swatgenx_app

# Try interactive mode to debug
bash run_docker.sh --interactive
```

### Resource Issues

```bash
# Check container resource usage
docker stats swatgenx_app
```

## Accessing the Application

- Web interface: http://localhost:8080 (when using alternative ports)
- API: http://localhost:5051 (when using alternative ports)

To use standard ports (only if available on your host):
- Web interface: http://localhost:80
- API: http://localhost:5050

## Verifying Deployment

To verify that all components are working correctly inside the Docker container:

```bash
# Run the verification script inside the container
docker exec swatgenx_app bash /docker-entrypoint.sh verify

# Or with a running container
docker exec -it swatgenx_app bash
# Then inside the container:
bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh
```

The verification script checks:
- Service status (Redis, Nginx, Flask, Celery)
- Port availability
- API functionality
- SWAT+ components
- File system permissions
- Python environment

A verification report will be generated in the container at `/data/SWATGenXApp/web_application/logs/`.
