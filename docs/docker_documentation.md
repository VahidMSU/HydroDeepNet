# Docker Configuration Documentation

This document provides an overview of the Docker configuration for the HydroDeepNet application, including details about the services, installation processes, and best practices implemented.

## Overview

The HydroDeepNet application is containerized using Docker to provide a consistent and reproducible environment for development and deployment. The Docker configuration includes:

- Ubuntu 22.04 base image
- QGIS 3.40 with related packages
- GDAL 3.8.4 compiled from source with debug mode
- SWAT+ 2.3.1 and SWAT+ Editor 3.0.8
- Redis for message brokering
- Celery for task processing
- NGINX for serving the frontend
- Apache for backend services

## Services

### Redis

Redis is used as a message broker for Celery and for caching. The configuration includes:

- Binding to localhost (127.0.0.1)
- Protected mode disabled for container environment
- Persistence enabled with volume mapping
- Health checks to ensure service availability

### Celery

Celery is used for asynchronous task processing, particularly for model creation and data processing tasks. The configuration includes:

- Worker prefetch multiplier set to 8 for better throughput
- Task time limits configured for long-running tasks
- Connection retry settings for reliability
- Memory limits per child process to prevent memory leaks

### Flask Application

The Flask application serves as the backend API. It is run using Gunicorn with:

- Binding to port 5000
- Access and error logging
- Multiple worker processes for handling concurrent requests

### NGINX

NGINX serves the React frontend and acts as a reverse proxy for the Flask API. The configuration includes:

- Serving static files from the React build
- Proxying API requests to the Flask application
- Proper caching headers for static assets

## Installation Processes

### SWAT+ Installation

The SWAT+ installation process includes:

1. Downloading the SWAT+ installer (version 2.3.1)
2. Extracting and running the installer
3. Setting appropriate permissions for SWAT+ components
4. Installing SWAT+ Editor (version 3.0.8)
5. Downloading and configuring required database files
6. Setting up proper directory structures and permissions

### Python Environment

The Python environment is set up with:

1. Creating a virtual environment
2. Installing dependencies from requirements.txt
3. Special handling for numpy version compatibility
4. Installing GDAL Python bindings to match the compiled version

## Best Practices Implemented

The Docker configuration follows these best practices:

1. **Layer Optimization**: Related RUN commands are combined to reduce the number of layers and image size.

2. **Error Handling**: Robust error checking for all installation steps, with clear error messages and exit codes.

3. **Service Management**: Proper service startup sequence with verification and health checks.

4. **Security**: Services run as non-root users with appropriate permissions.

5. **Configuration Management**: Proper configuration files for all services with environment variable support.

6. **Resource Management**: Memory and CPU limits for services to prevent resource exhaustion.

7. **Logging**: Comprehensive logging for all services with proper log rotation.

## Environment Variables

The following environment variables can be used to customize the Docker environment:

- `FLASK_ENV`: Set to "production" or "development"
- `FLASK_RUN_PORT`: Port for the Flask application (default: 5000)
- `CELERY_WORKER_PREFETCH_MULTIPLIER`: Number of tasks each worker can prefetch (default: 8)
- `CELERY_TASK_SOFT_TIME_LIMIT`: Soft time limit for tasks in seconds (default: 43200)
- `CELERY_TASK_TIME_LIMIT`: Hard time limit for tasks in seconds (default: 86400)
- `CELERY_MAX_TASKS_PER_CHILD`: Maximum number of tasks a worker can process before restarting (default: 100)
- `CELERY_MAX_MEMORY_PER_CHILD_MB`: Maximum memory usage per worker in MB (default: 8192)

## Volumes

The Docker configuration uses the following volumes:

- `/var/lib/redis`: Redis data persistence
- `/data/SWATGenXApp/GenXAppData`: Application data storage

## Ports

The Docker container exposes the following ports:

- `5000`: Flask API
- `80`: NGINX (frontend)

## Maintenance

### Logs

Logs are stored in the following locations:

- Redis: `/var/log/redis/redis-server.log`
- Celery: `/data/SWATGenXApp/codes/web_application/logs/celery/celery-worker.log`
- Flask: `/data/SWATGenXApp/codes/web_application/logs/gunicorn-access.log` and `/data/SWATGenXApp/codes/web_application/logs/gunicorn-error.log`
- NGINX: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`

### Health Checks

The Docker container includes health checks for:

- Redis: `redis-cli ping`
- Flask API: `curl -f http://localhost:5000/`

These health checks ensure that the services are running properly and can be used by container orchestration systems to monitor the container health.
