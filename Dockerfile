FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/data/SWATGenXApp/codes" \
    FLASK_APP=run.py \
    FLASK_RUN_PORT=5000 \
    FLASK_ENV=production \
    XDG_RUNTIME_DIR=/tmp/runtime-www-data \
    HOME=/data/SWATGenXApp/Users

# Install essential services first (Redis, NGINX, Apache)
RUN apt-get update && apt-get install -y \
    redis-server \
    nginx \
    apache2 apache2-dev \
    python3-venv python3-pip \
    nodejs npm \
    sudo \
    gnupg software-properties-common wget curl lsb-release \
    build-essential cmake sqlite3 libsqlite3-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev python3-dev swig xvfb gdb \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create Redis configuration with proper permissions
RUN mkdir -p /etc/redis && \
    mkdir -p /var/log/redis && \
    mkdir -p /var/lib/redis && \
    echo "bind 127.0.0.1" > /etc/redis/redis.conf && \
    echo "protected-mode no" >> /etc/redis/redis.conf && \
    echo "port 6379" >> /etc/redis/redis.conf && \
    echo "dir /var/lib/redis" >> /etc/redis/redis.conf && \
    echo "daemonize yes" >> /etc/redis/redis.conf && \
    echo "loglevel notice" >> /etc/redis/redis.conf && \
    echo "logfile /tmp/redis-server.log" >> /etc/redis/redis.conf && \
    chmod 777 /etc/redis && \
    chmod 644 /etc/redis/redis.conf && \
    chown -R redis:redis /etc/redis

# Create NGINX directories with proper permissions
RUN mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi \
    /var/lib/nginx/proxy /var/lib/nginx/scgi \
    /var/lib/nginx/uwsgi /var/cache/nginx /run/nginx && \
    chown -R www-data:www-data /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx && \
    chmod -R 755 /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx

# Create runtime directory for QGIS
RUN mkdir -p /tmp/runtime-www-data && \
    chown www-data:www-data /tmp/runtime-www-data && \
    chmod 700 /tmp/runtime-www-data

# Create user runtime directory with proper permissions
RUN mkdir -p /run/user/33 && \
    chown www-data:www-data /run/user/33 && \
    chmod 700 /run/user/33

# Create logs directory with proper permissions for all services
RUN mkdir -p /data/SWATGenXApp/codes/web_application/logs/celery && \
    mkdir -p /var/log/redis && \
    chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs && \
    chown -R redis:redis /var/log/redis

# Create directories and set up working directory
WORKDIR /data/SWATGenXApp/codes

# Create the virtual environment inside the container
RUN python3 -m venv /data/SWATGenXApp/codes/.venv
ENV VIRTUAL_ENV=/data/SWATGenXApp/codes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Add Celery configuration environment variables
ENV CELERY_WORKER_PREFETCH_MULTIPLIER=8 \
    CELERY_TASK_SOFT_TIME_LIMIT=43200 \
    CELERY_TASK_TIME_LIMIT=86400 \
    CELERY_DISABLE_RATE_LIMITS=true \
    CELERY_BROKER_CONNECTION_RETRY=true \
    CELERY_BROKER_CONNECTION_MAX_RETRIES=20 \
    CELERY_REDIS_MAX_CONNECTIONS=500 \
    CELERY_MAX_TASKS_PER_CHILD=100 \
    CELERY_MAX_MEMORY_PER_CHILD_MB=8192

# Copy everything to the container
COPY . .

# Prepare a modified requirements file
WORKDIR /data/SWATGenXApp/codes
RUN grep -v "mod_wsgi" requirements.txt | grep -v "numpy==" | grep -v "GDAL==" > requirements_docker.txt || cp requirements.txt requirements_docker.txt

# Activate the virtual environment and install Python dependencies
RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install numpy with a compatible version first
    pip install --no-cache-dir 'numpy>=1.26.4,<2.0.0' && \
    # Install the most critical dependencies first
    pip install --no-cache-dir gunicorn celery redis flask

# Install GDAL from apt repository
RUN apt-get update && \
    apt-get install -y libgdal-dev gdal-bin python3-gdal && \
    export CPLUS_INCLUDE_PATH=/usr/include/gdal && \
    export C_INCLUDE_PATH=/usr/include/gdal && \
    pip install --no-cache-dir GDAL==$(gdal-config --version)

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Install QGIS after GDAL is properly installed
RUN apt-get update && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.qgis.org/downloads/qgis-archive-keyring.gpg | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" | tee /etc/apt/sources.list.d/qgis.list && \
    apt-get update && \
    apt-get install -y qgis qgis-plugin-grass python3-qgis --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# SWAT+ Installation - with improved error handling
RUN mkdir -p /data/SWATGenXApp/codes/swatplus_installation && \
    cd /data/SWATGenXApp/codes && \
    echo "Downloading SWAT+ installer..." && \
    wget https://plus.swat.tamu.edu/downloads/2.3/2.3.1/swatplus-linux-installer-2.3.1.tgz -P /data/SWATGenXApp/codes/swatplus_installation/ && \
    if [ ! -f /data/SWATGenXApp/codes/swatplus_installation/swatplus-linux-installer-2.3.1.tgz ]; then \
        echo "Error: SWAT+ installer download failed" && exit 1; \
    fi && \
    tar -xvf /data/SWATGenXApp/codes/swatplus_installation/swatplus-linux-installer-2.3.1.tgz -C /data/SWATGenXApp/codes/swatplus_installation/ && \
    cd /data/SWATGenXApp/codes/swatplus_installation && \
    chmod +x installforall.sh && \
    echo "Running SWAT+ installer..." && \
    ./installforall.sh && \
    echo "Setting permissions..." && \
    chown -R www-data:www-data /usr/local/share/SWATPlus && \
    chown -R www-data:www-data /usr/share/qgis/python/plugins/QSWATPlusLinux3_64 && \
    # Download SWAT+ Editor
    echo "Downloading SWAT+ Editor..." && \
    wget https://github.com/swat-model/swatplus-editor/archive/refs/tags/v3.0.8.tar.gz -P /data/SWATGenXApp/codes/swatplus_installation/ && \
    if [ ! -f /data/SWATGenXApp/codes/swatplus_installation/v3.0.8.tar.gz ]; then \
        echo "Error: SWAT+ Editor download failed" && exit 1; \
    fi && \
    tar -xvf /data/SWATGenXApp/codes/swatplus_installation/v3.0.8.tar.gz -C /data/SWATGenXApp/codes/swatplus_installation/ && \
    # Create the target directory for SWATPlusEditor if it doesn't exist
    mkdir -p /usr/local/share/SWATPlusEditor && \
    # Move the SWATPlusEditor to the target directory
    mv /data/SWATGenXApp/codes/swatplus_installation/swatplus-editor-3.0.8 /usr/local/share/SWATPlusEditor/swatplus-editor && \
    # Download additional required files
    echo "Downloading additional required files..." && \
    wget https://plus.swat.tamu.edu/downloads/3.0/3.0.0/swatplus_datasets.sqlite -P /data/SWATGenXApp/codes/swatplus_installation/ && \
    if [ ! -f /data/SWATGenXApp/codes/swatplus_installation/swatplus_datasets.sqlite ]; then \
        echo "Error: swatplus_datasets.sqlite download failed" && exit 1; \
    fi && \
    wget https://plus.swat.tamu.edu/downloads/swatplus_wgn.zip -P /data/SWATGenXApp/codes/swatplus_installation/ && \
    if [ ! -f /data/SWATGenXApp/codes/swatplus_installation/swatplus_wgn.zip ]; then \
        echo "Error: swatplus_wgn.zip download failed" && exit 1; \
    fi && \
    wget https://plus.swat.tamu.edu/downloads/swatplus_soils.zip -P /data/SWATGenXApp/codes/swatplus_installation/ && \
    if [ ! -f /data/SWATGenXApp/codes/swatplus_installation/swatplus_soils.zip ]; then \
        echo "Error: swatplus_soils.zip download failed" && exit 1; \
    fi && \
    # Extract the downloaded zip files
    echo "Extracting database files..." && \
    unzip /data/SWATGenXApp/codes/swatplus_installation/swatplus_wgn.zip -d /data/SWATGenXApp/codes/swatplus_installation/ && \
    unzip /data/SWATGenXApp/codes/swatplus_installation/swatplus_soils.zip -d /data/SWATGenXApp/codes/swatplus_installation/ && \
    # Create directories if they don't exist
    echo "Setting up database directories..." && \
    mkdir -p /usr/local/share/SWATPlus/Databases && \
    mkdir -p ${HOME}/.local/share/SWATPlus/Databases && \
    # Copy database files to the target directory
    echo "Copying database files to system location..." && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_datasets.sqlite /usr/local/share/SWATPlus/Databases/ && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_soils.sqlite /usr/local/share/SWATPlus/Databases/ && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_wgn.sqlite /usr/local/share/SWATPlus/Databases/ && \
    # Copy for internal testing
    echo "Copying database files for user testing..." && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_datasets.sqlite ${HOME}/.local/share/SWATPlus/Databases/ && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_soils.sqlite ${HOME}/.local/share/SWATPlus/Databases/ && \
    cp /data/SWATGenXApp/codes/swatplus_installation/swatplus_wgn.sqlite ${HOME}/.local/share/SWATPlus/Databases/ && \
    # Set permissions for QSWATPlus files
    echo "Setting permissions for QSWATPlus files..." && \
    chmod -R 755 /usr/share/qgis/python/plugins/QSWATPlusLinux3_64 && \
    # Clean up
    echo "SWAT+ installation completed successfully!" && \
    rm -rf /data/SWATGenXApp/codes/swatplus_installation

# Build React frontend with increased memory limit
WORKDIR /data/SWATGenXApp/codes/web_application/frontend
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm install && npm run build

# NGINX setup for serving React build and static files
WORKDIR /data/SWATGenXApp/codes/web_application
COPY ./docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Set up user
RUN usermod -u 33 www-data && \
    groupmod -g 33 www-data && \
    mkdir -p /data/SWATGenXApp/Users && \
    mkdir -p /data/SWATGenXApp/GenXAppData && \
    chown -R www-data:www-data /data/SWATGenXApp && \
    chmod -R 755 /data/SWATGenXApp


# Create NGINX directories with proper permissions
RUN mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi \
    /var/lib/nginx/proxy /var/lib/nginx/scgi \
    /var/lib/nginx/uwsgi /var/cache/nginx /run/nginx && \
    chown -R www-data:www-data /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx && \
    chmod -R 755 /var/lib/nginx /var/cache/nginx /var/log/nginx /run/nginx

# Create runtime directory for QGIS
RUN mkdir -p /tmp/runtime-www-data && \
    chown www-data:www-data /tmp/runtime-www-data && \
    chmod 700 /tmp/runtime-www-data

# Create user runtime directory with proper permissions
RUN mkdir -p /run/user/33 && \
    chown www-data:www-data /run/user/33 && \
    chmod 700 /run/user/33

# Create logs directory with proper permissions for all services
RUN mkdir -p /data/SWATGenXApp/codes/web_application/logs/celery && \
    mkdir -p /var/log/redis && \
    chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs && \
    chown -R redis:redis /var/log/redis

# Create volume for Redis data persistence
VOLUME ["/var/lib/redis"]

# Create entrypoint script for proper service management
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Function to log messages' >> /entrypoint.sh && \
    echo 'log() {' >> /entrypoint.sh && \
    echo '    echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] $1"' >> /entrypoint.sh && \
    echo '}' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start Redis and wait for it to be ready' >> /entrypoint.sh && \
    echo 'log "Starting Redis server..."' >> /entrypoint.sh && \
    echo 'redis-server /etc/redis/redis.conf' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Wait for Redis to be ready' >> /entrypoint.sh && \
    echo 'log "Waiting for Redis to be ready..."' >> /entrypoint.sh && \
    echo 'max_retries=30' >> /entrypoint.sh && \
    echo 'count=0' >> /entrypoint.sh && \
    echo 'while ! redis-cli ping &>/dev/null; do' >> /entrypoint.sh && \
    echo '    count=$((count + 1))' >> /entrypoint.sh && \
    echo '    if [ $count -ge $max_retries ]; then' >> /entrypoint.sh && \
    echo '        log "Redis failed to start after $max_retries attempts"' >> /entrypoint.sh && \
    echo '        exit 1' >> /entrypoint.sh && \
    echo '    fi' >> /entrypoint.sh && \
    echo '    log "Waiting for Redis... ($count/$max_retries)"' >> /entrypoint.sh && \
    echo '    sleep 1' >> /entrypoint.sh && \
    echo 'done' >> /entrypoint.sh && \
    echo 'log "Redis is ready!"' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start Flask application with gunicorn' >> /entrypoint.sh && \
    echo 'log "Starting Flask application..."' >> /entrypoint.sh && \
    echo 'cd /data/SWATGenXApp/codes' >> /entrypoint.sh && \
    echo 'gunicorn -b 0.0.0.0:5000 run:app --daemon --access-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-access.log --error-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-error.log' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start Celery worker' >> /entrypoint.sh && \
    echo 'log "Starting Celery worker..."' >> /entrypoint.sh && \
    echo 'cd /data/SWATGenXApp/codes/web_application' >> /entrypoint.sh && \
    echo 'celery -A celery_worker worker --loglevel=info --concurrency=4 --max-tasks-per-child=$CELERY_MAX_TASKS_PER_CHILD --logfile=/data/SWATGenXApp/codes/web_application/logs/celery/celery-worker.log --detach' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start NGINX in foreground' >> /entrypoint.sh && \
    echo 'log "Starting NGINX..."' >> /entrypoint.sh && \
    echo 'nginx -g '"'"'daemon off; pid /tmp/nginx.pid;'"'"'' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD redis-cli ping && \
      curl -f http://localhost:5000/ || exit 1

# Now switch to www-data user for application runtime
USER www-data

# Expose Flask and NGINX ports
EXPOSE 5000 80

# Use entrypoint script to start all services
ENTRYPOINT ["/entrypoint.sh"]
