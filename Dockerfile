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

# Install gosu first for proper user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Add QGIS 3.40 repository and install dependencies
# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y \
    gnupg software-properties-common wget curl lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.qgis.org/downloads/qgis-archive-keyring.gpg | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" | tee /etc/apt/sources.list.d/qgis.list \
    && apt-get update \
    && apt-get install -y \
    qgis qgis-plugin-grass python3-qgis python3-venv python3-pip nginx \
    build-essential cmake sqlite3 libsqlite3-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev python3-dev swig xvfb gdb \
    redis-server \
    apache2 apache2-dev \
    unzip \
    --no-install-recommends \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install GDAL 3.8.4 from source with Debug mode
RUN wget https://github.com/OSGeo/gdal/releases/download/v3.8.4/gdal-3.8.4.tar.gz && \
    if [ ! -f gdal-3.8.4.tar.gz ]; then \
        echo "Error: GDAL download failed" && exit 1; \
    fi && \
    tar xzf gdal-3.8.4.tar.gz && \
    cd gdal-3.8.4 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON .. && \
    cmake --build . --parallel $(nproc) && \
    cmake --build . --target install --parallel $(nproc) && \
    cd ../.. && \
    rm -rf gdal-3.8.4*

# Ensure GDAL uses the correct library
RUN ln -sf /usr/local/lib/libgdal.so.34 /usr/local/lib/libgdal.so && \
    ldconfig

# Create Redis configuration
RUN mkdir -p /etc/redis && \
    cat > /etc/redis/redis.conf << EOF
bind 127.0.0.1
protected-mode no
port 6379
dir /var/lib/redis
daemonize yes
loglevel notice
logfile /var/log/redis/redis-server.log
EOF

# Create directories and set up working directory
WORKDIR /data/SWATGenXApp/codes

# Create the virtual environment inside the container
RUN python3 -m venv /data/SWATGenXApp/codes/.venv
ENV VIRTUAL_ENV=/data/SWATGenXApp/codes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy everything to the container
COPY . .

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

# Prepare a modified requirements file
WORKDIR /data/SWATGenXApp/codes
RUN grep -v "mod_wsgi" requirements.txt | grep -v "numpy==" > requirements_docker.txt || cp requirements.txt requirements_docker.txt

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

# Activate the virtual environment and install Python dependencies
RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install numpy with a compatible version first
    pip install --no-cache-dir 'numpy>=1.26.4,<2.0.0' && \
    # Install the most critical dependencies first
    pip install --no-cache-dir gunicorn celery redis flask && \
    # Then try to install most of the requirements
    pip install --no-cache-dir -r requirements_docker.txt && \
    # Make sure essential packages are installed
    pip install --no-cache-dir scipy gdal pandas matplotlib scikit-learn

# Ensure Python GDAL bindings are installed
RUN pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==3.8.4

# Build React frontend
WORKDIR /data/SWATGenXApp/codes/web_application/frontend
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
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Start Redis and wait for it to be ready
log "Starting Redis server..."
redis-server /etc/redis/redis.conf

# Wait for Redis to be ready
log "Waiting for Redis to be ready..."
max_retries=30
count=0
while ! redis-cli ping &>/dev/null; do
    count=$((count + 1))
    if [ $count -ge $max_retries ]; then
        log "Redis failed to start after $max_retries attempts"
        exit 1
    fi
    log "Waiting for Redis... ($count/$max_retries)"
    sleep 1
done
log "Redis is ready!"

# Start Flask application with gunicorn
log "Starting Flask application..."
cd /data/SWATGenXApp/codes
gunicorn -b 0.0.0.0:5000 run:app --daemon --access-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-access.log --error-logfile /data/SWATGenXApp/codes/web_application/logs/gunicorn-error.log

# Start Celery worker
log "Starting Celery worker..."
cd /data/SWATGenXApp/codes/web_application
celery -A celery_worker worker --loglevel=info --concurrency=4 --max-tasks-per-child=$CELERY_MAX_TASKS_PER_CHILD --logfile=/data/SWATGenXApp/codes/web_application/logs/celery/celery-worker.log --detach

# Start NGINX in foreground
log "Starting NGINX..."
nginx -g 'daemon off; pid /tmp/nginx.pid;'
EOF

RUN chmod +x /entrypoint.sh

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
