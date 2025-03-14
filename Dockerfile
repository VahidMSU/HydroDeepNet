FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install gosu first for proper user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Add QGIS 3.40 repository and install dependencies
RUN apt-get update && apt-get install -y \
    gnupg software-properties-common wget curl lsb-release unzip sudo \
    redis-server supervisor && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.qgis.org/downloads/qgis-archive-keyring.gpg | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" | tee /etc/apt/sources.list.d/qgis.list && \
    apt-get update && \
    apt-get install -y \
    qgis qgis-plugin-grass python3-qgis python3-venv python3-pip nginx \
    build-essential cmake sqlite3 libsqlite3-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev python3-dev swig xvfb gdb \
    --no-install-recommends && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install GDAL 3.8.4 from source with Debug mode
RUN wget https://github.com/OSGeo/gdal/releases/download/v3.8.4/gdal-3.8.4.tar.gz && \
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

WORKDIR /data/SWATGenXApp/codes

# Create the virtual environment inside the container
RUN python3 -m venv /data/SWATGenXApp/codes/.venv
ENV VIRTUAL_ENV=/data/SWATGenXApp/codes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy everything to the container
COPY . .

# Run SWAT+ installation script with proper environment
# Add sudo without password for root
RUN echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    # Create necessary directories with proper ownership
    mkdir -p /usr/local/share/SWATPlus/Databases && \
    mkdir -p /root/.local/share/SWATPlus/Databases && \
    chmod +x /data/SWATGenXApp/codes/scripts/swatplus_installation.sh && \
    HOME=/root /data/SWATGenXApp/codes/scripts/swatplus_installation.sh && \
    # Verify installation directories
    ls -la /usr/local/share/SWATPlus/Databases/ && \
    ls -la /usr/share/qgis/python/plugins/ && \
    # Make sure permissions are set correctly
    chmod -R a+rw /usr/local/share/SWATPlus/Databases && \
    chmod -R a+rX /usr/share/qgis/python/plugins/QSWATPlusLinux3_64

# Activate the virtual environment and install Python dependencies
WORKDIR /data/SWATGenXApp/codes
RUN . $VIRTUAL_ENV/bin/activate && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scipy celery redis gunicorn

# Ensure Python GDAL bindings are installed
RUN pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==3.8.4

# Build React frontend
WORKDIR /data/SWATGenXApp/codes/web_application/frontend
RUN npm install && npm run build

# NGINX setup for serving React build and static files
WORKDIR /data/SWATGenXApp/codes/web_application
COPY ./docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Configure Redis for Celery
RUN sed -i 's/bind 127.0.0.1/bind 0.0.0.0/g' /etc/redis/redis.conf && \
    sed -i 's/# requirepass foobared/requirepass redispassword/g' /etc/redis/redis.conf && \
    mkdir -p /var/log/celery /var/run/celery && \
    chown -R www-data:www-data /var/log/celery /var/run/celery

# Create Supervisor configuration
COPY ./docker/supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

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

ENV XDG_RUNTIME_DIR=/tmp/runtime-www-data

# Prepare start script for services
COPY ./docker/start-services.sh /start-services.sh
RUN chmod +x /start-services.sh && \
    chown root:root /start-services.sh

# Now switch to www-data user for application runtime
USER www-data

# Set a writable HOME directory for www-data
ENV HOME=/data/SWATGenXApp/Users

# Expose ports: Flask, NGINX, and Redis
EXPOSE 5000 80 6379

ENV FLASK_APP=run.py
ENV FLASK_RUN_PORT=5000
ENV PYTHONPATH="/data/SWATGenXApp/codes"
ENV CELERY_BROKER_URL="redis://:redispassword@localhost:6379/0"
ENV CELERY_RESULT_BACKEND="redis://:redispassword@localhost:6379/0"

# Switch back to root for supervisord start (it will manage services with proper users)
USER root

# Use supervisord to manage all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
