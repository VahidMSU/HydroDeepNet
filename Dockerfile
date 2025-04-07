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
RUN apt-get update && \
    apt-get install -y software-properties-common wget curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    add-apt-repository ppa:ubuntugis/ppa -y && \
    apt-get update && \
    apt-get install -y python3 python3-venv libgdal-dev wget libmpich-dev gosu && \
    rm -rf /var/lib/apt/lists/*

# Ensure GDAL uses the correct library
RUN ln -sf /usr/local/lib/libgdal.so.34 /usr/local/lib/libgdal.so && \
    ldconfig

# Create directories and set up working directory
WORKDIR /data/SWATGenXApp/codes

# Create the virtual environment inside the container
RUN python3 -m venv /data/SWATGenXApp/codes/.venv
ENV VIRTUAL_ENV=/data/SWATGenXApp/codes/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy everything to the container
COPY . .
# Ensure GDAL install script is executable
RUN chmod +x ./scripts/dependencies/install_gdal.sh
# Add GDAL build step to install GDAL 3.8.4 from source
RUN ./scripts/dependencies/install_gdal.sh

# Prepare a modified requirements file
WORKDIR /data/SWATGenXApp/codes
RUN grep -v "mod_wsgi" requirements.txt | grep -v "numpy==" > requirements_docker.txt || cp requirements.txt requirements_docker.txt

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

WORKDIR /data/SWATGenXApp/codes/scripts

### install Redis: /data/SWATGenXApp/codes/scripts/dependencies/install_redis.sh
RUN chmod +x ./dependencies/install_redis.sh
RUN ./dependencies/install_redis.sh

RUN chmod +x ./dependencies/install_celery.sh
### install celery: /data/SWATGenXApp/codes/scripts/dependencies/install_celery.sh
USER www-data
RUN ./dependencies/install_celery.sh
USER root

### INSTALL QGIS: /data/SWATGenXApp/codes/scripts/dependencies/install_qgis.sh
RUN chmod +x ./dependencies/install_qgis.sh
RUN ./dependencies/install_qgis.sh

##  INSTALL swatplus: /data/SWATGenXApp/codes/scripts/dependencies/install_swatplus.sh
RUN chmod +x ./dependencies/install_swatplus.sh
RUN ./dependencies/install_swatplus.sh

# Install and configure Nginx using custom script
RUN chmod +x ./dependencies/install_nginx.sh && \
    ./dependencies/install_nginx.sh

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

# Copy the entrypoint script
COPY ./scripts/docker-tools/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create volume for Redis data persistence
VOLUME ["/var/lib/redis"]

# Expose Flask and NGINX ports
EXPOSE 5000 80

# Use entrypoint script to start all services
ENTRYPOINT ["/entrypoint.sh"]
