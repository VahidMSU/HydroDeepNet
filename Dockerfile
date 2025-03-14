FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install gosu first for proper user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Add QGIS 3.40 repository and install dependencies
RUN apt-get update && apt-get install -y \
    gnupg software-properties-common wget curl lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.qgis.org/downloads/qgis-archive-keyring.gpg | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" | tee /etc/apt/sources.list.d/qgis.list && \
    apt-get update && \
    apt-get install -y \
    qgis qgis-plugin-grass python3-qgis python3-venv python3-pip nginx \
    build-essential cmake sqlite3 libsqlite3-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev python3-dev swig xvfb gdb \
    redis-server \
    apache2 apache2-dev \
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

RUN chmod +x ./scripts/swatplus_installation.sh && \
    ./scripts/swatplus_installation.sh

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
    pip install --no-cache-dir -r requirements_docker.txt || echo "Some packages failed to install" && \
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

# Create logs directory
RUN mkdir -p /data/SWATGenXApp/codes/web_application/logs && \
    chown www-data:www-data /data/SWATGenXApp/codes/web_application/logs

ENV XDG_RUNTIME_DIR=/tmp/runtime-www-data

# Now switch to www-data user for application runtime
USER www-data

# Set a writable HOME directory for www-data
ENV HOME=/data/SWATGenXApp/Users

# Expose Flask and NGINX ports
EXPOSE 5000 80

ENV FLASK_APP=run.py
ENV FLASK_RUN_PORT=5000
ENV PYTHONPATH="/data/SWATGenXApp/codes"
ENV FLASK_ENV=production

# Start Redis, Flask API, Celery worker & NGINX
CMD ["sh", "-c", "redis-server --daemonize yes && gunicorn -b 0.0.0.0:5000 run:app & cd /data/SWATGenXApp/codes/web_application && celery -A celery_worker worker --loglevel=info --concurrency=4 & nginx -g 'daemon off; pid /tmp/nginx.pid;'"]
