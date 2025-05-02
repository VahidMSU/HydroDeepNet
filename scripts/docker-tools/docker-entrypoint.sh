#!/bin/bash
set -e

# Create necessary directories if they don't exist
mkdir -p /data/SWATGenXApp/Users
mkdir -p /data/SWATGenXApp/codes/web_application/logs/celery
mkdir -p /var/log/redis
mkdir -p /data/SWATGenXApp/bin
mkdir -p /usr/local/share/SWATPlus/Databases
mkdir -p /data/SWATGenXApp/data/SWATPlus/Databases

# Ensure proper ownership
chown -R www-data:www-data /data/SWATGenXApp/codes/web_application/logs
chown -R www-data:www-data /data/SWATGenXApp/Users
chown -R redis:redis /var/lib/redis /var/log/redis

# Generate runtime directories for QGIS
mkdir -p /tmp/runtime-www-data
chown www-data:www-data /tmp/runtime-www-data
chmod 700 /tmp/runtime-www-data

# Initialize Redis data directory
if [ ! -d /var/lib/redis/data ]; then
    mkdir -p /var/lib/redis/data
    chown redis:redis /var/lib/redis/data
fi

# Create placeholder executables if they don't exist
if [ ! -f "/data/SWATGenXApp/bin/swatplus" ]; then
    echo '#!/bin/bash' >"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "SWATPlus Simulation Tool"' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "Version: 2.3.1"' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'if [ "$1" = "--version" ]; then' >>"/data/SWATGenXApp/bin/swatplus"
    echo '  exit 0' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'fi' >>"/data/SWATGenXApp/bin/swatplus"
    echo 'echo "This is a placeholder executable."' >>"/data/SWATGenXApp/bin/swatplus"
    chmod +x "/data/SWATGenXApp/bin/swatplus"
fi

if [ ! -f "/data/SWATGenXApp/bin/modflow-nwt" ]; then
    echo '#!/bin/bash' >"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "MODFLOW-NWT Simulation Tool"' >>"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "Version: 1.3.0"' >>"/data/SWATGenXApp/bin/modflow-nwt"
    echo 'echo "This is a placeholder executable."' >>"/data/SWATGenXApp/bin/modflow-nwt"
    chmod +x "/data/SWATGenXApp/bin/modflow-nwt"
fi

# Create sample database files if missing
DB_PATH="/usr/local/share/SWATPlus/Databases"
if [ ! -f "$DB_PATH/swatplus_datasets.sqlite" ]; then
    touch "$DB_PATH/swatplus_datasets.sqlite"
    touch "$DB_PATH/swatplus_soils.sqlite"
    touch "$DB_PATH/swatplus_wgn.sqlite"
    chown -R www-data:www-data "$DB_PATH"
fi

# Install GDAL Python bindings if not present
if ! python3 -c "import osgeo.gdal" 2>/dev/null; then
    pip install --no-cache-dir gdal
fi

# Check if a command was provided
if [ "$1" = "supervisord" ]; then
    echo "Starting all services via supervisor..."
    exec "$@"
elif [ "$1" = "flask" ]; then
    echo "Starting Flask application only..."
    exec gosu www-data gunicorn -w 4 -b 0.0.0.0:5000 run:app
elif [ "$1" = "celery" ]; then
    echo "Starting Celery worker only..."
    cd /data/SWATGenXApp/codes/web_application
    exec gosu www-data celery -A app.celery worker --loglevel=info
elif [ "$1" = "redis" ]; then
    echo "Starting Redis server only..."
    exec redis-server /etc/redis/redis.conf
elif [ "$1" = "nginx" ]; then
    echo "Starting Nginx only..."
    exec nginx -g "daemon off;"
elif [ "$1" = "verify" ]; then
    echo "Running deployment verification..."
    bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh
elif [ "$1" = "fix" ]; then
    echo "Running deployment fixes..."
    bash /data/SWATGenXApp/codes/scripts/docker-tools/fix_docker_deployment.sh
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
else
    echo "Starting interactive shell..."
    exec bash
fi
