#!/bin/bash
# restart_services.sh
# Complete service restart script for SWATGenX application
# This will restart Redis, Celery workers, and the Flask application

# Use set -e only for initialization, will disable it for service control
set -e

# Define colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color


source "/data/SWATGenXApp/codes/scripts/global_path.sh"


# Create a config file to store the web server preference
WEBSERVER_PREF_FILE="${CONFIG_DIR}/webserver_preference"

echo "Base directory: $BASE_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Bin directory: $BIN_DIR"
echo "Log directory: $LOG_DIR"
echo "Config directory: $CONFIG_DIR"
echo "Systemd directory: $SYSTEMD_DIR"
echo "Apache directory: $APACHE_DIR"


# Define log file
LOG_FILE="${LOG_DIR}/restart_services.log"
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
   exit 1
fi

# Web server selection with persistence
echo -e "\n${YELLOW}=====================================================${NC}"
echo -e "${GREEN}=== SWATGenX Production Web Server Selection ===${NC}"
echo -e "${YELLOW}=====================================================${NC}"
echo -e "Please select your web server for production:"
echo -e "  ${GREEN}0) None${NC} - Do not start any web server"
echo -e "  ${GREEN}1) Apache${NC} - Traditional and widely used"
echo -e "  ${GREEN}2) Nginx${NC} - Lightweight and high performance"

# Check if there's a saved preference
SAVED_PREFERENCE=""
if [ -f "$WEBSERVER_PREF_FILE" ]; then
    SAVED_PREFERENCE=$(cat "$WEBSERVER_PREF_FILE")
    if [ "$SAVED_PREFERENCE" = "apache" ]; then
        echo -e "${YELLOW}Previously selected: Apache${NC}"
    elif [ "$SAVED_PREFERENCE" = "nginx" ]; then
        echo -e "${YELLOW}Previously selected: Nginx${NC}"
    elif [ "$SAVED_PREFERENCE" = "none" ]; then
        echo -e "${YELLOW}Previously selected: None${NC}"
    fi
fi

echo -e "${YELLOW}Automatically using previous selection or Apache in 15 seconds...${NC}"

# Read user input with a 15 second timeout
read -t 15 -p "Enter your choice [0/1/2]: " WEB_SERVER_CHOICE

# Default to the saved preference or Apache if no input
if [ -z "$WEB_SERVER_CHOICE" ]; then
    if [ ! -z "$SAVED_PREFERENCE" ]; then
        WEB_SERVER="$SAVED_PREFERENCE"
        log "Using saved web server preference: $WEB_SERVER"
    else
        WEB_SERVER="apache"
        log "No input provided. Defaulting to Apache"
    fi
elif [ "$WEB_SERVER_CHOICE" = "0" ]; then
    WEB_SERVER="none"
    log "Selected no web server"
elif [ "$WEB_SERVER_CHOICE" = "1" ]; then
    WEB_SERVER="apache"
    log "Selected web server: Apache"
elif [ "$WEB_SERVER_CHOICE" = "2" ]; then
    WEB_SERVER="nginx"
    log "Selected web server: Nginx"
else
    warning "Invalid selection: '$WEB_SERVER_CHOICE'. Must be 0, 1, or 2. Defaulting to Apache."
    WEB_SERVER="apache"
fi

# Save the preference for future runs
echo "$WEB_SERVER" > "$WEBSERVER_PREF_FILE"
log "Saved web server preference: $WEB_SERVER"

echo -e "\n${GREEN}Selected web server for production: ${YELLOW}$(echo $WEB_SERVER | tr '[:lower:]' '[:upper:]')${NC}"
echo -e "This preference has been saved for future runs.\n"

log "Starting service restart procedure"

# Detect Redis service name - could be redis.service or redis-server.service
REDIS_SERVICE=""
if systemctl list-unit-files | grep -q "redis-server.service"; then
    REDIS_SERVICE="redis-server.service"
elif systemctl list-unit-files | grep -q "redis.service"; then
    REDIS_SERVICE="redis.service"
else
    error "Redis service not found. Please check if Redis is installed properly."
    exit 1
fi
log "Detected Redis service: $REDIS_SERVICE"

# Function to replace environment variables in a file
replace_env_vars() {
    local input_file="$1"
    local output_file="$2"
    
    # Create a copy of the input file
    cp "$input_file" "$output_file"
    
    # Replace all environment variables with their values
    sed -i "s|\${BASE_DIR}|$BASE_DIR|g" "$output_file"
    sed -i "s|\${SCRIPT_DIR}|$SCRIPT_DIR|g" "$output_file"
    sed -i "s|\${WEBAPP_DIR}|$WEBAPP_DIR|g" "$output_file"
    sed -i "s|\${DEPENDENCIES_DIR}|$DEPENDENCIES_DIR|g" "$output_file"
    sed -i "s|\${HOME_DIR}|$HOME_DIR|g" "$output_file"
    sed -i "s|\${USER_DIR}|$USER_DIR|g" "$output_file"
    sed -i "s|\${DATA_DIR}|$DATA_DIR|g" "$output_file"
    sed -i "s|\${LOG_DIR}|$LOG_DIR|g" "$output_file"
    sed -i "s|\${CONFIG_DIR}|$CONFIG_DIR|g" "$output_file"
    sed -i "s|\${SYSTEMD_DIR}|$SYSTEMD_DIR|g" "$output_file"
    sed -i "s|\${APACHE_DIR}|$APACHE_DIR|g" "$output_file"
    sed -i "s|\${BIN_DIR}|$BIN_DIR|g" "$output_file"
}

# Copy systemd service files
log "Copying systemd service files..."
if [ -d "$SYSTEMD_DIR" ]; then
    # Copy all service files found in config directory to systemd
    for service_file in "$CONFIG_DIR"/*.service; do
        if [ -f "$service_file" ]; then
            service_name=$(basename "$service_file")
            # Skip redis.service if it's a system service
            if [ "$service_name" = "redis.service" ] && systemctl list-unit-files | grep -q "^redis.service"; then
                log "Skipping $service_name as it's already a system service"
                continue
            fi
            
            # Create a temporary file with variables replaced
            TEMP_CONF="${service_file}.temp"
            replace_env_vars "$service_file" "$TEMP_CONF"
            
            # Copy the processed file to systemd
            cp "$TEMP_CONF" "$SYSTEMD_DIR/$service_name" && \
            log "Copied $service_name to $SYSTEMD_DIR/" || \
            error "Failed to copy $service_name to $SYSTEMD_DIR/"
            rm "$TEMP_CONF"
        fi
    done
else
    error "Systemd directory $SYSTEMD_DIR not found. Cannot copy service files."
fi

# Copy Apache configuration files
log "Copying Apache configuration files..."
if [ -d "$APACHE_DIR" ]; then
    # Copy only Apache .conf files to Apache sites-available
    for conf_file in "$CONFIG_DIR"/*.conf; do
        # Skip files that end with .nginx.conf (Nginx configs)
        if [[ "$conf_file" == *.nginx.conf ]]; then
            log "Skipping Nginx config file: $conf_file"
            continue
        fi
        
        # Skip template files
        if [[ "$conf_file" == *.template ]]; then
            log "Skipping template file: $conf_file"
            continue
        fi
        
        if [ -f "$conf_file" ]; then
            conf_name=$(basename "$conf_file")
            # Create a temporary file with variables replaced
            TEMP_CONF="${conf_file}.temp"
            replace_env_vars "$conf_file" "$TEMP_CONF"
            
            # Copy the processed file
            cp "$TEMP_CONF" "$APACHE_DIR/$conf_name" && \
            log "Copied $conf_name to $APACHE_DIR/" || \
            error "Failed to copy $conf_name to $APACHE_DIR/"
            rm "$TEMP_CONF"
        fi
    done
    
    # Enable Apache sites
    log "Enabling Apache sites..."
    if [ -f "$APACHE_DIR/000-default.conf" ]; then
        a2ensite 000-default.conf 2>/dev/null || log "000-default.conf already enabled"
    fi
    if [ -f "$APACHE_DIR/ciwre-bae.conf" ]; then
        a2ensite ciwre-bae.conf 2>/dev/null || log "ciwre-bae.conf already enabled"
    fi
    
    # Reload Apache to apply changes - but don't exit if it fails
    log "Reloading Apache service..."
    if systemctl is-active --quiet apache2; then
        systemctl reload apache2 || error "Failed to reload Apache. Please check Apache configuration."
    else
        warning "Apache is not running, skipping reload"
    fi
else
    warning "Apache configuration directory $APACHE_DIR not found. Skipping Apache configuration."
fi

# Check and reload Nginx if it's running
log "Checking Nginx status..."
if command -v nginx &> /dev/null; then
    # Copy Nginx configuration files if they exist
    NGINX_DIR="/etc/nginx/sites-available"
    NGINX_ENABLED_DIR="/etc/nginx/sites-enabled"
    
    if [ -d "$NGINX_DIR" ]; then
        log "Copying Nginx configuration files..."
        
        # Clean up any leftover temporary files from previous runs
        find "$CONFIG_DIR" -name "*.temp" -type f -delete
        
        # Only process .nginx.conf files
        for conf_file in "$CONFIG_DIR"/*.nginx.conf; do
            if [ -f "$conf_file" ]; then
                conf_name=$(basename "$conf_file" .nginx.conf)
                
                # Create a temporary file with variables replaced
                TEMP_CONF="${conf_file}.temp"
                replace_env_vars "$conf_file" "$TEMP_CONF"
                
                # Copy the modified file
                cp "$TEMP_CONF" "$NGINX_DIR/${conf_name}.conf" && \
                log "Copied ${conf_name}.conf to $NGINX_DIR/" || \
                error "Failed to copy ${conf_name}.conf to $NGINX_DIR/"
                rm -f "$TEMP_CONF"
            fi
        done
        
        # Reload Nginx if it's running
        if systemctl is-active --quiet nginx; then
            log "Reloading Nginx service..."
            nginx -t && systemctl reload nginx || error "Failed to reload Nginx. Please check Nginx configuration."
        else
            warning "Nginx is not running, skipping reload"
        fi
    else
        warning "Nginx configuration directory $NGINX_DIR not found. Skipping Nginx configuration."
    fi
else
    log "Nginx not installed, skipping Nginx configuration"
fi

# Process template files for both Apache and Nginx if any exist
for template_file in "$CONFIG_DIR"/*.template; do
    if [ -f "$template_file" ]; then
        template_name=$(basename "$template_file" .template)
        target_file="$CONFIG_DIR/$template_name"
        
        log "Processing template file: $template_file"
        cp "$template_file" "$target_file.temp"
        
        # Replace template markers with actual values
        sed -i "s|##BASE_DIR##|$BASE_DIR|g" "$target_file.temp"
        sed -i "s|##WEBAPP_DIR##|$WEBAPP_DIR|g" "$target_file.temp"
        sed -i "s|##LOG_DIR##|$LOG_DIR|g" "$target_file.temp"
        sed -i "s|##USER_DIR##|$USER_DIR|g" "$target_file.temp"
        sed -i "s|##DATA_DIR##|$DATA_DIR|g" "$target_file.temp"
        
        # Move to final location
        mv "$target_file.temp" "$target_file"
        log "Created $template_name from template"
    fi
done

# Reload systemd to recognize the new service files
log "Reloading systemd daemon..."
systemctl daemon-reload

# Check Redis configuration
REDIS_CONF="/etc/redis/redis.conf"
if [ -f "$REDIS_CONF" ]; then
    log "Checking Redis configuration..."
    # Ensure Redis is listening on localhost
    if ! grep -q "^bind 127.0.0.1" "$REDIS_CONF"; then
        warning "Redis is not configured to bind to localhost. Adding bind directive."
        # Add or update bind directive
        if grep -q "^#.*bind" "$REDIS_CONF"; then
            sed -i 's/^#.*bind.*/bind 127.0.0.1/' "$REDIS_CONF"
        else
            echo "bind 127.0.0.1" >> "$REDIS_CONF"
        fi
    fi
    
    # Ensure Redis is not protected-mode (which can cause connection issues)
    if grep -q "^protected-mode yes" "$REDIS_CONF"; then
        warning "Redis is in protected mode. Disabling for local connections."
        sed -i 's/^protected-mode yes/protected-mode no/' "$REDIS_CONF"
    fi
    
    log "Redis configuration checked and updated if necessary."
else
    error "Redis configuration file not found at $REDIS_CONF"
    # Create a minimal Redis config if it doesn't exist
    log "Creating minimal Redis configuration..."
    mkdir -p /etc/redis
    cat > "$REDIS_CONF" << EOF
bind 127.0.0.1
protected-mode no
port 6379
dir /var/lib/redis
EOF
    log "Created minimal Redis configuration."
fi

# Disable set -e for the rest of the script to prevent early exit on errors
set +e

# Stop existing services in reverse dependency order
log "Stopping Flask application service..."
systemctl stop flask-app.service 2>/dev/null || log "Flask app service was not running"

log "Stopping Celery worker service..."
systemctl stop celery-worker.service 2>/dev/null || log "Celery worker service was not running"

log "Stopping Redis service..."
systemctl stop $REDIS_SERVICE 2>/dev/null || log "Redis service was not running"

# Wait for services to stop completely...
# for processes to fully stop
log "Waiting for services to stop completely..."
sleep 5

# Check for any remaining Redis processes
log "Checking for remaining Redis processes..."
if pgrep -f "redis-server" > /dev/null; then
    warning "Redis processes still running. Attempting to kill..."
    pkill -f "redis-server" || log "Failed to kill Redis processes"
    sleep 2
fi

# Check for any remaining Celery processes
log "Checking for remaining Celery processes..."
if pgrep -f "celery" > /dev/null; then
    warning "Celery processes still running. Attempting to kill..."
    pkill -f "celery" || log "Failed to kill Celery processes"
    sleep 2
fi

# Check for any remaining gunicorn processes on port 5001
log "Checking for gunicorn processes on port 5001..."
if lsof -i :5001 > /dev/null 2>&1; then
    warning "Gunicorn process still using port 5001. Attempting to kill..."
    pkill -f "gunicorn" || log "Failed to kill gunicorn processes"
    sleep 2
fi

# Check for any remaining gunicorn processes on port 5050
log "Checking for gunicorn processes on port 5050..."
if lsof -i :5050 > /dev/null 2>&1; then
    warning "Gunicorn process still using port 5050. Attempting to kill..."
    pkill -f "gunicorn" || log "Failed to kill gunicorn processes"
    # Try using lsof to find and kill the process directly
    pid=$(lsof -t -i:5050 2>/dev/null)
    if [ ! -z "$pid" ]; then
        warning "Killing process directly using PID: $pid"
        kill -9 $pid
    fi
    sleep 2
fi

# Create required directories with correct permissions
log "Ensuring proper directories and permissions..."
mkdir -p /var/lib/redis
chown -R www-data:www-data ${LOG_DIR}
# Only attempt to change Redis directory permissions if it exists and we have the right user
if [ -d "/var/lib/redis" ] && getent passwd redis >/dev/null; then
    chown -R redis:redis /var/lib/redis
    chmod 750 /var/lib/redis
fi

# Start services in dependency order
log "Starting Redis service..."
systemctl start $REDIS_SERVICE
sleep 3

# Verify Redis is running
if systemctl is-active --quiet $REDIS_SERVICE; then
    log "Redis service started successfully"
    
    # Test Redis connection directly
    if redis-cli ping > /dev/null; then
        log "Redis connection test successful"
    else
        error "Redis connection test failed. Please check Redis configuration."
    fi
else
    error "Failed to start Redis service"
    systemctl status $REDIS_SERVICE --no-pager
fi

# Start Celery worker
log "Starting Celery worker service..."
systemctl start celery-worker.service
sleep 3

# Verify Celery worker is running
if systemctl is-active --quiet celery-worker.service; then
    log "Celery worker service started successfully"
else
    error "Failed to start Celery worker service"
    systemctl status celery-worker.service --no-pager
fi

# Start Celery beat if it exists
if [ -f "$SYSTEMD_DIR/celery-beat.service" ]; then
    log "Starting Celery beat service..."
    systemctl start celery-beat.service
    sleep 2
    
    if systemctl is-active --quiet celery-beat.service; then
        log "Celery beat service started successfully"
    else
        warning "Failed to start Celery beat service"
        systemctl status celery-beat.service --no-pager
    fi
fi

# Start Flask application
log "Starting Flask application service..."
systemctl start flask-app.service
sleep 3

# Verify Flask application is running
if systemctl is-active --quiet flask-app.service; then
    log "Flask application service started successfully"
else
    error "Failed to start Flask application service"
    systemctl status flask-app.service --no-pager
fi

# Web server configuration/startup
if [ "$WEB_SERVER" = "nginx" ]; then
    log "Configuring and starting Nginx web server..."
    
    # Check if Nginx is installed
    if ! command -v nginx &> /dev/null; then
        warning "Nginx is not installed. Installing Nginx..."
        apt-get update && apt-get install -y nginx || error "Failed to install Nginx"
    fi
    
    # Stop Apache first to free up ports 80 and 443
    log "Stopping Apache to free up ports for Nginx..."
    systemctl stop apache2 || warning "Apache was not running or couldn't be stopped properly"
    systemctl disable apache2 || warning "Could not disable Apache autostart"
    
    # Create Nginx configuration directory if needed
    NGINX_DIR="/etc/nginx/sites-available"
    NGINX_ENABLED_DIR="/etc/nginx/sites-enabled"
    mkdir -p "$NGINX_DIR" "$NGINX_ENABLED_DIR"
    
    # Clean up any existing configurations in sites-enabled to avoid conflicts
    log "Cleaning up existing Nginx configurations..."
    rm -f "$NGINX_ENABLED_DIR"/* || warning "Could not clean up Nginx configurations"
    
    # Copy Nginx configuration files if they exist
    log "Copying Nginx configuration files..."
    NGINX_CONF_FOUND=false
    for conf_file in "$CONFIG_DIR"/*.nginx.conf; do
        if [ -f "$conf_file" ]; then
            NGINX_CONF_FOUND=true
            conf_name=$(basename "$conf_file" .nginx.conf)
            
            # Replace hardcoded paths with variables
            TEMP_CONF="${conf_file}.temp"
            cp "$conf_file" "$TEMP_CONF"
            sed -i "s|/data/SWATGenXApp/codes|${BASE_DIR}|g" "$TEMP_CONF"
            sed -i "s|/data/SWATGenXApp/GenXAppData|${BASE_DIR}/../GenXAppData|g" "$TEMP_CONF"
            sed -i "s|/data/SWATGenXApp/Users|${BASE_DIR}/../Users|g" "$TEMP_CONF"
            
            # Copy the modified file
            cp "$TEMP_CONF" "$NGINX_DIR/${conf_name}.conf" && \
            log "Copied ${conf_name}.conf to $NGINX_DIR/" || \
            error "Failed to copy ${conf_name}.conf to $NGINX_DIR/"
            rm -f "$TEMP_CONF"
            
            # Create symlinks in sites-enabled
            ln -sf "$NGINX_DIR/${conf_name}.conf" "$NGINX_ENABLED_DIR/${conf_name}.conf" && \
            log "Enabled ${conf_name}.conf site" || \
            error "Failed to enable ${conf_name}.conf site"
        fi
    done
    
    # If no Nginx config files exist, create a default one
    if [ "$NGINX_CONF_FOUND" = false ]; then
        warning "No Nginx configuration files found. Creating a default configuration..."
        
        cat > "$NGINX_DIR/swatgenx.conf" << EOF
server {
    listen 80;
    server_name localhost;

    # Serve the React build output
    root ${BASE_DIR}/web_application/frontend/build;
    index index.html;

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files
    location /static/ {
        alias ${BASE_DIR}/web_application/frontend/build/static/;
    }

    # All other requests go to the React app
    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF
        ln -sf "$NGINX_DIR/swatgenx.conf" "$NGINX_ENABLED_DIR/swatgenx.conf" && \
        log "Created and enabled default Nginx configuration with port 80"
    fi
    
    # Test Nginx configuration
    log "Testing Nginx configuration..."
    nginx -t && log "Nginx configuration test passed" || error "Nginx configuration test failed"
    
    # Restart Nginx
    log "Restarting Nginx web server..."
    systemctl restart nginx || error "Failed to restart Nginx. Please check Nginx configuration."
    
    # Verify Nginx is running
    if systemctl is-active --quiet nginx; then
        log "Nginx web server started successfully"
        systemctl enable nginx
        log "Nginx enabled to start on boot"
    else
        error "Failed to start Nginx web server"
        systemctl status nginx --no-pager
        warning "Falling back to Apache web server..."
        WEB_SERVER="apache"
    fi
elif [ "$WEB_SERVER" = "apache" ]; then
    log "Configuring and starting Apache web server..."
    
    # Stop Nginx if it's running
    if systemctl is-active --quiet nginx; then
        log "Stopping Nginx to avoid port conflicts..."
        systemctl stop nginx
        systemctl disable nginx
    fi
    
    # Enable Apache sites if they exist
    if [ -f "$APACHE_DIR/000-default.conf" ]; then
        a2ensite 000-default.conf 2>/dev/null || log "000-default.conf already enabled"
    fi
    if [ -f "$APACHE_DIR/ciwre-bae.conf" ]; then
        a2ensite ciwre-bae.conf 2>/dev/null || log "ciwre-bae.conf already enabled"
    fi
    
    # Make sure required Apache modules are enabled
    log "Enabling required Apache modules..."
    a2enmod proxy proxy_http proxy_wstunnel rewrite headers ssl 2>/dev/null || log "Apache modules already enabled"
    
    # Restart Apache to apply changes
    log "Starting Apache web server..."
    systemctl start apache2 || error "Failed to start Apache. Please check Apache configuration."
    
    # Verify Apache is running
    if systemctl is-active --quiet apache2; then
        log "Apache web server started successfully"
        systemctl enable apache2
        log "Apache enabled to start on boot"
    else
        error "Failed to start Apache web server"
        systemctl status apache2 --no-pager
    fi
elif [ "$WEB_SERVER" = "none" ]; then
    log "No web server selected. Stopping any running web servers..."
    systemctl stop apache2 > /dev/null 2>&1
    systemctl disable apache2 > /dev/null 2>&1
    systemctl stop nginx > /dev/null 2>&1
    systemctl disable nginx > /dev/null 2>&1
    log "Both Apache and Nginx have been stopped."
fi

# Enable services to start on boot
log "Enabling services to start on boot..."
systemctl enable $REDIS_SERVICE 2>/dev/null || warning "Could not enable Redis service - it may be linked or controlled by another unit"
systemctl enable celery-worker.service
systemctl enable flask-app.service
if [ -f "$SYSTEMD_DIR/celery-beat.service" ]; then
    systemctl enable celery-beat.service
fi

# Clean up any temporary files before exiting
find "$CONFIG_DIR" -name "*.temp" -type f -delete
log "Cleaned up temporary files"

# Final status check
log "Checking all service statuses..."
echo -e "\n${GREEN}=== Redis Service Status ===${NC}"
systemctl status $REDIS_SERVICE --no-pager
echo -e "\n${GREEN}=== Celery Worker Service Status ===${NC}"
systemctl status celery-worker.service --no-pager
echo -e "\n${GREEN}=== Flask Application Service Status ===${NC}"
systemctl status flask-app.service --no-pager

# Display web server status
if [ "$WEB_SERVER" = "apache" ]; then
    echo -e "\n${GREEN}=== Apache Web Server Status ===${NC}"
    systemctl status apache2 --no-pager
elif [ "$WEB_SERVER" = "nginx" ]; then
    echo -e "\n${GREEN}=== Nginx Web Server Status ===${NC}"
    systemctl status nginx --no-pager
else
    echo -e "\n${GREEN}=== No Web Server Selected ===${NC}"
fi

# Output final status for log
SERVICES_OK=true
if ! systemctl is-active --quiet $REDIS_SERVICE; then
    SERVICES_OK=false
fi
if ! systemctl is-active --quiet celery-worker.service; then
    SERVICES_OK=false
fi
if ! systemctl is-active --quiet flask-app.service; then
    SERVICES_OK=false
fi
if [ "$WEB_SERVER" = "apache" ] && ! systemctl is-active --quiet apache2; then
    SERVICES_OK=false
fi
if [ "$WEB_SERVER" = "nginx" ] && ! systemctl is-active --quiet nginx; then
    SERVICES_OK=false
fi

if [ "$SERVICES_OK" = true ]; then
    log "All services restarted successfully!"
else
    error "One or more services failed to start. Please check the logs."
fi

# Print monitoring instructions
echo -e "\n${GREEN}=== Monitoring Instructions ===${NC}"
echo -e "To monitor Redis: ${YELLOW}redis-cli monitor${NC}"
echo -e "To check Celery worker log: ${YELLOW}tail -f ${LOG_DIR}/celery-worker.log${NC}"
echo -e "To check Flask application log: ${YELLOW}tail -f ${LOG_DIR}/flask-app.log${NC}"
echo -e "To view task status: ${YELLOW}curl -X GET http://localhost:5050/api/user_tasks${NC}"

# Print web server status based on selection
echo -e "\n${GREEN}=== Production Web Server Information ===${NC}"
echo -e "Active web server: ${YELLOW}$(echo $WEB_SERVER | tr '[:lower:]' '[:upper:]')${NC}"
if [ "$WEB_SERVER" = "apache" ]; then
    echo -e "Apache config: ${YELLOW}${APACHE_DIR}/000-default.conf${NC}"
    echo -e "Apache logs: ${YELLOW}/var/log/apache2/error.log${NC}"
elif [ "$WEB_SERVER" = "nginx" ]; then
    echo -e "Nginx config: ${YELLOW}/etc/nginx/sites-enabled/swatgenx.conf${NC}"
    echo -e "Nginx logs: ${YELLOW}/var/log/nginx/error.log${NC}"
else
    echo -e "No web server selected. No configuration or logs available."
fi
echo -e "To change web server, run this script again and select a different option."

log "Service restart procedure completed"