#!/bin/sh

# Set ownership to apache user (usually www-data)
APACHE_USER="www-data"
APACHE_GROUP="www-data"

# Process each directory
echo "Setting permissions for /data/SWATGenXApp/GenXAppData"
sudo chown -R $APACHE_USER:$APACHE_GROUP "/data/SWATGenXApp/GenXAppData"
sudo chmod -R 755 "/data/SWATGenXApp/GenXAppData"

echo "Setting permissions for /data/SWATGenXApp/Users"
sudo chown -R $APACHE_USER:$APACHE_GROUP "/data/SWATGenXApp/Users"
sudo chmod -R 755 "/data/SWATGenXApp/Users"

echo "Setting permissions for /data/SWATGenXApp/codes/web_application/logs"
sudo chown -R $APACHE_USER:$APACHE_GROUP "/data/SWATGenXApp/codes/web_application/logs"
sudo chmod -R 777 "/data/SWATGenXApp/codes/web_application/logs"

# Set SELinux context if SELinux is enabled
if command -v sestatus >/dev/null 2>&1; then
    if sestatus | grep -q "SELinux status:\s*enabled"; then
        echo "Setting SELinux context"
        sudo semanage fcontext -a -t httpd_sys_content_t "/data/SWATGenXApp/GenXAppData(/.*)?"
        sudo semanage fcontext -a -t httpd_sys_rw_content_t "/data/SWATGenXApp/Users(/.*)?"
        sudo semanage fcontext -a -t httpd_log_t "/data/SWATGenXApp/codes/web_application/logs(/.*)?"
        sudo restorecon -R -v /data/SWATGenXApp/
    fi
fi

echo "Permissions set successfully!"
