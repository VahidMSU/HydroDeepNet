#!/bin/bash
# Create log directories if they don't exist
mkdir -p /var/log/supervisor /var/log/redis /var/log/flask /var/log/celery
chown -R www-data:www-data /var/log/flask /var/log/celery
chown -R redis:redis /var/log/redis

# Ensure Redis data directory exists and has correct permissions
mkdir -p /var/lib/redis
chown -R redis:redis /var/lib/redis
chmod -R 770 /var/lib/redis

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
