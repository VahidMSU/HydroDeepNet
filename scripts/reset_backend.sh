#!/bin/bash

# Don't call self recursively
# sh ./reset_backend.sh  <- This line was causing recursion

echo "Setting up backend services..."

# Copy configuration files
sudo cp ./ciwre-bae.conf /etc/apache2/sites-available/ciwre-bae.conf && echo "✓ ciwre-bae.conf copied"
sudo cp ./000-default.conf /etc/apache2/sites-available/000-default.conf && echo "✓ 000-default.conf copied"
sudo cp ./celery-worker.service /etc/systemd/system/celery-worker.service && echo "✓ celery-worker.service copied"

## Set root permissions
echo "Setting file permissions..."
sudo chown root:root /etc/apache2/sites-available/ciwre-bae.conf
sudo chown root:root /etc/apache2/sites-available/000-default.conf
sudo chown root:root /etc/systemd/system/celery-worker.service

## Restart services
echo "Restarting services..."
sudo systemctl daemon-reload && echo "✓ Daemon reloaded"
# Enable redis
sudo systemctl restart redis-server && echo "✓ Redis server restarted" || echo "⚠ Failed to restart redis-server"
sudo systemctl restart celery-worker.service && echo "✓ Celery worker restarted" || echo "⚠ Failed to restart celery-worker"
sudo systemctl restart flask-app.service && echo "✓ Flask app restarted" || echo "⚠ Failed to restart flask-app"
sudo systemctl restart apache2 && echo "✓ Apache2 restarted" || echo "⚠ Failed to restart apache2"

echo "Backend reset completed."
