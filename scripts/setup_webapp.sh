## Load daemon services
echo "Setting up daemon services..."
sudo cp ./flask_app.conf /etc/apache2/sites-available/flask_app.conf
sudo cp ./ciwre-bae.conf /etc/apache2/sites-available/ciwre-bae.conf
sudo cp ./000-default.conf /etc/apache2/sites-available/000-default.conf
sudo cp ./celery-worker.service /etc/systemd/system/celery-worker.service

## Set proper root permissions
echo "Setting up root permissions..."
sudo chown root:root /etc/apache2/sites-available/flask_app.conf
sudo chown root:root /etc/apache2/sites-available/ciwre-bae.conf
sudo chown root:root /etc/apache2/sites-available/000-default.conf
sudo chown root:root /etc/systemd/system/celery-worker.service

echo "Restarting services..."
sudo systemctl daemon-reload
echo "Restarting redis-server, celery-worker, flask-app, and apache2..."
sudo systemctl restart redis-server
sudo systemctl restart celery-worker.service
sudo systemctl restart flask-app.service
sudo systemctl restart apache2

bash check_services.sh
bash check_static_dirs.sh

## Frontend setup
cd /data/SWATGenXApp/codes/web_application/frontend

echo "Auditing packages and cleaning up unused dependencies..."
npm audit fix
npm prune

echo "Building the application..."
npm run build

bash check_services.sh
bash check_static_dirs.sh

echo "Starting the application..."
# Run npm start in the background so the script can complete
npm start &
echo "Application started in the background"