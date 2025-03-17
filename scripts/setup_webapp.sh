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