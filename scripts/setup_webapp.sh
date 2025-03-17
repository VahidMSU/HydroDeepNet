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

npm start &
echo "Frontend setup complete"
echo "Web application setup complete"
echo "SWATGenXApp is now running at http://localhost:3000"
echo "Please visit the URL to access the application"
echo "Please note that the application is running in the background"
echo "To stop the application, run the following command:"
echo "killall node"
echo "To restart the application, run the following command:"
echo "bash restart_services.sh"