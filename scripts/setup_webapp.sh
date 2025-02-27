## load demon service

sudo cp ./flask_app.conf /etc/apache2/sites-available/flask_app.conf
sudo cp ./ciwre-bae.conf /etc/apache2/sites-available/ciwre-bae.conf
sudo cp ./000-default.conf /etc/apache2/sites-available/000-default.conf
sudo cp ./celery-worker.service /etc/systemd/system/celery-worker.service
## root permission
sudo chown root:root /etc/apache2/sites-available/flask_app.conf
sudo chown root:root /etc/apache2/sites-available/ciwre-bae.conf
sudo chown root:root /etc/apache2/sites-available/000-default.conf
sudo chown root:root /etc/systemd/system/celery-worker.service
## restart service
sudo systemctl daemon-reload
# Enable redis
sudo systemctl restart redis-server
sudo systemctl restart celery-worker.service
sudo systemctl restart flask-app.service
sudo systemctl restart apache2


cd /data/SWATGenXApp/codes/web_application/frontend

# Update package-lock.json to match package.json
#npm install

# Use npm ci for faster, more reliable builds
#npm ci

# Run update, audit fix, and build in parallel
#npx concurrently "npm update" "npm audit fix" "npm run build"
npm run build
# Start the application
npm start