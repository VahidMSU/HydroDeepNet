## load demon service
cp ./flask_app.conf /etc/apache2/sites-available/flask_app.conf
cp ./ciwre-bae.conf /etc/apache2/sites-available/ciwre-bae.conf
cp ./000-default.conf /etc/apache2/sites-available/000-default.conf
cp ./celery-worker.service /etc/systemd/system/celery-worker.service

## root permission
chown root:root /etc/apache2/sites-available/flask_app.conf
chown root:root /etc/apache2/sites-available/ciwre-bae.conf
chown root:root /etc/apache2/sites-available/000-default.conf
chown root:root /etc/systemd/system/celery-worker.service

## restart service
systemctl daemon-reload
# Enable redis
systemctl restart redis-server
systemctl restart celery-worker.service
systemctl restart flask-app.service
systemctl restart apache2

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