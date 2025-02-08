## load demon service

sudo cp ./flask_app.conf /etc/apache2/sites-available/flask_app.conf
sudo cp ./ciwre-bae.conf /etc/apache2/sites-available/ciwre-bae.conf
sudo cp ./000-default.conf /etc/apache2/sites-available/000-default.conf
## root permission
sudo chown root:root /etc/apache2/sites-available/flask_app.conf
sudo chown root:root /etc/apache2/sites-available/ciwre-bae.conf
sudo chown root:root /etc/apache2/sites-available/000-default.conf
## restart service
sudo systemctl daemon-reload
sudo systemctl restart flask-app.service
sudo systemctl restart apache2
cd /data/SWATGenXApp/codes/web_application/frontend
npm run build
npm start