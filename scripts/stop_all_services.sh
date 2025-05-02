###
sudo systemctl status redis.service
sudo systemctl stop redis.service
sudo systemctl status redis.service

sudo systemctl status celery-worker.service
sudo systemctl stop celery-worker.service
sudo systemctl status celery-worker.service

sudo systemctl status flask-app.service
sudo systemctl stop flask-app.service
sudo systemctl status flask-app.service

sudo systemctl status nginx.service
sudo systemctl stop nginx
sudo systemctl status nginx.service

sudo systemctl status apache2.service
sudo systemctl stop apache2.service
sudo systemctl status apache2.service
