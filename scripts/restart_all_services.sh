###
sudo systemctl status redis.service
sudo systemctl restart redis.service
sudo systemctl status redis.service

sudo systemctl status celery-worker.service
sudo systemctl restart celery-worker.service
sudo systemctl status celery-worker.service

sudo systemctl status flask-app.service
sudo systemctl restart flask-app.service
sudo systemctl status flask-app.service

sudo systemctl status nginx.service
sudo systemctl restart nginx
sudo systemctl status nginx.service

sudo systemctl status apache2.service
sudo systemctl restart apache2.service
sudo systemctl status apache2.service
