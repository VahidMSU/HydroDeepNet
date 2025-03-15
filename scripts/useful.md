## check ports listening
sudo lsof -i :5000
sudo lsof -i :3000
sudo lsof -i :80

## kill example
sudo kill -9 136168 1533998

## Restart 
sudo systemctl restart flask-app.service
sudo systemctl restart apache2

## Configure your Git user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"