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