## Frontend setup

source global_path.sh
cd $FRONTEND_DIR
echo "Auditing packages and cleaning up unused dependencies..."
npm audit fix
npm prune
echo "Building the application..."
npm run build

