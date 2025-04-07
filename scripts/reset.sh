source global_path.sh
cp "${DATA_DIR}/USGS/FPS_CONUS_stations.geojson" "${FRONTEND_DIR}/public/static/stations.geojson"
sudo bash kill_port_process.sh
sudo bash setup_webapp.sh
sudo bash restart_services.sh
cd $FRONTEND_DIR
npm start