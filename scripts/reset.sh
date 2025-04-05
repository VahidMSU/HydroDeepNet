cp /data/SWATGenXApp/GenXAppData/USGS/FPS_CONUS_stations.geojson /data/SWATGenXApp/codes/web_application/frontend/public/static/stations.geojson
sudo bash kill_port_process.sh
sudo bash setup_webapp.sh
sudo bash restart_services.sh
cd /data/SWATGenXApp/codes/web_application/frontend
npm start