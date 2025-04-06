# SWATGenX Deployment Checklist

## System Services Setup
- [ ] Copy service files to systemd:
  - [ ] celery-worker.service
  - [ ] celery-multi-worker.service (if using)
  - [ ] celery-beat.service (if using scheduled tasks)
  - [ ] flask-app.service
  - [ ] redis service config (system-provided)
  
- [ ] Copy Apache config files:
  - [ ] 000-default.conf
  - [ ] ciwre-bae.conf
  
- [ ] Run `systemctl daemon-reload`
- [ ] Enable all services to start on boot
- [ ] Run the full restart_services.sh script

## File Permissions
- [ ] Ensure /data/SWATGenXApp/codes/web_application/logs is owned by www-data
- [ ] Ensure proper Redis permissions
- [ ] Set proper permissions for user directories

## Verification Steps
- [ ] Verify Redis is running: `redis-cli ping`
- [ ] Verify Celery workers: `celery -A celery_worker inspect ping`
- [ ] Verify Flask API is responding: `curl http://localhost:5050/api/diagnostic/status`
- [ ] Verify Apache is serving the frontend: `curl -I https://ciwre-bae.campusad.msu.edu`
