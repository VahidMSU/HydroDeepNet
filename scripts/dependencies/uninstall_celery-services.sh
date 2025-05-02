# Remove systemd service files related to Celery using sudo


CURRENT_DIR=$(dirname "$(readlink -f "$0")")
source "${CURRENT_DIR}/../global_path.sh"

## run only with sudo
echo "This script must be run with sudo."
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or use sudo."
    exit 1
fi

if [ -f "${SYSTEMD_DIR}/celery-worker.service" ]; then
    echo "Removing celery-worker.service from systemd..."
    rm -f "${SYSTEMD_DIR}/celery-worker.service" && echo "celery-worker.service removed."
fi

if [ -f "${SYSTEMD_DIR}/celery-beat.service" ]; then
    echo "Removing celery-beat.service from systemd..."
    rm -f "${SYSTEMD_DIR}/celery-beat.service" && echo "celery-beat.service removed."
fi

echo "Reloading systemd daemon..."
systemctl daemon-reload
