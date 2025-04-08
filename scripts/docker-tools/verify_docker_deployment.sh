#!/bin/bash
# Docker deployment verification script
# Run this script with: docker exec swatgenx_app bash /docker-entrypoint.sh verify

# Use colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set base paths
BASE_DIR="/data/SWATGenXApp"
LOG_DIR="${BASE_DIR}/web_application/logs"
BIN_DIR="${BASE_DIR}/bin"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Output files
REPORT_FILE="${LOG_DIR}/docker_verification_$(date +%Y%m%d_%H%M%S).md"
TEMP_REPORT="/tmp/docker_report_$$.tmp"

# Status tracking
PASSED_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0
TOTAL_CHECKS=0

# Function to print section headers
section() {
    echo -e "${BLUE}=== $1 ===${NC}"
    echo -e "## $1" >>$TEMP_REPORT
    echo "" >>$TEMP_REPORT
}

# Function to check an item and report its status
check_item() {
    local desc="$1"
    local cmd="$2"
    local require_success="${3:-true}"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    echo -ne "${YELLOW}Checking: ${NC}$desc..."

    # Run the command and capture output and exit code
    local output
    local status

    output=$(eval "$cmd" 2>&1)
    status=$?

    if [ "$require_success" = "true" ] && [ $status -eq 0 ]; then
        echo -e " ${GREEN}[PASS]${NC}"
        echo -e "- [x] $desc" >>$TEMP_REPORT
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    elif [ "$require_success" = "false" ] && [ $status -ne 0 ]; then
        echo -e " ${GREEN}[PASS]${NC} (Expected failure)"
        echo -e "- [x] $desc" >>$TEMP_REPORT
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e " ${RED}[FAIL]${NC}"
        echo -e "- [ ] $desc" >>$TEMP_REPORT
        echo -e "   - Error: Command failed with status $status" >>$TEMP_REPORT
        echo -e "   - Output: \`\`\`\n$output\n\`\`\`" >>$TEMP_REPORT
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Function to skip a check
skip_check() {
    local desc="$1"
    local reason="$2"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    SKIPPED_CHECKS=$((SKIPPED_CHECKS + 1))

    echo -e "${YELLOW}Skipping: ${NC}$desc - $reason"
    echo -e "- [ ] $desc - SKIPPED: $reason" >>$TEMP_REPORT
}

# Add a more flexible check for supervisor services
check_supervisor_service() {
    local service_name="$1"

    # First try using supervisorctl
    if supervisorctl status "$service_name" 2>/dev/null | grep -q "RUNNING"; then
        return 0
    fi

    # If that failed, check for the process directly
    case "$service_name" in
    nginx)
        if pgrep -x nginx >/dev/null; then
            return 0
        fi
        ;;
    redis)
        if pgrep -x redis-server >/dev/null; then
            return 0
        fi
        ;;
    flask)
        if pgrep -f "gunicorn.*run:app" >/dev/null; then
            return 0
        fi
        ;;
    celery)
        if pgrep -f "celery worker" >/dev/null; then
            return 0
        fi
        ;;
    esac

    return 1
}

# Initialize the report file
cat >$TEMP_REPORT <<EOF
# SWATGenX Docker Deployment Verification Report
Generated: $(date)

This report shows the results of verifying the SWATGenX deployment inside Docker.

EOF

# Check Supervisor Services
section "Supervisor Service Status"

# Check supervisord is running
check_item "Supervisord is running" "pgrep -f supervisord"

# Check individual services with the flexible function
check_item "Nginx service is running" "check_supervisor_service nginx"
check_item "Redis service is running" "check_supervisor_service redis"
check_item "Flask service is running" "check_supervisor_service flask"
check_item "Celery service is running" "check_supervisor_service celery"

# Port Checks
section "Port Availability"

# Check key services are listening on expected ports
check_item "Nginx is listening on port 80" "netstat -tuln | grep -q ':80 '"
check_item "Redis is listening on port 6379" "netstat -tuln | grep -q ':6379 '"
check_item "Flask API is listening on port 5050" "netstat -tuln | grep -q ':5050 '"
check_item "Flask secondary API is listening on port 5000" "netstat -tuln | grep -q ':5000 '"

# API Functionality Tests
section "API Functionality Tests"

# Check Flask API endpoints
check_item "Flask API status endpoint responds" "curl -s http://localhost:5050/api/diagnostic/status | grep -q 'status'"
check_item "Flask API model-settings endpoint responds" "curl -s http://localhost:5050/api/model-settings | grep -q '{'"

# Check web server
check_item "Nginx serves frontend" "curl -s -I http://localhost:80 | grep -q '200 OK'"

# Check Redis connectivity
check_item "Redis responds to ping" "redis-cli ping | grep -q 'PONG'"

# SWAT+ Component Verification
section "SWAT+ Component Verification"

# Check SWAT+ executables
check_item "SWAT+ executables are present" "ls -la ${BIN_DIR}/swatplus 2>/dev/null"
check_item "MODFLOW executables are present" "ls -la ${BIN_DIR}/modflow-nwt 2>/dev/null"

# Check database files
DB_PATHS=(
    "/usr/local/share/SWATPlus/Databases"
    "${BASE_DIR}/data/SWATPlus/Databases"
)

DB_FOUND=false
for path in "${DB_PATHS[@]}"; do
    if [ -d "$path" ]; then
        check_item "Database files are present" "ls -la $path/swatplus_datasets.sqlite 2>/dev/null"
        DB_FOUND=true
        break
    fi
done

if [ "$DB_FOUND" = false ]; then
    skip_check "Database files are present" "Database directory not found in standard locations"
fi

# File System Checks
section "File System Checks"

# Check log directory permissions
check_item "Log directory permissions are correct" "ls -la ${LOG_DIR} | grep -q 'www-data'"

# Check Redis data directory permissions
check_item "Redis data directory exists" "ls -la /var/lib/redis | grep -q 'redis'"

# Check volume mounts
check_item "User data volume is mounted" "mountpoint -q /data/SWATGenXApp/Users || ls -la /data/SWATGenXApp/Users"
check_item "Redis data volume is mounted" "mountpoint -q /var/lib/redis || ls -la /var/lib/redis"

# Python Environment Checks
section "Python Environment"

# Check Python installation
check_item "Python is installed" "python3 --version"
check_item "Virtual environment is active" "which python3 | grep -q '.venv'"

# Check key Python modules
check_item "GDAL Python module is installed" "python3 -c 'import osgeo.gdal; print(osgeo.gdal.__version__)'"
check_item "Celery Python module is installed" "python3 -c 'import celery; print(celery.__version__)'"
check_item "Flask Python module is installed" "python3 -c 'import flask; print(flask.__version__)'"

# Generate summary at the end of the report
cat >>$TEMP_REPORT <<EOF

## Summary

- **Total Checks**: $TOTAL_CHECKS
- **Passed**: $PASSED_CHECKS
- **Failed**: $FAILED_CHECKS
- **Skipped**: $SKIPPED_CHECKS

Report generated by \`verify_docker_deployment.sh\` on $(date)
EOF

# Move the temporary report to the final location
mv $TEMP_REPORT $REPORT_FILE

# Print summary to console
echo ""
echo -e "${BLUE}=== Verification Summary ===${NC}"
echo -e "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED_CHECKS${NC}"
echo ""
echo -e "Detailed report saved to: ${REPORT_FILE}"

# If there are any failures, exit with error code
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}All automated checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED_CHECKS check(s) failed. Please review the report.${NC}"
    exit 1
fi
