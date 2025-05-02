#!/bin/bash
# verify_deployment.sh
# This script automatically checks all the items in the deployment checklist
# and generates a report indicating which items passed and which failed

# Use colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/global_path.sh"

# Output files
REPORT_FILE="${LOG_DIR}/deployment_verification_$(date +%Y%m%d_%H%M%S).md"
TEMP_REPORT="/tmp/deployment_report_$$.tmp"

# Status tracking
PASSED_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0
TOTAL_CHECKS=0

# Function to print section headers
section() {
    echo -e "${BLUE}=== $1 ===${NC}"
    echo -e "## $1" >> $TEMP_REPORT
    echo "" >> $TEMP_REPORT
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
        echo -e "- [x] $desc" >> $TEMP_REPORT
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    elif [ "$require_success" = "false" ] && [ $status -ne 0 ]; then
        echo -e " ${GREEN}[PASS]${NC} (Expected failure)"
        echo -e "- [x] $desc" >> $TEMP_REPORT
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e " ${RED}[FAIL]${NC}"
        echo -e "- [ ] $desc" >> $TEMP_REPORT
        echo -e "   - Error: Command failed with status $status" >> $TEMP_REPORT
        echo -e "   - Output: \`\`\`\n$output\n\`\`\`" >> $TEMP_REPORT
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
    echo -e "- [ ] $desc - SKIPPED: $reason" >> $TEMP_REPORT
}

# Initialize the report file
cat > $TEMP_REPORT << EOF
# SWATGenX Deployment Verification Report
Generated: $(date)

This report shows the results of automatically verifying the SWATGenX deployment.

EOF

# Check if running as root (required for some checks)
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}This script should be run as root to perform all checks.${NC}"
    echo -e "${YELLOW}Running with limited permissions. Some checks may be skipped.${NC}"
    echo -e "**Warning: Script not run with root privileges. Some checks may be incomplete.**\n" >> $TEMP_REPORT
    IS_ROOT=false
else
    IS_ROOT=true
fi

# Check Service Status Section
section "Service Status Verification"

# Check Redis
check_item "Redis service is active" "systemctl is-active --quiet redis-server.service"

# Check Celery worker
check_item "Celery worker is active" "systemctl is-active --quiet celery-worker.service"

# Check Celery beat if configured
if [ -f "$SYSTEMD_DIR/celery-beat.service" ]; then
    check_item "Celery beat is active" "systemctl is-active --quiet celery-beat.service"
else
    skip_check "Celery beat is active" "Service file not found (scheduled tasks may not be configured)"
fi

# Check Flask application
check_item "Flask application is active" "systemctl is-active --quiet flask-app.service"

# Check web server status based on preference
WEB_SERVER_PREF=$(cat "$CONFIG_DIR/webserver_preference" 2>/dev/null || echo "apache")

if [ "$WEB_SERVER_PREF" = "apache" ]; then
    check_item "Apache web server is active" "systemctl is-active --quiet apache2.service"
elif [ "$WEB_SERVER_PREF" = "nginx" ]; then
    check_item "Nginx web server is active" "systemctl is-active --quiet nginx.service"
else
    skip_check "Web server is active" "No web server selected in preferences"
fi

# Service Functionality Tests Section
section "Service Functionality Tests"

# Check Redis connectivity
check_item "Redis responds to ping" "redis-cli ping | grep -q 'PONG'"

# Check Celery task submission
CELERY_TEST="cd ${WEBAPP_DIR} && ${BASE_DIR}/.venv/bin/python -c \"from celery_worker import celery; result=celery.send_task('tasks.debug_task'); print(f'Task ID: {result.id}')\""
check_item "Celery worker accepts tasks" "$CELERY_TEST"

# Check Flask API accessibility
check_item "Flask API status endpoint responds" "curl -s http://localhost:5050/api/diagnostic/status | grep -q 'status'"

# Check Flask API model settings endpoint
check_item "Flask API model-settings endpoint responds" "curl -s http://localhost:5050/api/model-settings | grep -q '{'"

# Check web server frontend access
if [ "$WEB_SERVER_PREF" = "apache" ] || [ "$WEB_SERVER_PREF" = "nginx" ]; then
    check_item "Web server HTTP access works" "curl -s -I http://localhost | grep -q '200 OK'"
    
    # Only check HTTPS if we have HTTPS configured
    if [ -f "/etc/ssl/certs/ciwre-bae_campusad_msu_edu_cert.cer" ]; then
        check_item "Web server HTTPS access works" "curl -sk -I https://localhost | grep -q '200 OK'"
    else
        skip_check "Web server HTTPS access works" "SSL certificates not found"
    fi
else
    skip_check "Web server HTTP access works" "No web server selected in preferences"
    skip_check "Web server HTTPS access works" "No web server selected in preferences"
fi

# File & Directory Verification Section
section "File & Directory Verification"

# Check log directory permissions
check_item "Log directory permissions are correct" "ls -la ${LOG_DIR} | grep -q 'www-data'"

# Check Redis data directory permissions
if [ "$IS_ROOT" = true ]; then
    check_item "Redis data directory permissions" "ls -la /var/lib/redis | grep -q 'redis'"
else
    skip_check "Redis data directory permissions" "Cannot check without root privileges"
fi

# Check static files accessibility
check_item "Static files are accessible via web server" "curl -s -I http://localhost/static/react/ 2>/dev/null | grep -q '200 OK'"

# SWAT+ Component Verification Section
section "SWAT+ Component Verification"

# Check SWAT+ executables
check_item "SWAT+ executables are present" "ls -la ${BIN_DIR}/swatplus 2>/dev/null"

# Check MODFLOW executables
check_item "MODFLOW executables are present" "ls -la ${BIN_DIR}/modflow-nwt 2>/dev/null"

# Check database files
DB_PATHS=(
    "/usr/local/share/SWATPlus/Databases"
    "${HOME}/.local/share/SWATPlus/Databases"
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

# Test SWAT+ executable if present
if [ -x "${BIN_DIR}/swatplus" ]; then
    check_item "SWAT+ executable runs" "${BIN_DIR}/swatplus --version 2>/dev/null"
else
    skip_check "SWAT+ executable runs" "Executable not found"
fi

# Add Application Testing Section
section "Application Testing"

echo -e "${YELLOW}The following checks require manual verification:${NC}"
echo -e "1. Open the web interface in a browser and verify the UI loads correctly"
echo -e "2. Test user login if authentication is enabled"
echo -e "3. Run a sample model to verify the whole pipeline works"
echo -e "4. Verify data visualization components function correctly"

echo -e "The following checks require manual verification:\n" >> $TEMP_REPORT
echo -e "- [ ] Open the web interface in a browser and verify the UI loads correctly" >> $TEMP_REPORT
echo -e "- [ ] Test user login if authentication is enabled" >> $TEMP_REPORT
echo -e "- [ ] Run a sample model to verify the whole pipeline works" >> $TEMP_REPORT
echo -e "- [ ] Verify data visualization components function correctly" >> $TEMP_REPORT

# Production Readiness Section
section "Production Readiness"

# Check log rotation
check_item "Logs are being properly rotated" "ls -la /etc/logrotate.d/ | grep -E 'apache|nginx|redis'"

# Check system resource limits
check_item "System resource limits are appropriate" "ulimit -n | awk '{if(\$1>=1024) exit 0; else exit 1}'"

# Generate summary at the end of the report
cat >> $TEMP_REPORT << EOF

## Summary

- **Total Checks**: $TOTAL_CHECKS
- **Passed**: $PASSED_CHECKS
- **Failed**: $FAILED_CHECKS
- **Skipped**: $SKIPPED_CHECKS

Report generated by \`verify_deployment.sh\` on $(date)
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

# If we're in an interactive terminal, offer to open the report
if [ -t 1 ] && command -v less &>/dev/null; then
    read -p "View the detailed report now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        less "$REPORT_FILE"
    fi
fi

# Exit with success if all checks passed, failure otherwise
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}All automated checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED_CHECKS check(s) failed. Please review the report.${NC}"
    exit 1
fi
