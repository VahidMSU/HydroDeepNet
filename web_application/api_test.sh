#!/bin/bash
# API Testing Script for SWATGenX
# Tests celery configuration, Redis connectivity, and task submission

# Set constants
API_BASE_URL="http://localhost:5050"
LOG_FILE="api_test_$(date +%Y%m%d_%H%M%S).log"
AUTH_TOKEN=""
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Authentication credentials
USERNAME="admin"
PASSWORD="Rafiei@110220"  # This should be modified in a production environment

# List of known good site numbers (as fallback)
KNOWN_SITE_NUMBERS=(
    "04080950" "01359513" "04115000" "04132000" "05471040" 
    "05469990" "06280300" "06906300" "13316530" "04144500"
)

# Default resolution values
LS_RESOLUTION=250  # Set explicitly to 250
DEM_RESOLUTION=30  # Set explicitly to 30

# Test configuration flags
SKIP_EMAIL_CHECK=true  # Set to true to skip email verification in test
DEBUG_EMAIL=true       # Set to true to debug email functionality

# Log function
log() {
    echo -e "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Login and get session token
login() {
    log "${YELLOW}Authenticating as ${USERNAME}${NC}"
    
    response=$(curl -s -c cookies.txt -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}" \
        "${API_BASE_URL}/api/login")
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" -eq 200 ]; then
        log "${GREEN}✓ Authentication successful${NC}"
        echo "$body" | jq .
        AUTH_TOKEN="$(grep -oP 'session=\K[^;]+' cookies.txt || echo "")"
        log "Session token stored for subsequent requests"
        return 0
    else
        log "${RED}✗ Authentication failed with status code: $status_code${NC}"
        echo "$body" | jq . 2>/dev/null || echo "$body"
        return 1
    fi
}

# Test function that makes API requests and checks responses
test_api() {
    endpoint=$1
    method=${2:-GET}
    data=${3:-"{}"}
    expected_status=${4:-200}
    
    log "${YELLOW}Testing $method $endpoint${NC}"
    
    # Add full request logging for debugging
    if [ "$method" == "POST" ]; then
        log "Request data: $data"
    fi
    
    # Create a temporary file to store the data for easier debugging
    if [ "$method" == "POST" ] && [ "$DEBUG_EMAIL" == "true" ]; then
        echo "$data" > /tmp/api_test_data.json
        log "Data saved to /tmp/api_test_data.json for inspection"
    fi
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -b cookies.txt -w "\n%{http_code}" -X $method "$API_BASE_URL$endpoint")
    else
        # Using verbose mode for detailed information when debugging is needed
        if [[ "$DEBUG_EMAIL" == "true" && "$endpoint" == *"model-settings"* ]]; then
            log "Running with verbose output for email debugging"
            response=$(curl -v -s -b cookies.txt -w "\n%{http_code}" \
                -X $method \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$API_BASE_URL$endpoint" 2>&1)
            log "Verbose response: $response"
        else
            response=$(curl -s -b cookies.txt -w "\n%{http_code}" \
                -X $method \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$API_BASE_URL$endpoint")
        fi
    fi
    
    # Extract status code and body
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    # Check status code
    if [ "$status_code" -eq "$expected_status" ]; then
        log "${GREEN}✓ Status code: $status_code (expected: $expected_status)${NC}"
    else
        log "${RED}✗ Status code: $status_code (expected: $expected_status)${NC}"
    fi
    
    # Parse JSON response
    echo "$body" | jq . 2>/dev/null || echo "$body"
    
    # Set global variable with response body to be used by other functions
    RESPONSE_BODY="$body"
    
    # Return 0 if test passed, 1 otherwise
    [ "$status_code" -eq "$expected_status" ]
    return $?
}

# Function to get a random successful site number from past tasks
get_random_site_number() {
    log "${YELLOW}Getting list of previously successful site numbers${NC}"
    
    # Fetch user tasks to see which sites were successfully processed
    test_api "/api/user_tasks" "GET"
    
    # Extract site numbers from successful tasks
    SUCCESSFUL_SITES=$(echo "$RESPONSE_BODY" | jq -r '.tasks[] | select(.status == "SUCCESS") | .site_no' 2>/dev/null)
    
    if [ -z "$SUCCESSFUL_SITES" ]; then
        # If no successful tasks found, use predefined list
        log "${YELLOW}No successful tasks found, using predefined site numbers${NC}"
        # Pick a random site from the known good list
        RANDOM_SITE=${KNOWN_SITE_NUMBERS[$RANDOM % ${#KNOWN_SITE_NUMBERS[@]}]}
    else
        # Convert successful sites to array
        readarray -t SITE_ARRAY <<< "$SUCCESSFUL_SITES"
        # Pick a random site from successful ones
        RANDOM_SITE=${SITE_ARRAY[$RANDOM % ${#SITE_ARRAY[@]}]}
    fi
    
    log "${GREEN}Selected random site number: $RANDOM_SITE${NC}"
    echo "$RANDOM_SITE"
}

# Main test execution
main() {
    echo "============== SWATGenX API Test =============="
    echo "Starting tests at $(date)"
    echo "API Base URL: $API_BASE_URL"
    echo "Log file: $LOG_FILE"
    echo "Authenticated user: $USERNAME"
    echo "Using resolution settings: LS=$LS_RESOLUTION, DEM=$DEM_RESOLUTION"
    if [ "$SKIP_EMAIL_CHECK" == "true" ]; then
        echo "⚠️ Email verification is disabled for testing"
    fi
    echo "=============================================="
    
    # Create log file
    > "$LOG_FILE"
    
    # First authenticate
    login
    if [ $? -ne 0 ]; then
        log "${RED}Authentication failed. Cannot continue tests.${NC}"
        exit 1
    fi
    
    # Test system info endpoint
    log "\n${YELLOW}🔍 Testing System Info${NC}"
    test_api "/api/debug/system-info" "GET"
    
    # Test Redis connectivity
    log "\n${YELLOW}🔍 Testing Redis Connectivity${NC}"
    test_api "/api/debug/redis-check" "GET"
    
    # Test Celery configuration
    log "\n${YELLOW}🔍 Testing Celery Configuration${NC}"
    test_api "/api/debug/celery-config" "GET"
    
    # Test worker status
    log "\n${YELLOW}🔍 Testing Worker Status${NC}"
    test_api "/api/debug/workers" "GET"
    
    # Get user tasks (requires authentication)
    log "\n${YELLOW}🔍 Testing User Tasks API${NC}"
    test_api "/api/user_tasks" "GET"
    
    # Get a random site number for testing
    SITE_NO=$(get_random_site_number)
    # Or use a fixed site for more consistent testing
    # SITE_NO="04144500"
    
    # Test launching a task with the random site number and correct resolutions
    log "\n${YELLOW}🔍 Testing Task Submission with Site $SITE_NO${NC}"
    log "Using LS Resolution: $LS_RESOLUTION, DEM Resolution: $DEM_RESOLUTION"
    
    # Create properly formatted JSON data for the model-settings request
    # Important: Make sure this is VALID JSON!
    MODEL_DATA="{\"username\":\"$USERNAME\",\"site_no\":\"$SITE_NO\",\"ls_resolution\":$LS_RESOLUTION,\"dem_resolution\":$DEM_RESOLUTION}"
    
    log "Task data: $MODEL_DATA"
    
    # Check if email debugging is enabled
    if [ "$DEBUG_EMAIL" == "true" ]; then
        # First, check the email settings 
        log "\n${YELLOW}🔍 Checking Email Configuration${NC}"
        test_api "/api/debug/email-check" "GET" || log "Email check endpoint not available"
    fi
    
    # Submit the task with enhanced debugging for email issues
    log "\n${YELLOW}🔍 Submitting Model Creation Task${NC}"
    if test_api "/api/model-settings" "POST" "$MODEL_DATA"; then
        # Extract task ID from response
        task_id=$(echo "$RESPONSE_BODY" | jq -r '.task_id')
        log "Task ID: $task_id"
        
        # Wait 5 seconds then check task status
        log "Waiting 5 seconds for task to start processing..."
        sleep 5
        
        # Check task status
        log "\n${YELLOW}🔍 Checking Task Status${NC}"
        test_api "/api/task_status/$task_id" "GET"
        
        # If email testing is enabled, check for email logs
        if [ "$DEBUG_EMAIL" == "true" ]; then
            log "\n${YELLOW}🔍 Checking Email Logs${NC}"
            test_api "/api/debug/email-logs" "GET" || log "Email logs endpoint not available"
        fi
    else
        log "${RED}Failed to submit test task${NC}"
        
        # Try to get more information about the failure
        log "\n${YELLOW}🔍 Getting Detailed Error Info${NC}"
        test_api "/api/debug/request" "GET"
    fi
    
    # Clean up cookies file
    if [ -f "cookies.txt" ]; then
        rm cookies.txt
        log "Removed cookies file"
    fi
    
    log "\n${GREEN}All tests completed!${NC}"
    log "See $LOG_FILE for detailed results"
}

# Run the main function
main "$@"
