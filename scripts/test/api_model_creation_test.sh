#!/bin/bash
# SWATGenX API Model Creation Test

# Configuration
HOST=${1:-"http://localhost:5050"} # Changed default from 3000 to 5050
USERNAME=${2:-"admin"}
PASSWORD=${3:-"admin"}
SITE_NUMBER=${4:-"05536265"}
LS_RESOLUTION=${5:-"250"}
DEM_RESOLUTION=${6:-"30"}
AUTH_ENDPOINT="/api/auth/login"
MODEL_ENDPOINT="/api/models"
TIMEOUT=10

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Header
echo -e "=== SWATGenX Model Creation Test ==="
echo -e "Host: $HOST"
echo -e "Username: $USERNAME"
echo -e "Site Number: $SITE_NUMBER"
echo -e "LS Resolution: $LS_RESOLUTION"
echo -e "DEM Resolution: $DEM_RESOLUTION"
echo -e ""

# Function for error handling
handle_error() {
    local status_code=$1
    local response=$2

    echo -e "${RED}Error: HTTP Status $status_code${NC}"

    if [[ ! -z "$response" ]]; then
        echo -e "Response: $response"
    else
        echo -e "No response body received"
    fi

    if [[ $status_code -eq 0 ]]; then
        echo -e "${YELLOW}Connection failed. Is the server running at $HOST?${NC}"
    elif [[ $status_code -eq 401 || $status_code -eq 403 ]]; then
        echo -e "${YELLOW}Authentication failed. Check your credentials.${NC}"
    elif [[ $status_code -eq 404 ]]; then
        echo -e "${YELLOW}API endpoint not found. Check if the API path is correct.${NC}"
    elif [[ $status_code -ge 500 ]]; then
        echo -e "${YELLOW}Server error. Check the server logs.${NC}"
    fi
}

# 1. Authentication
echo -e "=== 1. Authenticating user ==="

# Check if the server is reachable before attempting authentication
if ! curl -s --head --fail --max-time 5 "$HOST" >/dev/null; then
    echo -e "${RED}Cannot connect to server at $HOST${NC}"
    echo -e "${YELLOW}Make sure the server is running and the URL is correct${NC}"
    echo -e "${YELLOW}Hint: The API might be running on port 5050 instead of 3000${NC}"
    echo -e "\n=== Test Complete ==="
    exit 1
fi

# Attempt authentication
AUTH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" \
    --max-time $TIMEOUT \
    "$HOST$AUTH_ENDPOINT")

# Extract status code and response body
AUTH_STATUS=$(echo "$AUTH_RESPONSE" | tail -n1)
AUTH_BODY=$(echo "$AUTH_RESPONSE" | sed '$d')

if [[ $AUTH_STATUS -ne 200 ]]; then
    handle_error $AUTH_STATUS "$AUTH_BODY"
    echo -e "Authentication failed. Cannot proceed with model creation."
    echo -e "\n=== Test Complete ==="
    exit 1
fi

# Extract token from response
TOKEN=$(echo $AUTH_BODY | grep -o '"token":"[^"]*' | cut -d'"' -f4)

if [[ -z "$TOKEN" ]]; then
    echo -e "${RED}Failed to extract authentication token from response!${NC}"
    echo -e "Response: $AUTH_BODY"
    echo -e "\n=== Test Complete ==="
    exit 1
fi

echo -e "${GREEN}Authentication successful!${NC}"

# 2. Create model
echo -e "\n=== 2. Creating model with Site $SITE_NUMBER ==="

MODEL_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"site_number\":\"$SITE_NUMBER\",\"ls_resolution\":$LS_RESOLUTION,\"dem_resolution\":$DEM_RESOLUTION}" \
    --max-time $TIMEOUT \
    "$HOST$MODEL_ENDPOINT")

# Extract status code and response body
MODEL_STATUS=$(echo "$MODEL_RESPONSE" | tail -n1)
MODEL_BODY=$(echo "$MODEL_RESPONSE" | sed '$d')

if [[ $MODEL_STATUS -ne 200 && $MODEL_STATUS -ne 201 ]]; then
    handle_error $MODEL_STATUS "$MODEL_BODY"
    echo -e "Model creation failed."
    echo -e "\n=== Test Complete ==="
    exit 1
fi

echo -e "${GREEN}Model creation initiated successfully!${NC}"
echo -e "Response: $MODEL_BODY"

# Extract task_id if available
TASK_ID=$(echo $MODEL_BODY | grep -o '"task_id":"[^"]*' | cut -d'"' -f4)

if [[ ! -z "$TASK_ID" ]]; then
    echo -e "Task ID: $TASK_ID"
    echo -e "\nYou can check the status of this task using:"
    echo -e "curl -X GET -H \"Authorization: Bearer $TOKEN\" $HOST/api/tasks/$TASK_ID"
fi

echo -e "\n=== Test Complete ==="
