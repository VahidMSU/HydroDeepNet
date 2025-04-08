#!/bin/bash
# SWATGenX API Authentication and Usage Examples
# This script demonstrates how to authenticate with the SWATGenX API
# and make subsequent API calls using the authentication token.

# Configuration - Change these values as needed
HOST="https://ciwre-bae.campusad.msu.edu" # Changed from 3000 to 5050
USERNAME="admin"
PASSWORD="admin@admin"
OUTPUT_FORMAT="pretty" # Options: pretty, compact

# Set to 1 to enable debug output
DEBUG=0

# Function to format JSON output
format_json() {
    if [ "$OUTPUT_FORMAT" = "pretty" ]; then
        python -m json.tool 2>/dev/null || cat
    else
        cat
    fi
}

# Function to debug output
debug() {
    if [ $DEBUG -eq 1 ]; then
        echo -e "\033[36m[DEBUG] $1\033[0m"
    fi
}

# Function to check connection
check_connection() {
    debug "Checking connection to $HOST..."
    curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$HOST" >/dev/null
    if [ $? -ne 0 ]; then
        echo -e "\033[31mERROR: Could not connect to $HOST\033[0m"
        echo "Please check that the server is running and the HOST variable is correct."
        echo "You can set a custom host with: HOST=http://your-host:port $0"
        exit 1
    fi
}

# Try different API endpoint formats
try_login_endpoints() {
    local auth_paths=("/api/login" "/login" "/api/auth/login")
    local success=0

    for path in "${auth_paths[@]}"; do
        debug "Trying login endpoint: $HOST$path"
        local response=$(curl -s -X POST "$HOST$path" \
            -H "Content-Type: application/json" \
            -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")

        debug "Response from $path: $response"

        # Check if response indicates success
        if echo "$response" | grep -q "success\|token\|user_id\|username"; then
            echo "Successfully authenticated using endpoint: $path"
            echo "$response"
            AUTH_RESPONSE="$response"
            LOGIN_ENDPOINT="$path"
            success=1
            break
        fi
    done

    return $success
}

# Allow overriding the host from command line
if [ ! -z "$1" ]; then
    HOST="$1"
    echo "Using custom host: $HOST"
fi

# Override HOST if set in environment
if [ ! -z "$HOST_ENV" ]; then
    HOST="$HOST_ENV"
    echo "Using environment-specified host: $HOST"
fi

echo "=== SWATGenX API Test Script ==="
echo "Host: $HOST"
echo "Username: $USERNAME"
echo ""

# Check connection before proceeding
check_connection

# 1. Login to get authentication token
echo "=== 1. Authenticating user ==="
debug "Trying multiple authentication endpoints..."

if try_login_endpoints; then
    echo "Authentication successful!"

    # Extract important information
    USER_ID=$(echo "$AUTH_RESPONSE" | grep -o '"id":[^,}]*' | cut -d ':' -f2)
    USERNAME=$(echo "$AUTH_RESPONSE" | grep -o '"username":"[^"]*"' | cut -d '"' -f4)
    IS_VERIFIED=$(echo "$AUTH_RESPONSE" | grep -o '"is_verified":[^,}]*' | cut -d ':' -f2)

    echo "User ID: $USER_ID"
    echo "Username: $USERNAME"
    echo "Verified: $IS_VERIFIED"

    # Store auth cookie if needed for subsequent requests
    AUTH_COOKIE=$(curl -s -i -X POST "$HOST$LOGIN_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" |
        grep -i "set-cookie" | cut -d':' -f2-)

    debug "Auth Cookie: $AUTH_COOKIE"

    # 2. Session validation
    echo ""
    echo "=== 2. Validating session ==="

    VALIDATE_RESPONSE=$(curl -s -X GET "$HOST/api/validate-session" \
        -H "Cookie: $AUTH_COOKIE" \
        -H "Content-Type: application/json")

    echo "Session validation response:"
    echo "$VALIDATE_RESPONSE" | format_json

    # 3. Example: Getting user tasks
    echo ""
    echo "=== 3. Getting user tasks ==="

    TASKS_RESPONSE=$(curl -s -X GET "$HOST/api/user_tasks" \
        -H "Cookie: $AUTH_COOKIE" \
        -H "Content-Type: application/json")

    echo "User tasks response:"
    echo "$TASKS_RESPONSE" | format_json

    # 4. Example: Search for USGS stations
    echo ""
    echo "=== 4. Searching for stations ==="
    SEARCH_TERM="06853"

    STATION_RESPONSE=$(curl -s -X GET "$HOST/api/search_site?search_term=$SEARCH_TERM" \
        -H "Cookie: $AUTH_COOKIE" \
        -H "Content-Type: application/json")

    echo "Station search response for term '$SEARCH_TERM':"
    echo "$STATION_RESPONSE" | format_json

    # 5. Logout
    echo ""
    echo "=== 5. Logging out ==="

    LOGOUT_RESPONSE=$(curl -s -X POST "$HOST/api/logout" \
        -H "Cookie: $AUTH_COOKIE" \
        -H "Content-Type: application/json")

    echo "Logout response:"
    echo "$LOGOUT_RESPONSE" | format_json

else
    echo "Authentication failed:"

    # Show details for the last attempted endpoint (default /api/login)
    debug "Showing detailed debug information for default endpoint /api/login"
    AUTH_RESPONSE=$(curl -s -X POST "$HOST/api/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")

    echo "$AUTH_RESPONSE" | format_json

    # Display more detailed error information
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/api/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")

    echo "HTTP Status Code: $HTTP_STATUS"

    # Check if the server is running Flask directly or behind Apache/Nginx
    debug "Checking for common backend paths..."
    for path in "/" "/api" "/login" "/api/v1/login"; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HOST$path")
        debug "Path $HOST$path returned: $STATUS"
    done

    if [ $DEBUG -eq 1 ]; then
        echo "Full response headers and body:"
        curl -v -X POST "$HOST/api/login" \
            -H "Content-Type: application/json" \
            -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}"
    fi

    echo ""
    echo "Troubleshooting tips:"
    echo "1. Make sure the API server is running at $HOST"
    echo "2. Check that the username and password are correct"
    echo "3. Try running with DEBUG=1 for more information: DEBUG=1 $0"
    echo "4. To use a different server: $0 http://different-host:port"
    echo "5. Verify Apache/Nginx is correctly proxying to the Flask application"
    echo "6. Check the Flask application logs: tail -f /data/SWATGenXApp/codes/web_application/logs/flask-app.log"
    echo "7. Verify the API endpoint exists by checking the Flask routes: curl $HOST/api/debug/routes"

    exit 1
fi

echo ""
echo "=== Usage Notes ==="
echo "1. Replace the HOST, USERNAME, and PASSWORD variables with your own values"
echo "2. To use this API in your own scripts, extract the session cookie from the login response"
echo "3. Include the session cookie with all subsequent requests in the Cookie header"
echo "4. For a complete list of API endpoints, visit $HOST/api/debug/routes"
echo "5. Set DEBUG=1 to enable detailed debug output"
echo "6. To use a different server: $0 http://different-host:port"
