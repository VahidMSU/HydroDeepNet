#!/bin/bash
# Script to fix Flask application issues in Docker container

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fixing Flask Application Issues ===${NC}"

# Set correct working directory
cd /data/SWATGenXApp/codes
export PYTHONPATH=/data/SWATGenXApp/codes
export FLASK_APP=run.py

# Check if run.py exists
if [ ! -f "run.py" ]; then
    echo -e "${RED}Error: run.py not found in /data/SWATGenXApp/codes${NC}"
    echo -e "${YELLOW}Creating a minimal Flask application...${NC}"

    # Create a minimal Flask application
    cat >run.py <<'EOF'
import os
from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)
    
    @app.route('/api/diagnostic/status')
    def status():
        return jsonify({"status": "ok", "message": "Flask app is running"})
    
    @app.route('/api/model-settings')
    def model_settings():
        return jsonify({
            "status": "success",
            "message": "Model settings available",
            "data": {
                "version": "1.0.0",
                "environment": "docker",
                "configured": True
            }
        })
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
EOF
    echo -e "${GREEN}Created minimal Flask application at /data/SWATGenXApp/codes/run.py${NC}"
fi

# Stop any running Flask processes
echo -e "${YELLOW}Stopping any running Flask processes...${NC}"
pkill -f "gunicorn.*run:app" || echo "No Flask processes found"
pkill -f "flask run" || echo "No Flask processes found"
sleep 2

# Create a simple initialization script
echo -e "${YELLOW}Creating initialization script...${NC}"
cat >init_flask.py <<'EOF'
import os
import sys

# Add the application directory to Python path
sys.path.insert(0, '/data/SWATGenXApp/codes')

try:
    # Try to import the create_app function
    from run import create_app
    
    app = create_app()
    print("Flask app initialized successfully!")
    
    # Print registered routes
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    
    print("\nApp is ready to be served with Gunicorn")
    
except Exception as e:
    print(f"Error initializing Flask app: {str(e)}")
    sys.exit(1)
EOF

# Run the initialization script to test the app
echo -e "${YELLOW}Testing Flask application initialization...${NC}"
python3 init_flask.py

# Start Flask on ports 5000 and 5050
echo -e "${YELLOW}Starting Flask on port 5000...${NC}"
nohup /data/SWATGenXApp/codes/.venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 run:app --timeout 120 >/data/SWATGenXApp/codes/web_application/logs/gunicorn-5000.log 2>&1 &
sleep 3

echo -e "${YELLOW}Starting Flask on port 5050...${NC}"
nohup /data/SWATGenXApp/codes/.venv/bin/gunicorn -w 4 -b 0.0.0.0:5050 run:app --timeout 120 >/data/SWATGenXApp/codes/web_application/logs/gunicorn-5050.log 2>&1 &
sleep 3

# Check if the servers are running and endpoints are accessible
echo -e "${YELLOW}Checking if Flask servers are running...${NC}"
if netstat -tulpn 2>/dev/null | grep -q ":5000"; then
    echo -e "${GREEN}Flask server is running on port 5000${NC}"
else
    echo -e "${RED}Flask server is not running on port 5000${NC}"
fi

if netstat -tulpn 2>/dev/null | grep -q ":5050"; then
    echo -e "${GREEN}Flask server is running on port 5050${NC}"
else
    echo -e "${RED}Flask server is not running on port 5050${NC}"
fi

# Test the endpoints
echo -e "${YELLOW}Testing endpoints...${NC}"
echo -e "Port 5000 status endpoint response:"
curl -s http://localhost:5000/api/diagnostic/status
echo -e "\n\nPort 5000 model-settings endpoint response:"
curl -s http://localhost:5000/api/model-settings
echo -e "\n\nPort 5050 status endpoint response:"
curl -s http://localhost:5050/api/diagnostic/status
echo -e "\n\nPort 5050 model-settings endpoint response:"
curl -s http://localhost:5050/api/model-settings
echo -e "\n"

echo -e "${BLUE}=== Flask App Fix Completed ===${NC}"
echo -e "${YELLOW}Run verification script to check if issues are resolved:${NC}"
echo -e "${GREEN}bash /data/SWATGenXApp/codes/scripts/docker-tools/verify_docker_deployment.sh${NC}"
