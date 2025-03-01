#!/bin/bash
set -e  # Exit on error

echo "===== SWAT GenX App Full Reset ====="

# Go to the scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || { echo "Cannot navigate to script directory"; exit 1; }

echo "1. Resetting ports..."
sh ./reset_ports.sh

echo "2. Resetting backend services..."
sh ./reset_backend.sh

echo "3. Application successfully reset!"
echo ""
echo "To start the web application, run: sh setup_webapp.sh"
