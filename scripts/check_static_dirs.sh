#!/bin/bash

# Set directory paths (make compatible with sh by avoiding Bash array syntax)
STATIC_DIR_1="/data/SWATGenXApp/GenXAppData/images"
STATIC_DIR_2="/data/SWATGenXApp/GenXAppData/videos"
STATIC_DIR_3="/data/SWATGenXApp/codes/web_application/logs"

echo "Checking static directories..."

# Check and create directories if needed
for DIR in "$STATIC_DIR_1" "$STATIC_DIR_2" "$STATIC_DIR_3"; do
  if [ -d "$DIR" ]; then
    echo "✅ Directory exists: $DIR"
  else
    echo "❌ Creating missing directory: $DIR"
    mkdir -p "$DIR"
    sudo chown -R www-data:www-data "$DIR"
    echo "   Directory created and permissions set"
  fi
done

echo "Static directory check completed."
