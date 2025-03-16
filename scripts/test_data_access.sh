#!/usr/bin/env bash

LOG_FILE="/data/SWATGenXApp/codes/web_application/logs/data_access_test.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting data access test..."

# Define the directories to check
DATA_DIRS=(
  "/data/SWATGenXApp/GenXAppData"
  "/data/SWATGenXApp/Users"
  "/data/SWATGenXApp/codes/SWATGenX"
  "/data/SWATGenXApp/codes/AI_agent"
)

# Test as the current user
log "Testing as current user: $(whoami)"
for dir in "${DATA_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    log "✅ Directory exists: $dir"
    
    # Check if we can list contents
    if ls -la "$dir" > /dev/null 2>&1; then
      log "✅ Can list contents of: $dir"
      
      # Try to access a file to read
      FILE_COUNT=$(find "$dir" -type f -name "*.py" -o -name "*.json" -o -name "*.csv" | head -n 1)
      if [ -n "$FILE_COUNT" ]; then
        TEST_FILE=$(find "$dir" -type f -name "*.py" -o -name "*.json" -o -name "*.csv" | head -n 1)
        if [ -r "$TEST_FILE" ]; then
          log "✅ Can read file: $TEST_FILE"
        else
          log "❌ Cannot read file: $TEST_FILE"
        fi
      else
        log "⚠️ No suitable test files found in $dir"
      fi
    else
      log "❌ Cannot list contents of: $dir"
    fi
  else
    log "❌ Directory does not exist: $dir"
  fi
done

# Check SELinux status if available
if command -v getenforce &> /dev/null; then
  SELINUX_STATUS=$(getenforce)
  log "SELinux status: $SELINUX_STATUS"
  
  if [ "$SELINUX_STATUS" = "Enforcing" ]; then
    log "⚠️ SELinux is enforcing - this may affect file permissions"
    log "Checking SELinux context on data directories..."
    
    for dir in "${DATA_DIRS[@]}"; do
      if [ -d "$dir" ]; then
        CONTEXT=$(ls -Z "$dir" | head -n 1)
        log "SELinux context for $dir: $CONTEXT"
      fi
    done
  fi
fi

# Test write operations if needed
for dir in "/data/SWATGenXApp/Users" "/data/SWATGenXApp/codes/web_application/logs"; do
  if [ -d "$dir" ]; then
    TEST_FILE="$dir/test_write_$(date +%s).txt"
    if touch "$TEST_FILE" 2>/dev/null; then
      log "✅ Can write to directory: $dir"
      # Clean up test file
      rm "$TEST_FILE"
    else
      log "❌ Cannot write to directory: $dir"
    fi
  fi
done

# Test permissions as www-data (Apache user)
if command -v sudo &> /dev/null; then
  log "Testing as www-data (Apache user)"
  for dir in "${DATA_DIRS[@]}"; do
    if sudo -u www-data test -d "$dir"; then
      log "✅ www-data can access directory: $dir"
      
      # Check if www-data can read files in this directory
      if sudo -u www-data ls "$dir" > /dev/null 2>&1; then
        log "✅ www-data can list contents of: $dir"
      else
        log "❌ www-data cannot list contents of: $dir"
      fi
    else
      log "❌ www-data cannot access directory: $dir"
    fi
  done
else
  log "⚠️ sudo not available, skipping www-data permission tests"
fi

log "Data access test completed. Check $LOG_FILE for detailed results."
