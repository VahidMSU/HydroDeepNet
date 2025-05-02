# SWATGenX Celery Management Suite

This document provides a comprehensive overview of the consolidated Celery management tools for SWATGenX.

## Overview

The Celery Management Suite provides a unified set of tools for monitoring, maintaining, and troubleshooting the Celery task queue system used by SWATGenX. All tools can be accessed through the central `celery-tools` command.

## Installation

To install the unified management suite:

```bash
# Run the installation script
sudo /data/SWATGenXApp/codes/scripts/install_celery_tools.sh
```

This will:
1. Set up the directory structure
2. Make all scripts executable
3. Install the `celery-tools` command in your path
4. Install required systemd services

## Basic Usage

The `celery-tools` command provides a central interface to all Celery management functions:

```bash
# Get help
celery-tools help

# Monitor Celery status
celery-tools monitor

# Inspect tasks in queues
celery-tools inspect

# Fix Redis WRONGTYPE errors (requires sudo)
sudo celery-tools fix-redis-batch

# Restart Celery workers after fixing issues (requires sudo)
sudo celery-tools restart
```

## Command Reference

### Monitoring Commands

```bash
# Check overall status
celery-tools monitor

# Check specific components
celery-tools monitor --workers
celery-tools monitor --queues
celery-tools monitor --tasks
celery-tools monitor --resources

# Enable watch mode (refresh every 30 seconds)
celery-tools monitor -w 30

# Save report to file
celery-tools monitor -o /path/to/report.txt
```

### Inspection Commands

```bash
# List all queues and their sizes
celery-tools inspect --list

# Inspect tasks in a specific queue
celery-tools inspect --inspect celery

# Show detailed task information
celery-tools inspect --inspect celery --full

# Limit the number of tasks shown
celery-tools inspect --inspect celery --limit 50
```

### Maintenance Commands

```bash
# Fix WRONGTYPE Redis errors with batch processing (for large datasets)
sudo celery-tools fix-redis-batch

# Clean up corrupted Celery task messages
sudo celery-tools cleanup

# Restart Celery workers after fixing issues
sudo celery-tools restart

# Restart stalled tasks (tasks inactive for over 1 hour)
sudo celery-tools restart-tasks

# Clear stuck tasks
sudo celery-tools clear-stuck

# Emergency reset (use with caution)
sudo celery-tools reset
```

## Common Scenarios

### 1. Routine Monitoring

For daily monitoring of the Celery system:

```bash
# Check overall system status
celery-tools monitor

# Set up continuous monitoring (refresh every minute)
celery-tools monitor -w 60
```

### 2. Troubleshooting Worker Issues

When workers are not processing tasks:

```bash
# Check worker status
celery-tools monitor --workers

# Check queues for pending tasks
celery-tools monitor --queues

# Restart workers if needed
sudo celery-tools restart
```

### 3. Cleaning Up Corrupted Redis Data

When encountering Redis errors in logs:

```bash
# Fix Redis WRONGTYPE errors with batch processing
sudo celery-tools fix-redis-batch --dry-run
sudo celery-tools fix-redis-batch --batch-size 100
sudo systemctl start redis-fix.service
sudo journalctl -f -u redis-fix.service
```

### 4. Handling Stuck Tasks

When tasks are stuck and not completing:

```bash
# List all active tasks
sudo celery-tools clear-stuck --list

# Restart tasks that have been inactive for over 2 hours
sudo celery-tools restart-tasks --stale-hours 2
```

### 5. Emergency Recovery

In case of major system failure:

```bash
# Perform emergency reset (with confirmation prompt)
sudo celery-tools reset

# Verify system status after reset
celery-tools monitor
```

## Advanced Configuration

All commands accept additional options that can be passed through the `celery-tools` interface. For detailed help on specific commands:

```bash
celery-tools monitor --help
celery-tools inspect --help
sudo celery-tools fix-redis-batch --help
# etc.
```

## Directory Structure

The Celery management tools are organized as follows:

```
/data/SWATGenXApp/codes/scripts/
├── celery-tools                 # Main command-line interface
├── install_celery_tools.sh      # Installation script
├── setup_celery_services.sh     # Setup script
└── celery-tools-services/            
    ├── utils/                   # Python utilities
    ├── services/                # Service wrappers and systemd units
    └── docs/                    # Documentation
```

## Scheduled Maintenance

For automated maintenance, add these cron jobs:

```bash
# Monitor Celery status every 15 minutes
*/15 * * * * /usr/local/bin/celery-tools monitor --alert-threshold 100 --email admin@example.com --clean -o /data/SWATGenXApp/codes/web_application/logs/celery_monitor_$(date +\%Y\%m\%d).log

# Restart stalled tasks hourly
0 * * * * /usr/local/bin/celery-tools restart-tasks --stale-hours 2 --yes
```
