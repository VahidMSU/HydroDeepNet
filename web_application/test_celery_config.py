#!/usr/bin/env python3
"""
Test Celery Configuration

This script validates that Celery is properly configured with the
environment variables specified in config.py.
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

# Add application path for imports
sys.path.insert(0, '/data/SWATGenXApp/codes')
sys.path.insert(0, '/data/SWATGenXApp/codes/web_application')

def print_header(text):
    """Print a section header"""
    border = "=" * len(text)
    print(f"\n{border}\n{text}\n{border}")

def test_redis_connection():
    """Test direct Redis connectivity"""
    print_header("Testing Redis Connection")
    
    try:
        from redis import Redis
        
        # Collect possible Redis URLs
        redis_urls = [
            os.environ.get('REDIS_URL', ''),
            'redis://localhost:6379/0',
            'redis://127.0.0.1:6379/0',
            'redis://redis:6379/0'
        ]
        
        # Filter out empty URLs
        redis_urls = [url for url in redis_urls if url]
        
        print(f"Testing {len(redis_urls)} possible Redis URLs")
        
        for url in redis_urls:
            print(f"\nTrying {url}:")
            try:
                client = Redis.from_url(url, socket_timeout=2)
                start_time = time.time()
                ping_result = client.ping()
                ping_time = time.time() - start_time
                
                print(f"  Ping successful: {ping_result}")
                print(f"  Response time: {ping_time*1000:.2f}ms")
                
                # Test set/get
                test_key = f"celery_test_{time.time()}"
                client.set(test_key, "test_value", ex=10)
                result = client.get(test_key)
                client.delete(test_key)
                
                print(f"  Set/Get test: {'Passed' if result == b'test_value' else 'Failed'}")
                print(f"  ✅ Redis at {url} is working correctly")
                
                # If we got here, set this as the REDIS_URL
                os.environ['REDIS_URL'] = url
                print(f"  Set REDIS_URL environment variable to {url}")
                
                return True, url
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        print("\n❌ All Redis connection attempts failed")
        return False, None
    except ImportError:
        print("❌ Redis module not installed. Run: pip install redis")
        return False, None

def test_celery_import():
    """Test importing Celery and checking its configuration"""
    print_header("Testing Celery Configuration")
    
    try:
        # Add venv to path if needed
        venv_path = '/data/SWATGenXApp/codes/.venv/lib/python3.8/site-packages'
        if os.path.exists(venv_path) and venv_path not in sys.path:
            sys.path.append(venv_path)
            print(f"Added virtualenv path: {venv_path}")
        
        # Try to import celery
        print("Importing celery_app module...")
        try:
            from celery_app import celery
            print("✅ Successfully imported celery from celery_app")
        except ImportError:
            print("❌ Failed to import celery from celery_app")
            print("Trying alternate import path...")
            
            # Try importing from an alternate location
            os.chdir('/data/SWATGenXApp/codes/web_application')
            sys.path.insert(0, os.getcwd())
            print(f"Changed directory to: {os.getcwd()}")
            
            try:
                from celery_app import celery
                print("✅ Successfully imported celery from celery_app after directory change")
            except ImportError as e:
                print(f"❌ Failed to import celery: {e}")
                return False
        
        # Check celery configuration
        print("\nCelery Configuration:")
        config_items = [
            ('broker_url', celery.conf.broker_url),
            ('result_backend', celery.conf.result_backend),
            ('task_default_rate_limit', celery.conf.task_default_rate_limit),
            ('worker_prefetch_multiplier', celery.conf.worker_prefetch_multiplier),
            ('worker_disable_rate_limits', celery.conf.worker_disable_rate_limits),
            ('task_soft_time_limit', celery.conf.task_soft_time_limit),
            ('task_time_limit', celery.conf.task_time_limit)
        ]
        
        for key, value in config_items:
            env_var = f'CELERY_{key.upper()}'
            env_value = os.environ.get(env_var)
            print(f"  {key}: {value}")
            if env_value:
                print(f"    Environment: {env_var}={env_value}")
                
        # Test worker connection
        print("\nTesting connection to workers...")
        try:
            inspect = celery.control.inspect()
            stats = inspect.stats()
            
            if stats:
                print(f"✅ Connected to {len(stats)} workers:")
                for worker_name, worker_stats in stats.items():
                    print(f"  - {worker_name}")
            else:
                print("❌ No workers found (inspect.stats() returned None)")
                
            # Check active tasks
            active = inspect.active()
            if active:
                total_active = sum(len(tasks) for tasks in active.values())
                print(f"Total active tasks: {total_active}")
            else:
                print("No active tasks found")
                
        except Exception as e:
            print(f"❌ Error connecting to workers: {e}")
            
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_task_submission():
    """Test submitting a task and tracking its status"""
    print_header("Testing Task Submission")
    
    try:
        # Import the task
        from app.swatgenx_tasks import create_model_task
        print("✅ Successfully imported create_model_task")
        
        # Submit a test task
        print("\nSubmitting test task...")
        task = create_model_task.apply_async(
            args=["test_user", "06853800", 250, 30],
            queue='model_creation'
        )
        
        print(f"Task ID: {task.id}")
        print(f"Task state: {task.state}")
        
        # Monitor task for a few seconds
        print("\nMonitoring task status for 10 seconds:")
        for i in range(10):
            status = task.status
            print(f"[{i+1}s] Status: {status}")
            if status in ['SUCCESS', 'FAILURE']:
                break
            time.sleep(1)
            
        # Final status check
        print(f"\nFinal task state: {task.state}")
        if task.successful():
            print("✅ Task completed successfully")
        elif task.failed():
            print(f"❌ Task failed: {task.result}")
        else:
            print(f"Task is still running (status: {task.status})")
            
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Celery Configuration")
    parser.add_argument("--redis-only", action="store_true", help="Only test Redis connection")
    parser.add_argument("--celery-only", action="store_true", help="Only test Celery configuration")
    parser.add_argument("--task-only", action="store_true", help="Only test task submission")
    parser.add_argument("--set-env", action="store_true", help="Set environment variables from config.py")
    args = parser.parse_args()
    
    print(f"\n==== Celery Configuration Test ====")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Set environment variables if requested
    if args.set_env:
        print_header("Setting Environment Variables")
        # These match the config.py variables
        os.environ['CELERY_WORKER_COUNT'] = '30'
        os.environ['CELERY_WORKER_CONCURRENCY'] = '8'
        os.environ['CELERY_MAX_TASKS_PER_CHILD'] = '100'
        os.environ['CELERY_MAX_MEMORY_PER_CHILD_MB'] = '8192'
        os.environ['CELERY_WORKER_PREFETCH_MULTIPLIER'] = '8'
        os.environ['CELERY_TASK_SOFT_TIME_LIMIT'] = '43200'
        os.environ['CELERY_TASK_TIME_LIMIT'] = '86400'
        os.environ['CELERY_DISABLE_RATE_LIMITS'] = 'true'
        os.environ['CELERY_BROKER_CONNECTION_RETRY'] = 'true'
        os.environ['CELERY_BROKER_CONNECTION_MAX_RETRIES'] = '20'
        os.environ['CELERY_REDIS_MAX_CONNECTIONS'] = '500'
        os.environ['CELERY_MODEL_CREATION_WORKER_PERCENT'] = '70'
        print("Environment variables set for testing")
    
    # Run tests based on arguments
    if args.redis_only or not (args.celery_only or args.task_only):
        redis_ok, redis_url = test_redis_connection()
    
    if args.celery_only or not (args.redis_only or args.task_only):
        celery_ok = test_celery_import()
    
    if args.task_only or not (args.redis_only or args.celery_only):
        task_ok = test_task_submission()
    
    print("\n==== Test Summary ====")
    if 'redis_ok' in locals():
        print(f"Redis Connection: {'✅ PASSED' if redis_ok else '❌ FAILED'}")
    if 'celery_ok' in locals():
        print(f"Celery Configuration: {'✅ PASSED' if celery_ok else '❌ FAILED'}")
    if 'task_ok' in locals():
        print(f"Task Submission: {'✅ PASSED' if task_ok else '❌ FAILED'}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
