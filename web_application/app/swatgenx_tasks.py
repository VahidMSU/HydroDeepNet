import logging
import os
import sys
import traceback
from celery import shared_task
from app.utils import single_swatplus_model_creation, LoggerSetup
from app.comm_utils import send_model_completion_email
from app.model_utils import MODFLOW_coverage
from app.models import User
from flask import current_app
from app.extensions import db
from app.utils import find_VPUID
from MODFLOW.MODGenX_API import create_modflow_model
import time
from redis import Redis

# Set up logging
log_dir = "/data/SWATGenXApp/codes/web_application/logs"
logger = LoggerSetup(log_dir, rewrite=False).setup_logger("CeleryTasks")

# Log system information for debugging
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {sys.path}")

# Test Redis connection directly
def test_redis_connection():
    """Test Redis connection directly, with multiple attempts."""
    max_retries = 5
    retry_delay = 1
    redis_urls = [
        os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
        'redis://127.0.0.1:6379/0',
        'redis://localhost:6379/0',
        'redis://redis:6379/0'
    ]
    
    for attempt in range(max_retries):
        for url in redis_urls:
            try:
                logger.info(f"Testing Redis connection at {url} (attempt {attempt+1})")
                client = Redis.from_url(url, socket_timeout=5, socket_connect_timeout=5)
                ping_result = client.ping()
                logger.info(f"Redis ping successful at {url}: {ping_result}")
                
                # Update environment variable with working URL
                if url != os.environ.get('REDIS_URL'):
                    os.environ['REDIS_URL'] = url
                    logger.info(f"Updated REDIS_URL to {url}")
                
                return True, url
            except Exception as e:
                logger.warning(f"Redis connection failed at {url}: {e}")
                
        if attempt < max_retries - 1:
            logger.info(f"Retrying Redis connection in {retry_delay} seconds")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 10)
    
    logger.error("All Redis connection attempts failed")
    return False, None

# Test Redis connection on module import
redis_ok, working_url = test_redis_connection()
if redis_ok:
    logger.info(f"Redis connection test successful at {working_url}")
else:
    logger.error("Redis connection test failed")

# Log Celery configuration for debugging
try:
    from celery_app import celery
    broker_url = celery.conf.broker_url
    backend_url = celery.conf.result_backend
    logger.info(f"Celery imported successfully! Broker URL: {broker_url}")
    logger.info(f"Celery result backend URL: {backend_url}")
    logger.info(f"Broker connection retry on startup: {celery.conf.broker_connection_retry_on_startup}")
    logger.info(f"Broker connection retry: {celery.conf.broker_connection_retry}")
except Exception as e:
    logger.error(f"Error accessing Celery configuration: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

@shared_task(bind=True, 
             autoretry_for=(Exception,), 
             retry_backoff=True, 
             retry_kwargs={'max_retries': 3},
             queue='model_creation',
             acks_late=True,    # Acknowledge task after it's completed to prevent loss of tasks
             reject_on_worker_lost=True)  # Requeue task if worker unexpectedly dies
def create_model_task(self, username, site_no, ls_resolution, dem_resolution):
    """
    Task to create a SWAT model
    """
    # Import the task tracker within the task to ensure it's available in the Celery worker context
    from app.task_tracker import task_tracker
    
    # Register the task in the tracker with initial metadata
    task_metadata = {
        "ls_resolution": ls_resolution,
        "dem_resolution": dem_resolution,
        "start_time": time.time()
    }
    task_tracker.register_task(self.request.id, username, site_no, task_metadata)
    
    logger.info(f"Starting model creation task for user {username}, site {site_no}")
    task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=5,
                               info={"message": f"Starting model creation for site {site_no}"})
    
    logger.info("Verifying Redis connection before proceeding...")
    redis_ok, _ = test_redis_connection()
    if not redis_ok:
        logger.error("Cannot proceed with task - Redis connection failed")
        self.update_state(state='FAILURE', meta={'error': 'Redis connection failed'})
        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_FAILURE, 
                                   info={"error": "Redis connection failed"})
        raise Exception("Redis connection failed")

    VPUID = find_VPUID(site_no)
    LEVEL = 'huc12'

    logger.info(f"Creating model for user {username}/{VPUID}/{LEVEL}/{site_no}")
    task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=10,
                               info={"message": f"Beginning SWAT model creation for watershed {VPUID}/{LEVEL}/{site_no}"})
    
    try:
        # Direct function call (runs as celery worker user)
        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=15,
                                   info={"message": "Starting SWAT+ model creation"})
        
        single_swatplus_model_creation(username, site_no, ls_resolution, dem_resolution)

        logger.info(f"SWATGenX has finished processing for user {username}, site {VPUID}/{LEVEL}/{site_no}")
        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=60,
                                   info={"message": "SWAT+ model creation completed, preparing MODFLOW model"})

        RESOLUTION = ls_resolution  ### 250
        MODEL_NAME = f'MODFLOW_{RESOLUTION}m'
        SWAT_MODEL_NAME = 'SWAT_MODEL_Web_Application'
        
        # Import the correct function from MODGenX_API
        
        if MODFLOW_coverage(site_no):
            try:
                task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=65,
                                          info={"message": "Starting MODFLOW model creation"})
                
                # Call create_modflow_model with all required parameters
                create_modflow_model(
                    username=username,
                    NAME=site_no, 
                    VPUID=VPUID,
                    LEVEL=LEVEL,
                    RESOLUTION=RESOLUTION,
                    MODEL_NAME=MODEL_NAME,
                    ML=False,  # Set default value for ML parameter
                    SWAT_MODEL_NAME=SWAT_MODEL_NAME
                )
                logger.info(f"MODGenX has finished processing for user {username}, site {VPUID}/{LEVEL}/{site_no}")
                task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=90,
                                          info={"message": "MODFLOW model creation completed"})
            except Exception as modflow_error:
                logger.error(f"Error creating MODFLOW model: {str(modflow_error)}")
                logger.error(traceback.format_exc())
                task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=75,
                                          info={"warning": f"MODFLOW model creation failed: {str(modflow_error)}"})
        
        # Prepare model info for email
        model_info = {
            "Site Number": site_no,
            "LS Resolution": ls_resolution,
            "DEM Resolution": dem_resolution,
            "Creation Date": self.request.id  # Using task ID as a unique identifier
        }
        
        # Calculate execution time
        execution_time = time.time() - task_metadata.get("start_time", time.time())
        execution_time_str = f"{execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        
        # Find user's email
        try:
            # Since we're in a Celery task, we need to create application context
            # Import what we need for this section
            from flask import Flask
            from app import create_app
            
            app = create_app()
            with app.app_context():
                task_tracker.update_task_status(self.request.id, task_tracker.STATUS_STARTED, progress=95,
                                           info={"message": "Model creation completed, sending email notification"})
                
                user = User.query.filter_by(username=username).first()
                if user and user.email:
                    # Send email notification
                    email_sent = send_model_completion_email(
                        username, 
                        user.email, 
                        site_no, 
                        model_info
                    )
                    if email_sent:
                        logger.info(f"Model completion email sent to {user.email}")
                        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_SUCCESS, progress=100,
                                                   info={"message": "Model created successfully",
                                                       "execution_time": execution_time_str,
                                                       "email_sent": True})
                    else:
                        logger.warning(f"Failed to send model completion email to {user.email}")
                        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_SUCCESS, progress=100,
                                                   info={"message": "Model created successfully",
                                                       "execution_time": execution_time_str,
                                                       "email_sent": False,
                                                       "warning": "Failed to send completion email"})
                else:
                    logger.error(f"Could not find email for user {username}")
                    task_tracker.update_task_status(self.request.id, task_tracker.STATUS_SUCCESS, progress=100,
                                               info={"message": "Model created successfully",
                                                   "execution_time": execution_time_str,
                                                   "email_sent": False,
                                                   "warning": "User email not found"})
        except Exception as email_error:
            logger.error(f"Error sending model completion email: {str(email_error)}")
            logger.error(traceback.format_exc())
            task_tracker.update_task_status(self.request.id, task_tracker.STATUS_SUCCESS, progress=100,
                                       info={"message": "Model created successfully",
                                           "execution_time": execution_time_str,
                                           "email_sent": False,
                                           "error": f"Email error: {str(email_error)}"})
            # We don't raise this exception since the model was created successfully
        
        return {"status": "success", "message": "Model created successfully"}
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        
        # Log failure in task tracker
        task_tracker.update_task_status(self.request.id, task_tracker.STATUS_FAILURE, 
                                   info={"error": str(e), "traceback": traceback.format_exc()})
        raise