import logging
import os
import sys
from celery import shared_task
from app.utils import single_model_creation, LoggerSetup

# Set up logging
log_dir = "/data/SWATGenXApp/codes/web_application/logs"
logger = LoggerSetup(log_dir, rewrite=False).setup_logger("CeleryTasks")

# Log system information for debugging
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {sys.path}")

# Log Celery configuration for debugging
try:
    from celery_app import celery
    broker_url = celery.conf.broker_url
    backend_url = celery.conf.result_backend
    logger.info(f"Celery imported successfully! Broker URL: {broker_url}")
    logger.info(f"Celery result backend URL: {backend_url}")
except Exception as e:
    logger.error(f"Error accessing Celery configuration: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")

@shared_task(bind=True, 
             autoretry_for=(Exception,), 
             retry_backoff=True, 
             retry_kwargs={'max_retries': 3})
def create_model_task(self, username, site_no, ls_resolution, dem_resolution):
    """
    Task to create a SWAT model
    """
    logger.info("AppManager initialized!")
    logger.info(f"Creating model for user {username}, site {site_no}")
    
    try:
        # Direct function call (runs as celery worker user)
        single_model_creation(username, site_no, ls_resolution, dem_resolution)
        logger.info(f"Model created successfully for user {username}, site {site_no}")
        return {"status": "success", "message": "Model created successfully"}
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise