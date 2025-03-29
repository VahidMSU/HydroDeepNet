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

    VPUID = find_VPUID(site_no)
    LEVEL = 'huc12'

    logger.info(f"Creating model for user {username}/{VPUID}/{LEVEL}/{site_no}")
    
    try:
        # Direct function call (runs as celery worker user)
        single_swatplus_model_creation(username, site_no, ls_resolution, dem_resolution)

        logger.info(f"SWATGenX has finished processing for user {username}, site {VPUID}/{LEVEL}/{site_no}")


        RESOLUTION = ls_resolution  ### 250
        MODEL_NAME = f'MODFLOW_{RESOLUTION}m'
        SWAT_MODEL_NAME = 'SWAT_MODEL_Web_Application'
        
        # Import the correct function from MODGenX_API
        
        if MODFLOW_coverage(site_no):
            try:
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
            except Exception as modflow_error:
                logger.error(f"Error creating MODFLOW model: {str(modflow_error)}")

        
        # Prepare model info for email
        model_info = {
            "Site Number": site_no,
            "LS Resolution": ls_resolution,
            "DEM Resolution": dem_resolution,
            "Creation Date": self.request.id  # Using task ID as a unique identifier
        }
        
        # Find user's email
        try:
            # Since we're in a Celery task, we need to create application context
            # Import what we need for this section
            from flask import Flask
            from app import create_app
            
            app = create_app()
            with app.app_context():
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
                    else:
                        logger.warning(f"Failed to send model completion email to {user.email}")
                else:
                    logger.error(f"Could not find email for user {username}")
        except Exception as email_error:
            logger.error(f"Error sending model completion email: {str(email_error)}")
            logger.error(traceback.format_exc())
            # We don't raise this exception since the model was created successfully
        
        return {"status": "success", "message": "Model created successfully"}
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise