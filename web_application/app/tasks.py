import logging
from celery import shared_task
from app.utils import single_model_creation, LoggerSetup

# Set up logging
log_dir = "/data/SWATGenXApp/codes/web_application/logs"
logger = LoggerSetup(log_dir, rewrite=False).setup_logger("CeleryTasks")

@shared_task(bind=True)
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
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise