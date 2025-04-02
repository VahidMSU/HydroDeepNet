import smtplib
from email.mime.text import MIMEText
import uuid
import socket
import time
import logging
from redis import Redis, ConnectionError, exceptions
from flask import current_app

logger = logging.getLogger(__name__)

# Email functionality (from emailex.py)
def send_verification_email(user, verification_code):
    """
    Sends a verification email with a verification code to the specified user.

    Args:
        user: User object containing email and username
        verification_code: The code for email verification

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Define the sender and recipient
        sender = "no-reply@ciwre.msu.edu"
        recipient = user.email

        # Create the email content
        subject = "Verify Your Account - SWATGenX Application"
        
        # Create a more professional verification email with code
        body = f"""
Hello {user.username},

Thank you for registering an account with SWATGenX Application. 
To verify your email address, please use the following verification code:

{verification_code}

Enter this code on the verification page to complete your registration.

If you did not register for this account, please ignore this email.

This is an automated message, please do not reply.
        """
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient

        # Define the SMTP server and port
        smtp_server = "express.mail.msu.edu"
        smtp_port = 25

        # Send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(sender, [recipient], msg.as_string())
        logger.info(f"Verification email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

def send_model_completion_email(username, email, site_no, model_info=None):
    """
    Sends an email notification when a model creation is complete.
    
    Args:
        username (str): The username of the user who created the model
        email (str): The email address of the recipient
        site_no (str): The site number/ID of the created model
        model_info (dict, optional): Additional model information
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Define the sender
    sender = "no-reply@ciwre.msu.edu"
    
    try:
        from app.model_utils import check_model_completion, check_qswat_model_files, check_meterological_data
        from app.utils import find_VPUID
    except ImportError:
        # When called outside Flask application context
        from model_utils import check_model_completion, check_qswat_model_files, check_meterological_data
        from utils import find_VPUID

    swat_model_exe_flag, swat_message = check_model_completion(username, site_no)
    qswat_plus_outputs_flag, qswat_message = check_qswat_model_files(username, site_no)
    meterological_data_flag, met_message = check_meterological_data(username, site_no)

    # Create the email content
    subject = f"SWAT Model Creation Complete - Site {site_no}"
    
    # Create a more informative message body
    body = f"""
Hello {username},

Your SWAT model for site {find_VPUID(site_no)}/huc12/{site_no} has been created and is now available.

Model Status Check Results:
1. SWAT Model Execution: {"✅ Successful" if swat_model_exe_flag else "❌ Failed"} - {swat_message}
2. QSWAT+ Processing: {"✅ Successful" if qswat_plus_outputs_flag else "❌ Failed"} - {qswat_message}
3. Meteorological Data: {"✅ Successful" if meterological_data_flag else "❌ Failed"} - {met_message}

Overall Status: {"✅ All checks passed" if all([swat_model_exe_flag, qswat_plus_outputs_flag, meterological_data_flag]) else "⚠️ Some checks failed"}

You can access your model files through the User Dashboard in the HydroDeepNet application.

Thank you for using HydroDeepNet!

This is an automated message, please do not reply.
"""
    
    # Add model info if provided
    if model_info:
        body += "\n\nModel Details:\n"
        for key, value in model_info.items():
            body += f"{key}: {value}\n"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = email
    
    # Define the SMTP server and port
    smtp_server = "express.mail.msu.edu"
    smtp_port = 25
    
    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(sender, [email], msg.as_string())
        logger.info(f"Model completion email sent successfully to {email}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send model completion email: {e}")
        return False

def send_model_start_email(username, email, site_no, model_info=None, task_id=None):
    """
    Sends an email notification when a model creation task is initiated.
    
    Args:
        username (str): The username of the user who created the model
        email (str): The email address of the recipient
        site_no (str): The site number/ID of the model being created
        model_info (dict, optional): Additional model information
        task_id (str, optional): The Celery task ID for reference
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Define the sender
    sender = "no-reply@ciwre.msu.edu"
    
    # Create the email content
    subject = f"SWAT Model Creation Started - Site {site_no}"
    try:
        from app.utils import find_VPUID
    except ImportError:
        # When called outside Flask application context
        from utils import find_VPUID

    # Create a message body
    body = f"""
Hello {username},

Your SWAT model creation for site {username}/{find_VPUID(site_no)}/huc12/{site_no} has been started and is now being processed.

This process can take some time to complete. You will receive another email when the model creation is finished.

Thank you for using HydroDeepNet!

This is an automated message, please do not reply.
"""
    
    # Add model info if provided
    if model_info:
        body += "\n\nModel Details:\n"
        for key, value in model_info.items():
            body += f"{key}: {value}\n"
    
    # Add task ID if provided
    if task_id:
        body += f"\nTask ID: {task_id}\n"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = email
    
    # Define the SMTP server and port
    smtp_server = "express.mail.msu.edu"
    smtp_port = 25
    
    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(sender, [email], msg.as_string())
        logger.info(f"Model start email sent successfully to {email}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send model start email: {e}")
        return False

# Redis utilities (from redis_utils.py)
def check_redis_health():
    """
    Check if Redis is available and healthy.
    Returns a dict with status information.
    """
    redis_urls = [
        current_app.config.get('REDIS_URL', 'redis://localhost:6379/0'),
        'redis://127.0.0.1:6379/0',  # Try explicit IP
        'redis://localhost:6379/0'   # Try explicit hostname
    ]
    
    # Try each URL until one works, with special handling for loading state
    for url in redis_urls:
        try:
            logger.info(f"Testing Redis health: {url}")
            client = Redis.from_url(
                url, 
                socket_timeout=2,
                socket_connect_timeout=2
            )
            
            # Special handling for Redis loading state
            max_loading_retries = 3
            for loading_attempt in range(max_loading_retries):
                try:
                    response = client.ping()
                    if response:
                        logger.info(f"Redis health check successful: {url}")
                        return {
                            'healthy': True,
                            'message': f'Redis is available at {url}',
                            'working_url': url
                        }
                except exceptions.BusyLoadingError:
                    logger.warning(f"Redis at {url} is loading dataset (attempt {loading_attempt+1}/{max_loading_retries})")
                    if loading_attempt < max_loading_retries - 1:
                        # Wait for Redis to finish loading, increasing delay each time
                        time.sleep(2 * (loading_attempt + 1))
                    else:
                        # Last attempt failed
                        return {
                            'healthy': False,
                            'message': 'Redis is still loading the dataset. Please try again later.',
                            'temporary': True
                        }
                
        except ConnectionError as e:
            logger.warning(f"Redis connection error at {url}: {e}")
        except socket.error as e:
            logger.warning(f"Socket error connecting to Redis at {url}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error checking Redis at {url}: {e}")
    
    # None of the URLs worked
    logger.error("All Redis connection attempts failed")
    return {
        'healthy': False,
        'message': 'Could not connect to Redis server. The service may be down or unreachable.'
    }

def get_working_redis_connection():
    """
    Try to establish a working Redis connection.
    Returns a Redis client or None if no connection works.
    """
    health_check = check_redis_health()
    if health_check['healthy']:
        try:
            return Redis.from_url(
                health_check['working_url'],
                socket_timeout=5,
                socket_connect_timeout=5
            )
        except Exception as e:
            logger.error(f"Error creating Redis client: {e}")
    
    return None