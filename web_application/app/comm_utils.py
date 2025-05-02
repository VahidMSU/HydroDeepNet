import smtplib
from email.mime.text import MIMEText
import uuid
import socket
import time
import logging
from redis import Redis, ConnectionError, exceptions
from flask import current_app
import ssl

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
        sender = "vahidr32@gmail.com"
        recipient = user.email

        # Debug log the Gmail configuration
        app_password = current_app.config.get('GMAIL_APP_PASSWORD')
        if not app_password:
            logger.error("GMAIL_APP_PASSWORD is not configured")
            return False
        logger.info(f"Using Gmail app password: {app_password[:4]}...{app_password[-4:]}")

        # Create the email content
        subject = "Verify Your Account - HydroDeepNet Application"

        # Create a more professional verification email with code
        body = f"""
Hello {user.username},

Thank you for registering an account with HydroDeepNet Application.
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

        # Define the SMTP server and port for Gmail
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # Send the email using Gmail's SMTP server with enhanced error handling
        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.set_debuglevel(1)  # Enable debug output
                logger.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}")

                # Start TLS
                try:
                    server.starttls(context=ssl.create_default_context())
                    logger.info("TLS connection established")
                except Exception as e:
                    logger.error(f"TLS connection failed: {str(e)}")
                    raise

                # Login
                try:
                    server.login(sender, app_password)
                    logger.info("SMTP login successful")
                except smtplib.SMTPAuthenticationError as e:
                    logger.error(f"SMTP authentication failed: {str(e)}")
                    logger.error("Please verify that:")
                    logger.error("1. The app password is correct")
                    logger.error("2. 2-factor authentication is enabled on the Gmail account")
                    logger.error("3. The app password was generated for the correct Gmail account")
                    raise
                except Exception as e:
                    logger.error(f"SMTP login failed: {str(e)}")
                    raise

                # Send email
                try:
                    server.sendmail(sender, [recipient], msg.as_string())
                    logger.info(f"Verification email sent successfully to {recipient}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to send email: {str(e)}")
                    raise

        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP connection error: {str(e)}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {str(e)}")
            return False
        except socket.timeout as e:
            logger.error(f"SMTP connection timeout: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during email sending: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Failed to send verification email: {str(e)}")
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
    sender = "vahidr32@gmail.com"

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

    # Define the SMTP server and port for Gmail
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Send the email with enhanced error handling
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.set_debuglevel(1)  # Enable debug output
            logger.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}")

            # Start TLS
            try:
                server.starttls(context=ssl.create_default_context())
                logger.info("TLS connection established")
            except Exception as e:
                logger.error(f"TLS connection failed: {str(e)}")
                raise

            # Login
            try:
                server.login(sender, current_app.config['GMAIL_APP_PASSWORD'])
                logger.info("SMTP login successful")
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP authentication failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"SMTP login failed: {str(e)}")
                raise

            # Send email
            try:
                server.sendmail(sender, [email], msg.as_string())
                logger.info(f"Model completion email sent successfully to {email}")
                return True
            except Exception as e:
                logger.error(f"Failed to send email: {str(e)}")
                raise

    except smtplib.SMTPConnectError as e:
        logger.error(f"SMTP connection error: {str(e)}")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {str(e)}")
        return False
    except socket.timeout as e:
        logger.error(f"SMTP connection timeout: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during email sending: {str(e)}")
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
    sender = "vahidr32@gmail.com"

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

    # Define the SMTP server and port for Gmail
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Send the email with enhanced error handling
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.set_debuglevel(1)  # Enable debug output
            logger.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}")

            # Start TLS
            try:
                server.starttls(context=ssl.create_default_context())
                logger.info("TLS connection established")
            except Exception as e:
                logger.error(f"TLS connection failed: {str(e)}")
                raise

            # Login
            try:
                server.login(sender, current_app.config['GMAIL_APP_PASSWORD'])
                logger.info("SMTP login successful")
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP authentication failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"SMTP login failed: {str(e)}")
                raise

            # Send email
            try:
                server.sendmail(sender, [email], msg.as_string())
                logger.info(f"Model start email sent successfully to {email}")
                return True
            except Exception as e:
                logger.error(f"Failed to send email: {str(e)}")
                raise

    except smtplib.SMTPConnectError as e:
        logger.error(f"SMTP connection error: {str(e)}")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {str(e)}")
        return False
    except socket.timeout as e:
        logger.error(f"SMTP connection timeout: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during email sending: {str(e)}")
        return False

# Redis utilities (from redis_utils.py)
def check_redis_health():
    """
    Comprehensive Redis health check with detailed diagnostics.

    Returns:
        dict: Health check results with keys:
            - healthy (bool): Whether Redis is working
            - message (str): Descriptive message
            - url (str): Working Redis URL if healthy
            - details (dict): Additional diagnostic information
    """
    import os
    import time
    import socket
    from redis import Redis, ConnectionError
    from flask import current_app

    start_time = time.time()
    details = {
        "checks_performed": [],
        "errors": [],
        "warnings": []
    }

    # URLs to test in order of preference
    redis_urls = [
        os.environ.get('REDIS_URL', ''),
        'redis://127.0.0.1:6379/0',
        'redis://localhost:6379/0',
        'redis://redis:6379/0'
    ]

    # Filter out empty URLs
    redis_urls = [url for url in redis_urls if url]
    if not redis_urls:
        redis_urls = ['redis://localhost:6379/0']

    details["urls_tested"] = redis_urls

    # Test each URL
    for url in redis_urls:
        try:
            current_app.logger.info(f"Testing Redis health at {url}")
            details["checks_performed"].append(f"Connection to {url}")

            client = Redis.from_url(
                url,
                socket_timeout=3,
                socket_connect_timeout=3,
                health_check_interval=30
            )

            # Basic ping test
            ping_start = time.time()
            ping_result = client.ping()
            ping_time = time.time() - ping_start
            details["checks_performed"].append(f"Ping to {url}")

            if not ping_result:
                details["warnings"].append(f"Redis at {url} ping returned {ping_result} instead of True")

            # Try to set and get a value
            test_key = f"redis_health_check_{time.time()}"
            set_start = time.time()
            client.set(test_key, "1", ex=60)
            set_time = time.time() - set_start

            get_start = time.time()
            test_value = client.get(test_key)
            get_time = time.time() - get_start

            client.delete(test_key)
            details["checks_performed"].append(f"Set/get test on {url}")

            if test_value != b"1":
                details["warnings"].append(f"Redis at {url} returned {test_value} instead of b'1'")

            # Check server info
            info_start = time.time()
            info = client.info()
            info_time = time.time() - info_start
            details["checks_performed"].append(f"Info command on {url}")

            # Record performance metrics
            details["performance"] = {
                "ping_time_ms": round(ping_time * 1000, 2),
                "set_time_ms": round(set_time * 1000, 2),
                "get_time_ms": round(get_time * 1000, 2),
                "info_time_ms": round(info_time * 1000, 2),
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }

            # Record server diagnostics
            details["server_info"] = {
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "total_connections_received": info.get("total_connections_received", 0)
            }

            # If we got here, Redis is working
            current_app.logger.info(f"Redis at {url} is healthy (ping: {ping_time*1000:.2f}ms)")

            # Update the environment variable to use this URL
            if url != os.environ.get('REDIS_URL'):
                os.environ['REDIS_URL'] = url
                current_app.logger.info(f"Updated REDIS_URL to {url}")

            return {
                "healthy": True,
                "message": f"Redis is working properly at {url}",
                "url": url,
                "details": details
            }

        except ConnectionError as e:
            details["errors"].append(f"Connection error to {url}: {str(e)}")
            current_app.logger.warning(f"Redis connection error at {url}: {e}")
            continue

        except socket.timeout as e:
            details["errors"].append(f"Socket timeout to {url}: {str(e)}")
            current_app.logger.warning(f"Redis socket timeout at {url}: {e}")
            continue

        except Exception as e:
            details["errors"].append(f"Unexpected error with {url}: {str(e)}")
            current_app.logger.error(f"Unexpected Redis error at {url}: {e}")
            import traceback
            details["last_exception_traceback"] = traceback.format_exc()
            continue

    # If we get here, all URLs failed
    current_app.logger.error("All Redis URLs failed health check")

    return {
        "healthy": False,
        "message": "Redis is not available on any tested URL",
        "url": None,
        "details": details
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