import smtplib
from email.mime.text import MIMEText
import uuid

def send_verification_email(recipient):
    """
    Sends a verification email with a unique code to the specified recipient.

    Args:
        recipient (str): The email address of the recipient.

    Returns:
        str: The verification code sent in the email.
    """
    # Define the sender and recipient
    sender = "no-reply@ciwre.msu.edu"

    # Create the email content
    subject = "Verification Email"
    verification_code = str(uuid.uuid4().int)[:8]
    body = f"Your verification code is: {verification_code}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Define the SMTP server and port
    smtp_server = "express.mail.msu.edu"
    smtp_port = 25

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(sender, [recipient], msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

    return verification_code

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
    
    # Create the email content
    subject = f"SWAT Model Creation Complete - Site {site_no}"
    
    # Create a more informative message body
    body = f"""
Hello {username},

Your SWAT model for site {site_no} has been successfully created and is now available.

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
        print(f"Model completion email sent successfully to {email}.")
        return True
    except Exception as e:
        print(f"Failed to send model completion email: {e}")
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
    
    # Create a message body
    body = f"""
Hello {username},

Your SWAT model creation for site {site_no} has been started and is now being processed.

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
        print(f"Model start email sent successfully to {email}.")
        return True
    except Exception as e:
        print(f"Failed to send model start email: {e}")
        return False

if __name__ == "__main__":
    recipient = "rafieiva@msu.edu"
    send_verification_email(recipient)