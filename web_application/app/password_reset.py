from flask import Blueprint, jsonify, current_app, request
from app.models import User
from app.extensions import db, csrf
from itsdangerous import URLSafeTimedSerializer
from itsdangerous import SignatureExpired, BadSignature
import smtplib
from email.mime.text import MIMEText
import logging

# Setup logging
logger = logging.getLogger(__name__)

password_reset_bp = Blueprint('password_reset', __name__)

def send_reset_password_email(user, reset_token):
    """
    Sends a password reset email with a unique token to the specified user.

    Args:
        user: User object containing email and username
        reset_token: The token for password reset

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Define the sender and recipient
        sender = "no-reply@ciwre.msu.edu"
        recipient = user.email

        # Create the email content
        subject = "Password Reset - SWATGenX Application"
        
        # Get the site URL from application config - this will be set based on environment
        site_url = current_app.config.get('SITE_URL')
        logger.info(f"Using site URL for password reset: {site_url}")
        
        # Create a professional reset email with the correct URL format
        body = f"""
Hello {user.username},

We received a request to reset your password for your SWATGenX Application account.
To reset your password, please click on the link below:

{site_url}/reset-password?token={reset_token}

This link will expire in 30 minutes for security reasons.

If you did not request a password reset, please ignore this email or contact support if you have concerns.

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
        logger.info(f"Password reset email sent to {recipient} with site URL: {site_url}")
        return True
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        return False

@password_reset_bp.route('/api/reset-password-request', methods=['POST'])
@csrf.exempt
def request_password_reset():
    """
    Handle request for password reset
    """
    current_app.logger.info("Password reset request received")
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            current_app.logger.warning("Password reset failed: missing email")
            return jsonify({"success": False, "message": "Email is required"}), 400
            
        # Find the user by email
        user = User.query.filter_by(email=email).first()
        
        # Always return success even if email not found (for security)
        if not user:
            current_app.logger.warning(f"Password reset request for non-existent email: {email}")
            return jsonify({
                "success": True, 
                "message": "If your email is registered, you will receive password reset instructions."
            })
        
        # Generate reset token
        reset_token = user.get_reset_token(expires_sec=1800)  # 30 minutes
        
        # Send reset email
        email_sent = send_reset_password_email(user, reset_token)
        
        if email_sent:
            current_app.logger.info(f"Password reset email sent to {email}")
            return jsonify({
                "success": True, 
                "message": "Password reset instructions have been sent to your email."
            })
        else:
            current_app.logger.error(f"Failed to send password reset email to {email}")
            return jsonify({
                "success": False, 
                "message": "Failed to send reset email. Please try again later."
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Password reset request error: {e}")
        return jsonify({
            "success": False, 
            "message": "An error occurred during the password reset request."
        }), 500

@password_reset_bp.route('/api/reset-password', methods=['POST'])
@csrf.exempt
def reset_password():
    """
    Handle actual password reset with token validation
    """
    current_app.logger.info("Password reset attempt received")
    try:
        data = request.get_json()
        token = data.get('token', '')
        new_password = data.get('password', '')
        
        if not token or not new_password:
            current_app.logger.warning("Password reset failed: missing token or new password")
            return jsonify({
                "success": False, 
                "message": "Reset token and new password are required"
            }), 400
        
        # Verify the token
        user = User.verify_token(token, expires_sec=1800, salt='password-reset')
        
        if not user:
            current_app.logger.warning(f"Password reset failed: invalid or expired token")
            return jsonify({
                "success": False, 
                "message": "Invalid or expired reset token. Please request a new reset link."
            }), 400
        
        # Update the password
        user.set_password(new_password)
        db.session.commit()
        
        current_app.logger.info(f"Password reset successful for user {user.username}")
        return jsonify({
            "success": True, 
            "message": "Your password has been updated successfully. You can now log in with your new password."
        })
        
    except Exception as e:
        current_app.logger.error(f"Password reset error: {e}")
        db.session.rollback()  # Roll back any failed transaction
        return jsonify({
            "success": False, 
            "message": "An error occurred during password reset."
        }), 500