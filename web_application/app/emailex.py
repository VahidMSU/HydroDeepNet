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

if __name__ == "__main__":
    recipient = "rafieiva@msu.edu"
    send_verification_email(recipient)