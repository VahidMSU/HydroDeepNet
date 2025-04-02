from app.extensions import db
from app.extensions import bcrypt
from flask_login import UserMixin
import time
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask import current_app
import secrets
import string
from sqlalchemy.exc import OperationalError

class User(db.Model, UserMixin):
    """ User Model for storing user related details """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    email = db.Column(db.String(120), unique=True, nullable=False)

    # User verification field
    is_verified = db.Column(db.Boolean, default=False)  # Set default to False to require verification
    verification_code = db.Column(db.String(8), nullable=True)
    
    # Define a class-level flag to track if we have enhanced columns
    _has_enhanced_security = None

    @property
    def password(self):
        """Prevent reading raw password."""
        raise AttributeError("password is not a readable attribute.")

    def set_password(self, plain_password):
        """Generate a password hash and store it."""
        self.password_hash = bcrypt.generate_password_hash(plain_password).decode('utf-8')
        
        # Try to update the enhanced security fields, but ignore errors if they don't exist
        try:
            if User._has_enhanced_security is None:
                # Check if these fields exist without querying
                User._has_enhanced_security = self._check_enhanced_security()
                
            if User._has_enhanced_security:
                self.last_password_change = time.time()
                self.failed_login_attempts = 0
                self.account_locked_until = None
        except Exception:
            pass
        
        # Clear the plain password from memory immediately
        plain_password = None

    def check_password(self, plain_password):
        """
        Compare given password with the stored hash.
        Also implements account lockout if enhanced security is available.
        """
        # Initialize enhanced security check only once
        if User._has_enhanced_security is None:
            User._has_enhanced_security = self._check_enhanced_security()
            
        # Check if account is locked - only if enhanced security is available
        if User._has_enhanced_security:
            try:
                if self.account_locked_until and time.time() < self.account_locked_until:
                    current_app.logger.warning(f"Login attempt for locked account: {self.username}")
                    return False
            except Exception:
                pass
            
        # Verify the password
        is_valid = bcrypt.check_password_hash(self.password_hash, plain_password)
        
        # Immediately clear the password from memory
        plain_password = None
        
        # Handle failed login tracking only if enhanced security is available
        if User._has_enhanced_security and hasattr(self, 'failed_login_attempts'):
            try:
                if is_valid:
                    # Reset failed attempts on successful login
                    self.failed_login_attempts = 0
                else:
                    # Increment failed attempts
                    if self.failed_login_attempts is None:
                        self.failed_login_attempts = 1
                    else:
                        self.failed_login_attempts += 1
                    
                    # Lock account after 5 failed attempts
                    if self.failed_login_attempts >= 5:
                        # Lock for 15 minutes (900 seconds)
                        self.account_locked_until = time.time() + 900
                        current_app.logger.warning(f"Account locked for user: {self.username} due to too many failed attempts")
                
                db.session.commit()
            except Exception as e:
                current_app.logger.warning(f"Could not update login attempt tracking: {e}")
            
        return is_valid
    
    @classmethod
    def _check_enhanced_security(cls):
        """Check if the enhanced security columns exist in the database"""
        try:
            # Try to query a user with selection of one of the new columns
            # We don't need the result, just to check if the column exists
            db.session.query(User.id, User.last_password_change).limit(1).all()
            return True
        except OperationalError:
            current_app.logger.info("Enhanced security columns not found in database. Using basic authentication.")
            return False
        except Exception as e:
            current_app.logger.warning(f"Error checking for enhanced security columns: {e}")
            return False

    def get_verification_token(self, expires_sec=3600):
        """Generate a secure token for email verification."""
        # Generate and store a verification code
        self.verification_code = self.generate_random_code(8)
        db.session.commit()
        return self.verification_code

    def get_reset_token(self, expires_sec=1800):
        """Generate a secure token for password reset."""
        s = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id}, salt='password-reset')

    @staticmethod
    def verify_token(token, expires_sec=3600, salt='email-verification'):
        """Verify a token and return the user."""
        # First check if it's a verification code (8 characters)
        if token and len(token) == 8:
            user = User.query.filter_by(verification_code=token).first()
            if user:
                return user
                
        # Fall back to token-based verification for backward compatibility
        s = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token, salt=salt, max_age=expires_sec)
            user_id = data.get('user_id')
            if not user_id:
                return None
            return User.query.get(user_id)
        except (SignatureExpired, BadSignature):
            return None

    @staticmethod
    def generate_random_code(length=8):
        """Generate a random verification code."""
        characters = string.ascii_letters + string.digits
        return ''.join(secrets.choice(characters) for _ in range(length))


class ContactMessage(db.Model):
    """ Contact Message Model for storing contact messages """

    __tablename__ = 'contact_message'
    __table_args__ = {'extend_existing': True}  # Add this line for ContactMessage as well
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
