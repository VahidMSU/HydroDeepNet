from app.extensions import db
from app.extensions import bcrypt
from flask_login import UserMixin
import time
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask import current_app
import secrets
import string
from sqlalchemy.exc import OperationalError, SQLAlchemyError

class User(db.Model, UserMixin):
    """ User Model for storing user related details """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    email = db.Column(db.String(120), unique=True, nullable=False)

    # User verification field
    is_verified = db.Column(db.Boolean, default=False)  # Set default to False to require verification
    verification_code = db.Column(db.String(8), nullable=True)
    
    # OAuth related fields
    oauth_provider = db.Column(db.String(20), nullable=True)  # 'google', 'microsoft', etc.
    oauth_id = db.Column(db.String(100), nullable=True)  # The unique ID from the provider
    oauth_name = db.Column(db.String(100), nullable=True)  # Full name from the provider
    oauth_picture = db.Column(db.String(255), nullable=True)  # Profile picture URL
    
    # Define a class-level flag to track if we have enhanced columns
    _has_enhanced_security = None
    _has_oauth_columns = None

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

    @classmethod
    def _check_oauth_columns(cls):
        """Check if the OAuth columns exist in the database"""
        try:
            # Try to query for OAuth columns
            db.session.query(User.id, User.oauth_provider).limit(1).all()
            return True
        except OperationalError:
            current_app.logger.warning("OAuth columns not found in database. OAuth login will not work.")
            return False
        except Exception as e:
            current_app.logger.warning(f"Error checking for OAuth columns: {e}")
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

    @staticmethod
    def find_or_create_oauth_user(email, provider, provider_id, name=None, picture=None):
        """
        Find an existing user by OAuth ID or email, or create a new one if not found.
        
        Args:
            email: User's email from OAuth provider
            provider: OAuth provider name (e.g., 'google')
            provider_id: Unique user ID from the provider
            name: User's full name from the provider
            picture: URL to user's profile picture
            
        Returns:
            User: The found or created user object
        """
        # Check if OAuth columns exist in the database
        if User._has_oauth_columns is None:
            User._has_oauth_columns = User._check_oauth_columns()
            
        if not User._has_oauth_columns:
            # If OAuth columns don't exist, just find user by email or create a new one
            try:
                user = User.query.filter_by(email=email).first()
                
                # Create a new user if not found
                if not user:
                    # Generate a unique username based on the email
                    base_username = email.split('@')[0]
                    username = base_username
                    counter = 1
                    
                    # Ensure username is unique
                    while User.query.filter_by(username=username).first():
                        username = f"{base_username}{counter}"
                        counter += 1
                        
                    # Create the new user
                    user = User(
                        username=username,
                        email=email,
                        is_verified=True  # OAuth users are automatically verified
                    )
                    db.session.add(user)
                    db.session.commit()
                
                return user
            except Exception as e:
                current_app.logger.error(f"Error finding/creating user by email: {e}")
                raise
        
        # Normal OAuth flow if columns exist
        try:
            # Try to find by OAuth provider and ID first
            user = User.query.filter_by(oauth_provider=provider, oauth_id=provider_id).first()
            
            # If not found, try to find by email
            if not user:
                user = User.query.filter_by(email=email).first()
                
            # Create a new user if still not found
            if not user:
                # Generate a unique username based on the email
                base_username = email.split('@')[0]
                username = base_username
                counter = 1
                
                # Ensure username is unique
                while User.query.filter_by(username=username).first():
                    username = f"{base_username}{counter}"
                    counter += 1
                    
                # Create the new user
                user = User(
                    username=username,
                    email=email,
                    oauth_provider=provider,
                    oauth_id=provider_id,
                    oauth_name=name,
                    oauth_picture=picture,
                    is_verified=True  # OAuth users are automatically verified
                )
                db.session.add(user)
                db.session.commit()
            
            # Update OAuth details for existing users that might be logging in with OAuth for the first time
            if user and (user.oauth_provider != provider or user.oauth_id != provider_id):
                user.oauth_provider = provider
                user.oauth_id = provider_id
                user.oauth_name = name or user.oauth_name
                user.oauth_picture = picture or user.oauth_picture
                user.is_verified = True  # Ensure OAuth users are verified
                db.session.commit()
                
            return user
        except SQLAlchemyError as e:
            current_app.logger.error(f"Database error in find_or_create_oauth_user: {e}")
            raise
        except Exception as e:
            current_app.logger.error(f"Unexpected error in find_or_create_oauth_user: {e}")
            raise


class ContactMessage(db.Model):
    """ Contact Message Model for storing contact messages """

    __tablename__ = 'contact_message'
    __table_args__ = {'extend_existing': True}  # Add this line for ContactMessage as well
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
