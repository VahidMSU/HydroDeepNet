from app.extensions import db
from app.extensions import bcrypt
from flask_login import UserMixin

class User(db.Model, UserMixin):
    
    """ User Model for storing user related details """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    email = db.Column(db.String(120), unique=True, nullable=False)

    # Add these fields
    is_verified = db.Column(db.Boolean, default=True)
    verification_code = db.Column(db.String(8), nullable=True)

    @property
    def password(self):
        """Prevent reading raw password."""
        raise AttributeError("password is not a readable attribute.")

    @password.setter
    def password(self, plain_password):
        """Generate a password hash and store it."""
        self.password_hash = bcrypt.generate_password_hash(plain_password).decode('utf-8')

    def check_password(self, plain_password):
        """Compare given password with the stored hash."""
        return bcrypt.check_password_hash(self.password_hash, plain_password)

class ContactMessage(db.Model):
    
    """ Contact Message Model for storing contact messages """

    __tablename__ = 'contact_message'
    __table_args__ = {'extend_existing': True}  # Add this line for ContactMessage as well
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
