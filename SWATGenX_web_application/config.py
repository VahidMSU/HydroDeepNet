import os
from datetime import timedelta

class Config:
    BASE_PATH = os.getenv('BASE_PATH', '/data2/MyDataBase/SWATGenXAppData/')
    USGS_PATH = os.getenv('USGS_PATH', '/data2/MyDataBase/SWATGenXAppData/USGS')
    
    # Read the secret key from a file or environment variable
    with open('/home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/ciwre-bae-crs/ciwre-bae.campusad.msu.edu.key') as f:
        SECRET_KEY = f.read().strip()
        
    # OR use environment variable for SECRET_KEY
    # SECRET_KEY = os.getenv('SECRET_KEY', 'your-default-secret-key')
    
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)  # Extend session lifetime as needed
    
    SESSION_COOKIE_SECURE = True  # Ensure cookies are only sent over HTTPS
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True  # Mitigate XSS attacks
    
    # For debugging/production control
    DEBUG = os.getenv('FLASK_DEBUG', False)
    TESTING = os.getenv('FLASK_TESTING', False)
