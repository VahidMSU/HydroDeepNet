import os
from datetime import timedelta  # Add this import

class Config:
    BASE_PATH = os.getenv('BASE_PATH', '/data/MyDataBase/SWATGenXAppData/')
    USGS_PATH = os.getenv('USGS_PATH', '/data/MyDataBase/SWATGenXAppData/USGS')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=10)  # Now timedelta is defined
