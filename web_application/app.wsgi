import sys
import os
import logging

# Set the Python path for your application
sys.path.insert(0, "/data/SWATGenXApp/codes/web_application")

# Ensure Matplotlib can write temporary files
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'

# Set environment variables for the Flask application
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'app.py'

# Logging setup
log_dir = "/data/SWATGenXApp/logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'myapp.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Starting the Flask application...")

# Import the Flask app
from app import create_app

# Create the WSGI application object
application = create_app()
