import sys
import logging
import os

# Set environment variable for Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'

# Add the application path to the system path
sys.path.insert(0, "/data/SWATGenXApp/codes/web_application")

# Import the create_app function after modifying sys.path
from app import create_app

# Create a directory for logs if it doesn't exist
log_dir = "/data/SWATGenXApp/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Set up logging to a writable directory (e.g., /data/SWATGenXApp/logs)
log_file_path = os.path.join(log_dir, 'myapp.log')
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log a message indicating the application startup
logging.info("Starting the Flask application...")

# Create the application object
application = create_app()
