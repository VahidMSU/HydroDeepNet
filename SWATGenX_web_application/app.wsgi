import sys
import logging
import os
from app import create_app

# Set environment variable for Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'

# Add the application path to the system path
sys.path.insert(0, "/home/rafieiva/MyDataBase/codebase/SWATGenX_web_application")

# Create the application object
application = create_app()

# Set up logging to a writable directory (e.g., /tmp or a folder in your app)
logging.basicConfig(filename='/home/rafieiva/MyDataBase/codebase/SWATGenX_web_application/myapp.log', level=logging.DEBUG)
