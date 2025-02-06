import os
import sys
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from waitress import serve

# Set up system paths
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')

# Import necessary modules
from app import create_app
from SWATGenX.SWATGenXLogging import LoggerSetup

# Set up logging
LOG_DIR = "/data/SWATGenXApp/codes/web_application/logs"
logger = LoggerSetup(LOG_DIR, rewrite=False).setup_logger("FlaskApp")

# Ensure Matplotlib cache directory exists
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Create Flask app
app = create_app()
logger.info("Flask application created")

# Apply proxy middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Expose `application` for Apache WSGI
application = app  

# Ensure there are no indentation errors here
if __name__ == '__main__':
    logger.info("Starting Waitress server on port 5050")
    serve(app, host='0.0.0.0', port=5050)