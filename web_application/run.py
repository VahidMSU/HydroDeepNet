from flask_socketio import SocketIO
import os
import sys
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

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

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Expose `application` for Apache WSGI
application = app  

if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server on port 5050")
    socketio.run(app, host='0.0.0.0', port=5050)
