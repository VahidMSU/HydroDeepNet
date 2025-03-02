import os
import sys
from flask_socketio import SocketIO
from app import create_app
from celery_app import celery

# Ensure system paths are properly set
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')  # Add SWATGenX to sys.path
sys.path.append('/data/SWATGenXApp/codes/AI_agent')  # Add AI_agent to sys.path


# Configure Matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Environment variables for Flask
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'run.py'

# Set up the logs directory
LOG_DIR = "/data/SWATGenXApp/codes/web_application/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize the Flask app
app = create_app()

# Configure static files path
app.static_folder = os.path.join(os.path.dirname(__file__), 'frontend', 'build', 'static')
app.static_url_path = '/static'

# Set up SocketIO with CORS allowed
socketio = SocketIO(app, cors_allowed_origins="*")

# Log that the Flask application has been created
app.logger.info("Flask application initialized for local SocketIO server.")

# Run the server
if __name__ == '__main__':
    # Start the Flask-SocketIO server
    print("Starting Flask-SocketIO server on port 5050")
    socketio.run(app, host='0.0.0.0', port=5050)
