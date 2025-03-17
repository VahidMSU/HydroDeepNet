import os
import sys
from app import create_app, socketio  # Import the socketio instance from app module


# Configure Matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Environment variables for Flask
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'run.py'

# Initialize the Flask app
app = create_app()

# Log that the Flask application has been created
app.logger.info("Flask application initialized for local SocketIO server.")

# Run the server
if __name__ == '__main__':
    # Start the Flask-SocketIO server
    print("Starting Flask-SocketIO server on port 5050")
    socketio.run(app, host='0.0.0.0', port=5050)
