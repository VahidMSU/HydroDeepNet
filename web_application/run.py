import os
import sys
from flask import jsonify, request
from app import create_app, socketio  # Import the socketio instance from app module

# Configure Matplotlib cache directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Environment variables for Flask
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'run.py'

# Run database migration to add OAuth columns if needed
try:
    from migrations.add_oauth_columns import run_migration
    migration_result = run_migration()
    if migration_result:
        print("Successfully applied database migrations for OAuth columns")
    else:
        print("Warning: Failed to apply database migrations")
except Exception as e:
    print(f"Error running database migrations: {e}")
    print("Continuing anyway...")

# Initialize the Flask app
app = create_app()

# Log that the Flask application has been created
app.logger.info("Flask application initialized for local SocketIO server.")

# Ensure all API errors return JSON responses
@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON instead of HTML for API errors."""
    if request.path.startswith('/api/'):
        app.logger.error(f"API error: {str(e)}")
        # Get the error code if it exists, otherwise use 500
        code = 500
        if hasattr(e, 'code'):
            code = e.code
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), code
    
    # Re-raise non-API errors
    return e

# Add a specific error handler for model-settings route
@app.route('/model-settings', methods=['GET', 'POST'])
def model_settings_fallback():
    """
    Fallback handler for direct model-settings requests that might bypass the normal route
    """
    app.logger.warning(f"Direct access to model-settings route detected: {request.method}")
    
    if request.method == 'POST':
        try:
            return app.view_functions['model.model_settings']()
        except Exception as e:
            app.logger.error(f"Error in model settings fallback: {e}")
            return jsonify({
                "status": "error",
                "message": "An error occurred processing your request",
                "details": str(e)
            }), 500
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid request method for this endpoint" 
        }), 405

# Run the server
if __name__ == '__main__':
    # Start the Flask-SocketIO server
    print("Starting Flask-SocketIO server on port 5050")
    socketio.run(app, host='0.0.0.0', port=5050)
