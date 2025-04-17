import os
import sys
from flask import jsonify, request
from app import create_app, socketio

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'run.py'

# Expose app early for CLI tools
app = create_app()

# === Migration ===
def run_oauth_migration():
    try:
        from migrations.add_oauth_columns import run_migration
        if run_migration():
            print("✅ Successfully applied database migrations for OAuth columns")
        else:
            print("⚠️ Warning: Failed to apply database migrations")
    except Exception as e:
        print(f"❌ Migration error: {e}")

# === Error Handlers ===
@app.errorhandler(Exception)
def handle_exception(e):
    if request.path.startswith('/api/'):
        app.logger.error(f"API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), getattr(e, 'code', 500)
    return e

@app.route('/model-settings', methods=['GET', 'POST'])
def model_settings_fallback():
    app.logger.warning(f"Direct access to model-settings route: {request.method}")
    if request.method == 'POST':
        try:
            return app.view_functions['model.model_settings']()
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Invalid method"}), 405

# === Entry Point ===
def main():
    run_oauth_migration()
    app.logger.info("Flask application initialized for local SocketIO server.")
    socketio.run(app, host='0.0.0.0', port=5050)

if __name__ == '__main__':
    main()
