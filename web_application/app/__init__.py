## /data/SWATGenXApp/codes/web_application/app/__init__.py
import sys
import os
from flask import Flask, abort
from flask_talisman import Talisman
from config import Config
from app.extensions import csrf, db, login_manager
from app.models import User
from app.utils import LoggerSetup
from app.sftp_routes import sftp_bp  # Import SFTP API routes
from .api_routes import api_bp  # Import API routes
from app.routes import AppManager
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit
from redis import Redis, ConnectionError
from app.routes_debug import register_debug_routes  # Add this import

# Ensure the system path includes SWATGenX
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
sys.path.append('/data/SWATGenXApp/codes/AI_agent') 

# Create socketio instance at module level to export it
socketio = SocketIO(cors_allowed_origins="*")

def create_app(config_class=Config):  # Update function signature
    """
    Creates and configures the Flask application.
    """
    # Set up logging
    log_dir = "/data/SWATGenXApp/codes/web_application/logs"
    logger = LoggerSetup(log_dir, rewrite=False).setup_logger("FlaskApp")
    logger.info("Initializing Flask application")

    # Initialize Flask app
    app = Flask(
        __name__,
        static_url_path='/static',
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'build', 'static')
    )

    # Clean up CORS configuration - use just one comprehensive configuration
    CORS(
        app,
        supports_credentials=True,
        resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "https://ciwre-bae.campusad.msu.edu"],
            },
            r"/login": {
                "origins": ["http://localhost:3000", "https://ciwre-bae.campusad.msu.edu"],
            },
            r"/signup": {
                "origins": ["http://localhost:3000", "https://ciwre-bae.campusad.msu.edu"],
            },
            r"/model-settings": {
                "origins": ["http://localhost:3000", "https://ciwre-bae.campusad.msu.edu"],
            },
            r"/*": {
                "origins": "*"  # Fallback for other routes
            }
        }
    )

    # Initialize SocketIO with the app
    socketio.init_app(app)
    
#    @socketio.on('connect')
#    def on_connect():
#        logger.info('Client connected')
#        emit('message', {'data': 'Connected to WebSocket!'})

#    @socketio.on('disconnect')
#    def on_disconnect():
#        logger.info('Client disconnected')
#        emit('message', {'data': 'Disconnected from WebSocket!'})

    # Load configurations
    app.config.from_object(config_class)  # Update to use config_class
    app.config.update({
        'SESSION_COOKIE_SECURE': False,  # ✅ Disable HTTPS for local testing
        'REMEMBER_COOKIE_SECURE': False,  # ✅ Disable HTTPS for local testing  
        'SESSION_COOKIE_HTTPONLY': True,
        'SESSION_COOKIE_SAMESITE': 'None',
        'PREFERRED_URL_SCHEME': 'https'
    })
    
    # Set Flask logger to use our custom logger
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)

    # Apply Flask-Talisman for security headers & HTTPS enforcement
    Talisman(
        app,
        force_https=False,  # ✅ Disable HTTPS redirect for local testing
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,
        strict_transport_security_include_subdomains=True,
        strict_transport_security_preload=True,
        content_security_policy=None  # Disable CSP blocking for now
    )

    logger.info("Applied Flask-Talisman security configurations")

    # Apply CSRF Protection
    #csrf.init_app(app)
    logger.info("CSRF protection enabled")

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'api_login'  # Corrected login view

    @login_manager.user_loader
    def load_user(user_id):
        logger.info(f"Loading user: {user_id}")
        return User.query.get(int(user_id))

    # Load test user
    with app.app_context():
        test_user = User.query.get(1)
        if (test_user):
            logger.info(f"Test user loaded: {test_user.username}")
        else:
            logger.warning("Test user with ID 1 not found.")

    # Register Blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    logger.info("Registered API blueprints")    
    
    app.register_blueprint(sftp_bp, url_prefix="/api/sftp")
    logger.info("Registered SFTP blueprints")
    
    # Ensure database tables exist
    with app.app_context():
        db.create_all()
        logger.info("Ensured database tables exist")

    # Import and initialize routes
    AppManager(app)
    logger.info("Application routes initialized")

    # Initialize Redis connection
    try:
        redis_client = Redis.from_url(app.config['REDIS_URL'])
        redis_client.ping()
    except ConnectionError:
        redis_client = None
        logger.error("Failed to connect to Redis")

    # Initialize rate limiting with Redis storage backend
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=app.config['REDIS_URL'] if redis_client else 'memory://'
    )

    logger.info("Rate limiting enabled with Redis storage backend")

    # Function to ensure path is within the base directory
    def secure_path(user_path, allowed_paths):
        abs_target_dir = os.path.abspath(os.path.realpath(user_path))  # Double sanitization

        if not any(abs_target_dir.startswith(os.path.abspath(base)) for base in allowed_paths):
            logger.error(f"Unauthorized path access attempt: {user_path}")
            abort(403, description="Unauthorized path access")
 
        return abs_target_dir

    # Example usage
    allowed_dirs = [
        "/data/SWATGenXApp/codes/web_application",
        "/data/SWATGenXApp/Users",
        "/data/SWATGenXApp/GenXAppData"
    ]
    secure_path("/data/SWATGenXApp/Users", allowed_dirs)

    # Apply secure headers
    @app.after_request
    def apply_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
        response.headers["Cache-Control"] = "no-store"
        return response

    # Register debug routes if in development
    register_debug_routes(app)

    return app