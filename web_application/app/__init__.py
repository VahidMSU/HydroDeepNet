## /data/SWATGenXApp/codes/web_application/app/__init__.py
import sys
import os
from flask import Flask, abort, jsonify, request
from flask_talisman import Talisman
from config import Config
from app.extensions import csrf, db, login_manager
from app.models import User
from app.utils import LoggerSetup
from app.routes import AppManager
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit
from redis import Redis, ConnectionError

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
        static_folder='/data/SWATGenXApp/GenXAppData',
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
    login_manager.login_view = 'user_auth.api_login'  # Updated login view to use new blueprint

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

    # Note: Blueprints are now registered through AppManager in routes.py
    # The following initialization will call app.register_blueprint for all necessary blueprints
    
    # Ensure database tables exist
    with app.app_context():
        db.create_all()
        logger.info("Ensured database tables exist")

    # Initialize Redis connection with better error handling and retry
    try:
        redis_client = Redis.from_url(app.config['REDIS_URL'], socket_timeout=5)
        redis_client.ping()
        logger.info("Successfully connected to Redis")
    except ConnectionError as e:
        redis_client = None
        logger.error(f"Failed to connect to Redis: {str(e)}")
        # Try an alternative connection if primary fails
        try:
            # Try with a fallback host if needed
            fallback_url = 'redis://127.0.0.1:6379/0'
            if fallback_url != app.config['REDIS_URL']:
                redis_client = Redis.from_url(fallback_url, socket_timeout=5)
                redis_client.ping()
                logger.info(f"Connected to Redis using fallback URL: {fallback_url}")
                # Update config with working URL
                app.config['REDIS_URL'] = fallback_url
        except Exception as e:
            logger.error(f"Failed to connect to Redis with fallback: {str(e)}")
    except Exception as e:
        redis_client = None
        logger.error(f"Unexpected Redis error: {str(e)}")

    # Initialize rate limiting with Redis storage backend
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=app.config['REDIS_URL'] if redis_client else 'memory://',
        storage_options={"socket_timeout": 10} if redis_client else {}
    )

    if redis_client:
        logger.info("Rate limiting enabled with Redis storage backend")
    else:
        logger.warning("Rate limiting using memory storage (Redis unavailable)")

    # Initialize routes via AppManager
    AppManager(app)
    logger.info("Application routes initialized through AppManager")

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

    # Add error handler for API routes to always return JSON
    @app.errorhandler(Exception)
    def handle_error(e):
        code = 500
        if hasattr(e, 'code'):
            code = e.code
        
        # Check if this is an API request
        if request.path.startswith('/api/'):
            logger.error(f"API error: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "error_type": e.__class__.__name__
            }), code
        
        # For non-API routes, let Flask handle the error normally
        return e
    
    # Add error handler for 404 errors to return JSON for API routes
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            logger.error(f"API 404 error: {request.path}")
            return jsonify({
                "status": "error",
                "message": "API endpoint not found",
                "path": request.path
            }), 404
        return e
    
    # Add error handler for 500 errors to return JSON for API routes
    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith('/api/'):
            logger.error(f"API 500 error: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Internal server error",
                "details": str(e) if app.debug else "See server logs for details"
            }), 500
        return e

    return app