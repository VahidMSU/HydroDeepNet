##/data/SWATGenXApp/codes/web_application/app/routes.py
from flask import current_app, request, redirect
from app.extensions import csrf
from app.utils import LoggerSetup
import os
# Import consolidated blueprints
from app.user_auth import user_auth_bp
from app.model import model_bp
from app.viz_report import viz_report_bp
from app.debug import debug_bp, register_debug_routes
from app.static import static_bp
from app.hydrogeo import hydrogeo_bp
from app.chatbot import chatbot_bp

class AppManager:
    def __init__(self, app):
        self.app = app
        log_dir = "/data/SWATGenXApp/codes/web_application/logs"
        self.app.logger = LoggerSetup(log_dir, rewrite=False).setup_logger("FlaskApp")
        self.app.logger.info("AppManager initialized!")

        # Add route normalization before registering blueprints
        self.setup_route_normalization()
        
        # Set up CSRF exclusions for API routes
        self.setup_csrf_exclusion()
        
        # Register routes
        self.init_routes()
        
    def setup_route_normalization(self):
        """Set up a before_request handler to normalize routes between dev and prod."""
        is_prod = os.environ.get('FLASK_ENV') == 'production'
        self.app.logger.info(f"Setting up route normalization in {'production' if is_prod else 'development'} mode")
        
        @self.app.before_request
        def normalize_route():
            """
            Handle route differences between development and production.
            
            In development, Flask directly handles all routes
            In production, Apache proxies requests and may use different prefixes
            """
            # Skip for static files
            if request.path.startswith('/static/'):
                return None
                
            # Skip for API routes that already have the /api prefix
            if request.path.startswith('/api/'):
                return None
                
            # Skip for download routes
            if request.path.startswith('/download/') or request.path.startswith('/download_directory/'):
                return None
                
            # Handle key paths that need special handling in production
            paths_needing_api_prefix = [
                '/model-settings', 
                '/model-confirmation',
                '/vision_system',
                '/reports'
            ]
            
            # In development, if a non-prefixed route is requested, 
            # check if there's an API version and redirect if needed
            if is_prod and request.path in paths_needing_api_prefix:
                api_path = f"/api{request.path}"
                self.app.logger.info(f"Normalizing route: {request.path} -> {api_path}")
                return redirect(api_path, code=307)  # 307 preserves the HTTP method
                
            return None

    def setup_csrf_exclusion(self):
        """Exclude API routes from CSRF protection to prevent issues with frontend calls."""
        try:
            from app.extensions import csrf
            
            # Register a proper CSRF exempt handler for API routes that returns a response
            @self.app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
            @csrf.exempt
            def csrf_exempt_api(path):
                # Forward the request to the appropriate route
                self.app.logger.debug(f"CSRF exempt route for: /api/{path}")
                # This is just a pass-through for CSRF exemption - we shouldn't actually get here
                # Return a proper response instead of None
                return redirect(f"/api/{path}", code=307)
            
            self.app.logger.info("Set up CSRF exclusion for API routes")
            
            # Alternative approach: use direct blueprint exemption
            # For each blueprint with 'api' in its name, exempt all its endpoints
            for blueprint_name, blueprint in self.app.blueprints.items():
                if 'api' in blueprint_name.lower() or 'debug' in blueprint_name.lower():
                    csrf.exempt(blueprint)
                    self.app.logger.info(f"Exempted blueprint {blueprint_name} from CSRF protection")
            
        except ImportError as e:
            self.app.logger.warning(f"Could not set up CSRF exclusion: {e}")
        except Exception as e:
            self.app.logger.error(f"Error setting up CSRF exclusion: {e}")

    def init_routes(self):
        """Register all blueprints."""
        # Register consolidated blueprints
        self.app.register_blueprint(user_auth_bp)
        self.app.register_blueprint(model_bp)
        self.app.register_blueprint(viz_report_bp)
        self.app.register_blueprint(static_bp)
        self.app.register_blueprint(hydrogeo_bp)
        self.app.register_blueprint(chatbot_bp)
        
        # Register debug routes conditionally
        register_debug_routes(self.app)
        
        self.app.logger.info("All blueprints registered")

