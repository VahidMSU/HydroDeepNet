from flask import Flask
from flask_wtf.csrf import CSRFProtect
from config import Config
from app.extensions import db, login_manager
from app.models import User  # Ensure this is imported after db is initialized
import sys
from app.utils import LoggerSetup
sys.path.append(r'/data/SWATGenXApp/codes/SWATGenX')

from flask_talisman import Talisman  # Import Flask-Talisman

from flask import Flask
from app.extensions import db, login_manager
from app.sftp_routes import sftp_bp  # Import the SFTP blueprint



def create_app():
    logger = LoggerSetup("/data/SWATGenXApp/codes/web_application/logs", rewrite=True)
    logger = logger.setup_logger("app")
    logger.info("Creating app")

    app = Flask(__name__, static_url_path='/static', static_folder='/data/MyDataBase')

    # Apply security configurations
    app.config.from_object(Config)
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['REMEMBER_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # ✅ Apply Flask-Talisman with HSTS enabled
    Talisman(
        app,
        force_https=True,  # Redirect all HTTP to HTTPS
        strict_transport_security=True,  # Enables HSTS
        strict_transport_security_max_age=31536000,  # HSTS duration (1 year)
        strict_transport_security_include_subdomains=True,  # Apply to subdomains
        strict_transport_security_preload=True,  # Enable HSTS Preloading
        content_security_policy=None  # Disable CSP blocking for now
    )

    # CSRF Protection
    csrf = CSRFProtect(app)

    # Initialize extensions
    db.init_app(app)

    app.register_blueprint(sftp_bp, url_prefix="/api")  # 🔹 Register with /api prefix

    login_manager.init_app(app)
    login_manager.login_view = 'login'

    
    @login_manager.user_loader
    def load_user(user_id):
        logger.info(f"User ID: {user_id}") 
        return User.query.get(int(user_id))

    with app.app_context():
        logger.info("Creating all tables")
        db.create_all()

    # Import routes
    from app.routes import AppManager
    hydro_geo_app = AppManager(app)
    logger.info("Initializing routes")

    return app
