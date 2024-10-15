from flask import Flask
from flask_wtf.csrf import CSRFProtect
from config import Config
from app.extensions import db, login_manager
from app.models import User  # Ensure this is imported after db is initialized

def create_app():
    print("Creating app")
    app = Flask(__name__)
    app.config.from_object(Config)

    # Cookie Settings for Secure Production
    app.config['SESSION_COOKIE_SECURE'] = True  # Cookies only over HTTPS
    app.config['REMEMBER_COOKIE_SECURE'] = True  # Secure "Remember me" cookie
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    print("App created")
    csrf = CSRFProtect(app)
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    @login_manager.user_loader
    def load_user(user_id):
        print(f"User ID: {user_id}")    
        return User.query.get(int(user_id))

    with app.app_context():
        print("Creating all tables")    
        db.create_all()

    from app.routes import init_routes
    print("Initializing routes")    
    init_routes(app)

    return app
