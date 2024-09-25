from flask import Flask
from flask_wtf.csrf import CSRFProtect
from config import Config
from app.extensions import db, login_manager
from app.models import User  # Ensure this is imported after db is initialized
from datetime import timedelta

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    csrf = CSRFProtect(app)
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()

    from app.routes import init_routes
    init_routes(app)

    return app
