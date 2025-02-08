from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
    CORS(app)
    return app
