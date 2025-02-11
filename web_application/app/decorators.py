from functools import wraps
from flask import current_app, jsonify
from flask_login import current_user

def maybe_private_mode_requires_login():
    return current_app.config.get('PRIVATE_MODE', True)

def conditional_login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if maybe_private_mode_requires_login() and not current_user.is_authenticated:
            return jsonify({"error": "Login required"}), 401
        return f(*args, **kwargs)
    return wrapper

def conditional_verified_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if maybe_private_mode_requires_login():
            if not current_user.is_authenticated:
                return jsonify({"error": "Login required"}), 401
            if not current_user.is_verified:
                return jsonify({"error": "Verification required"}), 403
        return f(*args, **kwargs)
    return wrapper
